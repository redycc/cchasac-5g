"""
cc-HASAC v6 — Behavioural-Cloning (BC) warm-start + encoder freeze.

ROOT CAUSE of v3 failure: cold-start.  Encoder outputs noise at t=0 →
workers adapt to noisy z → stuck in bad local optimum even after encoder
improves.

TWO FIXES vs v3:
  Fix 1  BC PRE-TRAIN  : Before RL, supervise (encoder + actor) with WMMSE
                         actions.  Encoder learns to produce meaningful z;
                         actor learns approximate WMMSE policy conditioned
                         on z.  cold-start problem eliminated.
  Fix 2  ENCODER FREEZE: Freeze encoder for first ENC_FREEZE_STEPS RL steps
                         so workers can stabilise on the BC-initialised z
                         before fine-tuning begins.

Everything else is identical to v3 (cc-HASAC A):
  - DeepSet encoder, parameter-shared actor, centralised twin Q-critic
  - Snapshot pool (N=50 train, N=20 eval, seed 9999)
  - K_HOLD=50, ENC_LR=1e-4, Z_KL_COEF=0.001, R1-partial obs (7-dim)
"""
import sys, os
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from envs.cc_env_r1partial import CCEnvR1Partial as CCEnv
from baseline import (Cfg, make_snapshot, metrics,
                      bl_wmmse, bl_pf_wmmse, bl_full_power,
                      dbm_to_w, noise_w_per_rb)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_BS        = 3
N_RB        = 4
KPM_DIM     = CCEnv.KPM_DIM            # 7
Z_DIM       = 8
WORKER_OBS  = KPM_DIM + N_BS + Z_DIM  # 18
SHARE_OBS   = N_BS * KPM_DIM + Z_DIM  # 29
HIDDEN      = 128
ACTOR_LR    = 3e-4
ENC_LR      = 1e-4
GAMMA       = 0.99
POLYAK      = 0.005
ALPHA_INIT  = 0.01
Z_KL_COEF   = 0.001
BUFFER_SIZE = 100_000
BATCH_SIZE  = 256
WARMUP      = 2_000
TRAIN_EVERY = 10
K_HOLD      = 50
NUM_STEPS   = 300_000
LOG_EVERY   = 10_000
EP_LEN      = 200
N_POOL      = 50
SEED        = 42
RESULTS_DIR = "/home/hyc1014/DL/FinalProject/results"

# v6 specific
BC_STEPS        = 1500   # supervised BC gradient steps before RL
BC_BATCH        = 32     # snapshots per BC update
BC_LR           = 1e-3   # higher LR for BC (supervised is easier)
ENC_FREEZE_STEPS = 10_000  # RL steps to keep encoder frozen after BC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGENT_IDS = torch.eye(N_BS, dtype=torch.float32).to(DEVICE)


# ── Modules ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(HIDDEN, HIDDEN)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GlobalContextEncoder(nn.Module):
    """DeepSet: [B, N_BS, KPM_DIM] → z [B, Z_DIM].  Permutation-invariant."""
    def __init__(self):
        super().__init__()
        self.cell = nn.Sequential(
            nn.Linear(KPM_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32),      nn.ReLU(),
        )
        self.proj = nn.Linear(32, Z_DIM)

    def forward(self, kpm):
        h = self.cell(kpm)
        z = self.proj(h.mean(dim=1))
        return z


class ValueNorm(nn.Module):
    def __init__(self, eps=1e-5, beta=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var",  torch.ones(1))
        self.eps  = eps
        self.beta = beta

    @property
    def std(self):
        return (self.var + self.eps).sqrt()

    def update(self, x):
        with torch.no_grad():
            self.mean = (1 - self.beta) * self.mean + self.beta * x.mean()
            self.var  = (1 - self.beta) * self.var  + self.beta * x.var()

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class CCBuffer:
    FIELDS = 7

    def __init__(self, cap=BUFFER_SIZE):
        self.buf = deque(maxlen=cap)

    def push(self, kpm, gkpm, acts, rews, kpm_n, gkpm_n, done):
        self.buf.append((
            kpm.astype(np.float32), gkpm.astype(np.float32),
            acts.astype(np.float32), rews.astype(np.float32),
            kpm_n.astype(np.float32), gkpm_n.astype(np.float32),
            np.float32(done),
        ))

    def sample(self, n):
        idx   = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.stack([b[j] for b in batch]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


# ── Runner ────────────────────────────────────────────────────────────────────

class CCHASACv6Runner:
    def __init__(self):
        self.encoder = GlobalContextEncoder().to(DEVICE)
        self.actor   = MLP(WORKER_OBS, N_RB * 2).to(DEVICE)
        self.q1      = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q2      = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q1_tgt  = deepcopy(self.q1)
        self.q2_tgt  = deepcopy(self.q2)
        self.vnorm   = ValueNorm().to(DEVICE)

        self.buf      = CCBuffer()
        self.env      = CCEnv(seed=SEED)
        self.eval_env = CCEnv(seed=SEED + 9999)

        cfg = self.env.cfg
        rng_pool         = np.random.default_rng(SEED + 1)
        self._train_pool = [make_snapshot(cfg, rng_pool) for _ in range(N_POOL)]
        self._eval_pool  = [make_snapshot(cfg, np.random.default_rng(9999))
                            for _ in range(20)]

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=ACTOR_LR)
        self.enc_opt    = torch.optim.Adam(self.encoder.parameters(), lr=ENC_LR)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=ACTOR_LR)

        self.log_alpha = torch.tensor(np.log(ALPHA_INIT),
                                      requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ACTOR_LR)
        self.target_entropy = -float(N_RB)

    # ── snapshot-pool reset helpers ────────────────────────────────────────────

    def _reset_from_pool(self, env, pool, idx):
        As, assoc = pool[idx]
        env.As, env.assoc = As.copy(), assoc.copy()
        env._step = 0
        P0 = np.full((N_BS, N_RB), env.Pmax * 0.5)
        m0 = metrics(P0, env.As, env.assoc, env.cfg, env.nw)
        env._kpm = env._build_kpm(m0, P0)
        return env._kpm.copy()

    def _reset_train(self):
        return self._reset_from_pool(self.env, self._train_pool,
                                     np.random.randint(N_POOL))

    def _reset_eval(self, idx):
        return self._reset_from_pool(self.eval_env, self._eval_pool, idx)

    # ── helpers ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_z(self, gkpm_np):
        t = torch.FloatTensor(gkpm_np).unsqueeze(0).to(DEVICE)
        return self.encoder(t).squeeze(0).cpu().numpy()

    def _actor_forward(self, worker_obs_t):
        B = worker_obs_t.shape[0]
        flat = worker_obs_t.reshape(B * N_BS, WORKER_OBS)
        out  = self.actor(flat)
        mean, log_std = out[:, :N_RB], out[:, N_RB:]
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        z   = mean + std * torch.randn_like(mean)
        a   = torch.tanh(z)
        lp  = (
            -((z - mean) ** 2) / (2 * std ** 2 + 1e-8)
            - log_std - 0.5 * np.log(2 * np.pi)
            - torch.log(1 - a.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        a   = (a + 1) / 2
        return a.reshape(B, N_BS, N_RB), lp.reshape(B, N_BS, 1)

    @torch.no_grad()
    def _get_actions(self, kpm_np, z_np):
        ids  = AGENT_IDS.cpu().numpy()
        z_bc = np.tile(z_np, (N_BS, 1))
        wobs = np.concatenate([kpm_np, ids, z_bc], -1)
        t = torch.FloatTensor(wobs).unsqueeze(0).to(DEVICE)
        a, _ = self._actor_forward(t)
        return a.squeeze(0).cpu().numpy()

    def _build_worker_obs(self, kpm_t, z_t):
        ids = AGENT_IDS.unsqueeze(0).expand(kpm_t.shape[0], -1, -1)
        z_e = z_t.unsqueeze(1).expand(-1, N_BS, -1)
        return torch.cat([kpm_t, ids, z_e], dim=-1)

    def _build_share_obs(self, kpm_t, z_t):
        flat_kpm = kpm_t.reshape(kpm_t.shape[0], -1)
        return torch.cat([flat_kpm, z_t], dim=-1)

    # ── Fix 1: BC pre-training ────────────────────────────────────────────────

    def pre_train_bc(self):
        """Supervise encoder + actor with WMMSE actions (Fix 1)."""
        rng = np.random.default_rng(12345)
        cfg, nw, Pmax = self.env.cfg, self.env.nw, self.env.Pmax

        # Build (global_kpm, wmmse_action) pairs for every training snapshot
        wmmse_acts, init_kpms = [], []
        for As, assoc in self._train_pool:
            P_w = bl_wmmse(As, cfg, nw, Pmax, rng)          # [N_BS, N_RB]
            wmmse_acts.append((P_w / Pmax).astype(np.float32))

            P0 = np.full((N_BS, N_RB), Pmax * 0.5)
            m0 = metrics(P0, As, assoc, cfg, nw)
            self.env.As, self.env.assoc = As.copy(), assoc.copy()
            kpm = self.env._build_kpm(m0, P0)                # [N_BS, KPM_DIM]
            init_kpms.append(kpm.astype(np.float32))

        wmmse_t = torch.FloatTensor(np.stack(wmmse_acts)).to(DEVICE)  # [N_POOL, N_BS, N_RB]
        kpms_t  = torch.FloatTensor(np.stack(init_kpms)).to(DEVICE)   # [N_POOL, N_BS, KPM_DIM]

        bc_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=BC_LR,
        )

        print("── BC pre-training ─────────────────────────────────────────", flush=True)
        last_loss = float("nan")
        for step in range(1, BC_STEPS + 1):
            idx   = np.random.choice(len(self._train_pool), BC_BATCH, replace=True)
            kpm_b = kpms_t[idx]    # [B, N_BS, KPM_DIM]
            tgt_b = wmmse_t[idx]   # [B, N_BS, N_RB]
            B     = len(idx)

            z    = self.encoder(kpm_b)             # [B, Z_DIM]
            wobs = self._build_worker_obs(kpm_b, z)# [B, N_BS, WORKER_OBS]

            flat = wobs.reshape(B * N_BS, WORKER_OBS)
            out  = self.actor(flat)                # [B*N_BS, N_RB*2]
            mean = out[:, :N_RB]                   # [B*N_BS, N_RB]
            pred = ((torch.tanh(mean) + 1) / 2).reshape(B, N_BS, N_RB)

            bc_loss = ((pred - tgt_b) ** 2).mean()
            bc_opt.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.actor.parameters()), 5.0)
            bc_opt.step()
            last_loss = bc_loss.item()

            if step % 300 == 0 or step == BC_STEPS:
                print(f"  [BC] step {step:4d}/{BC_STEPS}  MSE={last_loss:.6f}", flush=True)

        # Quick sanity: BC-policy sum_rate on 5 eval snapshots
        sr_bc = []
        for ep in range(5):
            kpm  = self._reset_eval(ep)
            gkpm = self.eval_env.get_global_kpm()
            z_np = self._get_z(gkpm)
            sr_ep, steps = 0.0, 0
            while True:
                acts = self._get_actions(kpm, z_np)
                kpm, _, done, m = self.eval_env.step(acts)
                sr_ep += m["sum_rate"]; steps += 1
                if steps % K_HOLD == 0:
                    gkpm  = self.eval_env.get_global_kpm()
                    z_np  = self._get_z(gkpm)
                if done:
                    break
            sr_bc.append(sr_ep / steps)
        print(f"  BC sanity eval (5 eps): sum_rate={np.mean(sr_bc):.4f} bps/Hz", flush=True)

    # ── RL update ─────────────────────────────────────────────────────────────

    def update(self, freeze_enc=False):
        kpm, gkpm, acts, rews, kpm_n, gkpm_n, dones = self.buf.sample(BATCH_SIZE)

        kpm_t    = torch.FloatTensor(kpm).to(DEVICE)
        gkpm_t   = torch.FloatTensor(gkpm).to(DEVICE)
        acts_t   = torch.FloatTensor(acts).to(DEVICE)
        rews_t   = torch.FloatTensor(rews).to(DEVICE)
        kpm_n_t  = torch.FloatTensor(kpm_n).to(DEVICE)
        gkpm_n_t = torch.FloatTensor(gkpm_n).to(DEVICE)
        done_t   = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        alpha = self.log_alpha.exp().detach()
        sum_r = rews_t.sum(dim=1, keepdim=True)

        z     = self.encoder(gkpm_t)
        z_det = z.detach()
        with torch.no_grad():
            z_next = self.encoder(gkpm_n_t)

        sobs      = self._build_share_obs(kpm_t,   z_det)
        sobs_next = self._build_share_obs(kpm_n_t, z_next)
        wobs_next = self._build_worker_obs(kpm_n_t, z_next)

        # Critic
        with torch.no_grad():
            na, nlp = self._actor_forward(wobs_next)
            na_flat  = na.reshape(BATCH_SIZE, -1)
            nlp_sum  = nlp.sum(dim=1)
            tgt = torch.min(
                self.q1_tgt(torch.cat([sobs_next, na_flat], -1)),
                self.q2_tgt(torch.cat([sobs_next, na_flat], -1)),
            ) - alpha * nlp_sum
            self.vnorm.update(tgt)
            tgt_norm = self.vnorm.normalize(tgt)
            backup = self.vnorm.normalize(
                sum_r + GAMMA * (1 - done_t) * self.vnorm.denormalize(tgt_norm))

        acts_flat = acts_t.reshape(BATCH_SIZE, -1)
        q1_l = ((self.q1(torch.cat([sobs, acts_flat], -1)) - backup) ** 2).mean()
        q2_l = ((self.q2(torch.cat([sobs, acts_flat], -1)) - backup) ** 2).mean()
        if not (q1_l + q2_l).isnan():
            self.critic_opt.zero_grad()
            (q1_l + q2_l).backward()
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
            self.critic_opt.step()

        # Actor + Encoder
        wobs_grad = self._build_worker_obs(kpm_t, z)
        sobs_grad = self._build_share_obs(kpm_t,  z)
        new_a, lp = self._actor_forward(wobs_grad)
        new_a_flat = new_a.reshape(BATCH_SIZE, -1)
        lp_sum     = lp.sum(dim=1)
        q_val = torch.min(
            self.q1(torch.cat([sobs_grad, new_a_flat], -1)),
            self.q2(torch.cat([sobs_grad, new_a_flat], -1)),
        )
        z_kl   = Z_KL_COEF * (z ** 2).mean()
        a_loss = (alpha * lp_sum - q_val).mean() + z_kl
        if not a_loss.isnan():
            self.actor_opt.zero_grad()
            self.enc_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
            if not freeze_enc:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
                self.enc_opt.step()

        # Alpha
        with torch.no_grad():
            _, lp_det = self._actor_forward(wobs_grad.detach())
            lp_det_sum = lp_det.sum(dim=1)
        al = -(self.log_alpha * (lp_det_sum + self.target_entropy * N_BS)).mean()
        self.alpha_opt.zero_grad(); al.backward(); self.alpha_opt.step()

        for p, tp in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, n_episodes=20, zero_z=False, shuffle_z=False):
        ep_rews, sum_rates = [], []
        n_episodes = min(n_episodes, len(self._eval_pool))
        for ep in range(n_episodes):
            kpm  = self._reset_eval(ep)
            gkpm = self.eval_env.get_global_kpm()
            z = np.zeros(Z_DIM, dtype=np.float32) if zero_z else self._get_z(gkpm)
            if shuffle_z:
                z = z[np.random.permutation(Z_DIM)]
            ep_r, sr_sum, steps = 0.0, 0.0, 0
            while True:
                acts = self._get_actions(kpm, z)
                kpm, rews, done, m = self.eval_env.step(acts)
                ep_r   += rews.sum()
                sr_sum += m['sum_rate']
                steps  += 1
                if steps % K_HOLD == 0:
                    gkpm = self.eval_env.get_global_kpm()
                    z = (np.zeros(Z_DIM, dtype=np.float32) if zero_z
                         else self._get_z(gkpm))
                    if shuffle_z:
                        z = z[np.random.permutation(Z_DIM)]
                if done:
                    break
            ep_rews.append(ep_r)
            sum_rates.append(sr_sum / steps)
        return float(np.mean(ep_rews)), float(np.mean(sum_rates))

    def _baseline_eval(self):
        cfg, nw, Pmax = self.env.cfg, self.env.nw, self.env.Pmax
        rng = np.random.default_rng(999)
        res = {"wmmse": [], "pf_wmmse": [], "full_power": []}
        for As, assoc in self._eval_pool:
            res["wmmse"].append(
                metrics(bl_wmmse(As, cfg, nw, Pmax, rng), As, assoc, cfg, nw)['sum_rate'])
            res["pf_wmmse"].append(
                metrics(bl_pf_wmmse(As, assoc, cfg, nw, Pmax, rng), As, assoc, cfg, nw)['sum_rate'])
            res["full_power"].append(
                metrics(bl_full_power(As, cfg, nw, Pmax), As, assoc, cfg, nw)['sum_rate'])
        return {k: float(np.mean(v)) for k, v in res.items()}

    # ── Training Loop ─────────────────────────────────────────────────────────

    def run(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "cc_hasac_v6_log.txt")
        rew_path = os.path.join(RESULTS_DIR, "cc_hasac_v6_rewards.npy")

        print(f"cc-HASAC v6 (BC pre-train + enc-freeze) | device={DEVICE} | "
              f"KPM={KPM_DIM} z={Z_DIM} obs={WORKER_OBS} | "
              f"pool={N_POOL} K_HOLD={K_HOLD} BC_STEPS={BC_STEPS} "
              f"ENC_FREEZE={ENC_FREEZE_STEPS}", flush=True)

        # ── Fix 1: BC pre-training ────────────────────────────────────────────
        self.pre_train_bc()

        # ── RL training loop ──────────────────────────────────────────────────
        kpm  = self._reset_train()
        gkpm = self.env.get_global_kpm()
        z    = self._get_z(gkpm)

        ep_rew, done_eps, log_lines = 0.0, [], []
        print(f"\n{'Step':>8}  {'AvgEpRew':>10}  {'SumRate':>10}  "
              f"{'DoneEps':>8}  {'EncFrz':>6}", flush=True)

        for step in range(1, NUM_STEPS + 1):
            freeze_enc = step <= ENC_FREEZE_STEPS  # Fix 2

            if step <= WARMUP:
                acts = np.random.uniform(0, 1, (N_BS, N_RB)).astype(np.float32)
            else:
                acts = self._get_actions(kpm, z)

            kpm_n, rews, done, _ = self.env.step(acts)
            gkpm_n = self.env.get_global_kpm()
            ep_rew += rews.sum()

            self.buf.push(kpm, gkpm, acts, rews, kpm_n, gkpm_n, float(done))
            kpm, gkpm = kpm_n, gkpm_n

            if step % K_HOLD == 0:
                z = self._get_z(gkpm)

            if done:
                done_eps.append(ep_rew)
                ep_rew = 0.0
                kpm    = self._reset_train()
                gkpm   = self.env.get_global_kpm()
                z      = self._get_z(gkpm)

            if step > WARMUP and step % TRAIN_EVERY == 0 and len(self.buf) >= BATCH_SIZE:
                self.update(freeze_enc=freeze_enc)

            if step == ENC_FREEZE_STEPS:
                print(f"  [step {step}] Encoder unfrozen — RL fine-tuning begins",
                      flush=True)

            if step % LOG_EVERY == 0:
                avg_rew, avg_sr = self.evaluate(n_episodes=5)
                frz_tag = "Y" if freeze_enc else "N"
                line = (f"{step:8d}  {avg_rew:10.2f}  {avg_sr:10.4f}  "
                        f"{len(done_eps):8d}  {frz_tag:>6}")
                print(line, flush=True)
                log_lines.append(line)

        np.save(rew_path, np.array(done_eps))
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        # ── z-Ablation ───────────────────────────────────────────────────────
        print("\n── z-Ablation Probe (20 held-out snapshots) ──", flush=True)
        r_z,    sr_z    = self.evaluate(n_episodes=20, zero_z=False)
        r_0,    sr_0    = self.evaluate(n_episodes=20, zero_z=True)
        r_shuf, sr_shuf = self.evaluate(n_episodes=20, shuffle_z=True)
        print(f"  With z    : ep_rew={r_z:.2f}  sum_rate={sr_z:.4f}", flush=True)
        print(f"  z ← 0     : ep_rew={r_0:.2f}  sum_rate={sr_0:.4f}  "
              f"Δsr={sr_z-sr_0:+.4f}", flush=True)
        print(f"  z shuffle : ep_rew={r_shuf:.2f}  sum_rate={sr_shuf:.4f}  "
              f"Δsr={sr_z-sr_shuf:+.4f}", flush=True)

        # ── Baseline Comparison ──────────────────────────────────────────────
        print("\n── Baseline Comparison (same 20 held-out snapshots) ──", flush=True)
        bl = self._baseline_eval()
        print(f"  full_power : {bl['full_power']:.4f} bps/Hz", flush=True)
        print(f"  wmmse      : {bl['wmmse']:.4f} bps/Hz", flush=True)
        print(f"  pf_wmmse   : {bl['pf_wmmse']:.4f} bps/Hz", flush=True)
        print(f"  cc-HASAC v6: {sr_z:.4f} bps/Hz  "
              f"(gap vs wmmse = {100*(bl['wmmse']-sr_z)/bl['wmmse']:+.1f}%)",
              flush=True)
        print(f"\n  Reference  : Ind-SAC A=28.1  cc-HASAC A=26.9  "
              f"Ind-SAC B=33.6  WMMSE=85.0", flush=True)

        np.save(os.path.join(RESULTS_DIR, "cc_hasac_v6_ablation.npy"),
                np.array([r_z, r_0, r_shuf, sr_z, sr_0, sr_shuf]))
        return done_eps


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CCHASACv6Runner().run()
