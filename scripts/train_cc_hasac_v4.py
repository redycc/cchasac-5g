"""
cc-HASAC v4 — α-gate curriculum + β slow encoder update.

Built on train_cc_hasac_A.py (Direction A).  Fixes the encoder cold-start
problem where early noisy z corrupts worker learning: previous experiments
showed z IS useful (z-ablation Δ=+4~+5 bps/Hz) but cc-HASAC still underperforms
Ind-SAC because workers adapt to noisy early-z and get stuck.

THE FIX: let workers stabilize first (like Ind-SAC), then gradually introduce z.

THREE CHANGES vs Direction A:
  Change 1 α-GATE on z input : multiply z by α before concatenation in
                                _build_worker_obs / _get_actions. α warms up
                                linearly 0 → 1 over ALPHA_WARMUP steps so workers
                                see (effectively) z=0 early and stabilize first.
  Change 2 β SLOW ENCODER     : encoder optimizer steps only every β steps
                                (actor + critic still step every update). β decays
                                from BETA_START (200) to BETA_END (10) over
                                BETA_DECAY_STEPS so the encoder changes slowly at
                                first (less noise) then catches up.
  Change 3 LOG α and β        : LOG_EVERY print now shows current alpha and beta.

Architecture (unchanged from A):
  Encoder f_θ  : [N_BS, KPM_DIM] → z ∈ R^Z_DIM  (DeepSet, perm-invariant)
  Actor  π_φ   : [KPM_DIM + N_BS + Z_DIM] → [N_RB*2]  (parameter-shared)
  Critic Q_ψ   : [N_BS*(KPM_DIM+Z_DIM) + N_BS*N_RB] → 1  (centralised)
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
KPM_DIM     = CCEnv.KPM_DIM            # 7  (R1-partial: 4 SINR + load + tput + n_ue)
Z_DIM       = 8
WORKER_OBS  = KPM_DIM + N_BS + Z_DIM  # 7+3+8 = 18
SHARE_OBS   = N_BS * KPM_DIM + Z_DIM  # 21+8 = 29
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

# ── v4 curriculum schedules ─────────────────────────────────────────────────
ALPHA_WARMUP    = 100_000   # Change 1: steps to ramp α from 0 → 1
BETA_START      = 200       # Change 2: encoder update interval at start
BETA_END        = 10        # Change 2: encoder update interval at end
BETA_DECAY_STEPS = 150_000  # Change 2: steps to decay β from start to end


def get_alpha(step):
    """Linear warmup of the z-gate: 0 → 1 over ALPHA_WARMUP steps."""
    return min(1.0, step / ALPHA_WARMUP)


def get_beta(step):
    """Encoder update interval: BETA_START → BETA_END over BETA_DECAY_STEPS."""
    frac = min(1.0, step / BETA_DECAY_STEPS)
    return int(BETA_START + frac * (BETA_END - BETA_START))


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

class CCBufferV3:
    """(kpm, global_kpm, actions, rewards, kpm_next, global_kpm_next, done)"""
    FIELDS = 7

    def __init__(self, cap=BUFFER_SIZE):
        self.buf = deque(maxlen=cap)

    def push(self, kpm, gkpm, acts, rews, kpm_n, gkpm_n, done):
        self.buf.append((
            kpm.astype(np.float32),
            gkpm.astype(np.float32),
            acts.astype(np.float32),
            rews.astype(np.float32),
            kpm_n.astype(np.float32),
            gkpm_n.astype(np.float32),
            np.float32(done),
        ))

    def sample(self, n):
        idx   = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.stack([b[j] for b in batch]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


# ── Runner ────────────────────────────────────────────────────────────────────

class CCHASACv4Runner:
    def __init__(self):
        self.encoder = GlobalContextEncoder().to(DEVICE)
        self.actor   = MLP(WORKER_OBS, N_RB * 2).to(DEVICE)
        self.q1      = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q2      = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q1_tgt  = deepcopy(self.q1)
        self.q2_tgt  = deepcopy(self.q2)
        self.vnorm   = ValueNorm().to(DEVICE)

        self.buf      = CCBufferV3()
        self.env      = CCEnv(seed=SEED)
        self.eval_env = CCEnv(seed=SEED + 9999)

        # ── Snapshot pools ──────────────────────────────────────────────────────
        cfg = self.env.cfg
        rng_pool         = np.random.default_rng(SEED + 1)
        self._train_pool = [make_snapshot(cfg, rng_pool) for _ in range(N_POOL)]
        self._eval_pool  = [make_snapshot(cfg, np.random.default_rng(9999))
                            for _ in range(20)]

        # ── separate optimizers, separate LRs ──────────────────────────────────
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=ACTOR_LR)
        self.enc_opt    = torch.optim.Adam(self.encoder.parameters(), lr=ENC_LR)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=ACTOR_LR)

        self.log_alpha = torch.tensor(np.log(ALPHA_INIT),
                                      requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ACTOR_LR)
        self.target_entropy = -float(N_RB)

        # ── v4 curriculum state ─────────────────────────────────────────────────
        self.update_count = 0       # Change 2: counts update() calls (for β gating)
        self.beta = BETA_START      # Change 2: current encoder update interval

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
    def _get_actions(self, kpm_np, z_np, alpha=1.0):
        ids  = AGENT_IDS.cpu().numpy()
        z_bc = np.tile(alpha * z_np, (N_BS, 1))   # Change 1: α-gate on z
        wobs = np.concatenate([kpm_np, ids, z_bc], -1)
        t = torch.FloatTensor(wobs).unsqueeze(0).to(DEVICE)
        a, _ = self._actor_forward(t)
        return a.squeeze(0).cpu().numpy()

    def _build_worker_obs(self, kpm_t, z_t, alpha=1.0):
        ids = AGENT_IDS.unsqueeze(0).expand(kpm_t.shape[0], -1, -1)
        z_e = (alpha * z_t).unsqueeze(1).expand(-1, N_BS, -1)  # Change 1: α-gate
        return torch.cat([kpm_t, ids, z_e], dim=-1)

    def _build_share_obs(self, kpm_t, z_t):
        flat_kpm = kpm_t.reshape(kpm_t.shape[0], -1)
        return torch.cat([flat_kpm, z_t], dim=-1)

    # ── update ───────────────────────────────────────────────────────────────

    def update(self, alpha_gate=1.0):
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

        # z: grad for actor/encoder, detached for critic
        z     = self.encoder(gkpm_t)
        z_det = z.detach()
        with torch.no_grad():
            z_next = self.encoder(gkpm_n_t)

        sobs      = self._build_share_obs(kpm_t,   z_det)
        sobs_next = self._build_share_obs(kpm_n_t, z_next)
        wobs_next = self._build_worker_obs(kpm_n_t, z_next, alpha_gate)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            na, nlp = self._actor_forward(wobs_next)
            na_flat = na.reshape(BATCH_SIZE, -1)
            nlp_sum = nlp.sum(dim=1)
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

        # ── Actor + Encoder update (separate optimizers, z-KL, β slow enc) ─────
        wobs_grad = self._build_worker_obs(kpm_t, z, alpha_gate)
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
            nn.utils.clip_grad_norm_(self.actor.parameters(),   10.0)
            self.actor_opt.step()
            # ── Change 2: β slow encoder update ───────────────────────────────
            if self.update_count % self.beta == 0:
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
                self.enc_opt.step()
            else:
                self.encoder.zero_grad()   # discard encoder grads this step

        self.update_count += 1

        # ── Alpha update ──────────────────────────────────────────────────────
        with torch.no_grad():
            _, lp_det = self._actor_forward(wobs_grad.detach())
            lp_det_sum = lp_det.sum(dim=1)
        al = -(self.log_alpha * (lp_det_sum + self.target_entropy * N_BS)).mean()
        self.alpha_opt.zero_grad(); al.backward(); self.alpha_opt.step()

        # ── Soft target update ──────────────────────────────────────────────────
        for p, tp in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, n_episodes=20, zero_z=False, shuffle_z=False, alpha=1.0):
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
                acts = self._get_actions(kpm, z, alpha)
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
        log_path = os.path.join(RESULTS_DIR, "cc_hasac_v4_log.txt")
        rew_path = os.path.join(RESULTS_DIR, "cc_hasac_v4_rewards.npy")

        kpm  = self._reset_train()
        gkpm = self.env.get_global_kpm()
        z    = self._get_z(gkpm)

        ep_rew, done_eps, log_lines = 0.0, [], []
        print(f"cc-HASAC v4 (R1-partial, freq_sel) | device={DEVICE} | KPM={KPM_DIM} z={Z_DIM} "
              f"obs={WORKER_OBS} | pool={N_POOL} K_HOLD={K_HOLD} "
              f"enc_lr={ENC_LR} z_kl={Z_KL_COEF}", flush=True)
        print(f"  v4 curriculum: alpha_warmup={ALPHA_WARMUP} "
              f"beta {BETA_START}->{BETA_END} over {BETA_DECAY_STEPS}", flush=True)
        print(f"{'Step':>8}  {'AvgEpRew':>10}  {'SumRate':>10}  {'DoneEps':>8}  "
              f"{'Alpha':>6}  {'Beta':>5}", flush=True)

        for step in range(1, NUM_STEPS + 1):
            alpha_gate = get_alpha(step)          # Change 1
            self.beta  = get_beta(step)           # Change 2

            if step <= WARMUP:
                acts = np.random.uniform(0, 1, (N_BS, N_RB)).astype(np.float32)
            else:
                acts = self._get_actions(kpm, z, alpha_gate)

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
                self.update(alpha_gate)

            if step % LOG_EVERY == 0:
                avg_rew, avg_sr = self.evaluate(n_episodes=5, alpha=alpha_gate)
                line = (f"{step:8d}  {avg_rew:10.2f}  {avg_sr:10.4f}  "
                        f"{len(done_eps):8d}  {alpha_gate:6.3f}  {self.beta:5d}")
                print(line, flush=True)
                log_lines.append(line)

        np.save(rew_path, np.array(done_eps))
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        # ── z-Ablation Probe (α=1 at eval, full z exposure) ──────────────────
        print("\n── z-Ablation Probe (20 held-out snapshots) ──", flush=True)
        r_z,    sr_z    = self.evaluate(n_episodes=20, zero_z=False, alpha=1.0)
        r_0,    sr_0    = self.evaluate(n_episodes=20, zero_z=True,  alpha=1.0)
        r_shuf, sr_shuf = self.evaluate(n_episodes=20, shuffle_z=True, alpha=1.0)
        print(f"  With z    : ep_rew={r_z:.2f}  sum_rate={sr_z:.4f}", flush=True)
        print(f"  z ← 0     : ep_rew={r_0:.2f}  sum_rate={sr_0:.4f}  "
              f"Δsr={sr_z-sr_0:+.4f}", flush=True)
        print(f"  z shuffle : ep_rew={r_shuf:.2f}  sum_rate={sr_shuf:.4f}  "
              f"Δsr={sr_z-sr_shuf:+.4f}", flush=True)

        # ── Baseline Comparison (same held-out pool) ─────────────────────────
        print("\n── Baseline Comparison (same 20 held-out snapshots) ──",
              flush=True)
        bl = self._baseline_eval()
        print(f"  full_power : {bl['full_power']:.4f} bps/Hz", flush=True)
        print(f"  wmmse      : {bl['wmmse']:.4f} bps/Hz", flush=True)
        print(f"  pf_wmmse   : {bl['pf_wmmse']:.4f} bps/Hz", flush=True)
        print(f"  cc-HASAC v4: {sr_z:.4f} bps/Hz  "
              f"(gap vs wmmse = {100*(bl['wmmse']-sr_z)/bl['wmmse']:+.1f}%)",
              flush=True)
        print(f"  Ind-SAC A  : 28.1095 bps/Hz (reference to beat)", flush=True)
        print(f"\n  Reference  : WMMSE=87.9  full_power=27.3 (project constants)",
              flush=True)

        np.save(os.path.join(RESULTS_DIR, "cc_hasac_v4_ablation.npy"),
                np.array([r_z, r_0, r_shuf, sr_z, sr_0, sr_shuf]))
        return done_eps


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CCHASACv4Runner().run()
