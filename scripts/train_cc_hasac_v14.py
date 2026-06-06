"""
cc-HASAC v14 — v13 + all short-term & medium-term improvements:

  1. Best-checkpoint: save best eval model, restore at end → stops post-peak decay
  2. Cosine enc LR decay: 1e-5 → 0 after unfreeze → smooth encoder stop
  3. Grid-opt BC target: near-global optimum (G=13, 2197 combos/RB, N_BS=3)
     vs v13's WMMSE local optimum → better supervised signal
  4. Richer encoder input: intf_norm added to global KPM (7→8-dim per BS),
     Z_DIM 8→16; worker obs stays R1-partial 7-dim (no direct intf leak to actor)
  5. BC_STEPS 1500→5000: lower BC MSE for stronger actor-z coupling
"""
import sys, os, math
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from envs.cc_env_r1partial import CCEnvR1Partial as CCEnv
from baseline import (Cfg, make_snapshot, metrics,
                      bl_wmmse, bl_pf_wmmse, bl_full_power, bl_grid,
                      dbm_to_w, noise_w_per_rb)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_BS        = 3
N_RB        = 4
KPM_DIM     = CCEnv.KPM_DIM            # 7  — worker obs per agent (R1-partial)
ENC_KPM_DIM = KPM_DIM + 1             # 8  — encoder input (adds intf_norm)
Z_DIM       = 16                       # 8→16: more coordination capacity
WORKER_OBS  = KPM_DIM + N_BS + Z_DIM  # 26  (7 obs + 3 id + 16 z)
SHARE_OBS   = N_BS * KPM_DIM + Z_DIM  # 37  (21 flat_kpm + 16 z)
HIDDEN      = 128
ACTOR_LR        = 3e-4
ENC_LR          = 1e-4
ENC_LR_FINETUNE = 1e-5   # starting LR after unfreeze, cosine decays to ~0
GAMMA       = 0.99
POLYAK      = 0.005
ALPHA_INIT  = 0.001
Z_KL_COEF   = 0.001
BUFFER_SIZE = 100_000
BATCH_SIZE  = 256
WARMUP      = 0
TRAIN_EVERY = 10
K_HOLD      = 50
NUM_STEPS   = 500_000
LOG_EVERY   = 10_000
EP_LEN      = 200
N_POOL      = 50
SEED        = 42
RESULTS_DIR = "/home/hyc1014/DL/FinalProject/results"

BC_STEPS         = 5000    # 1500→5000: deeper supervised pretraining
BC_BATCH         = 32
BC_LR            = 1e-3
ENC_FREEZE_STEPS = 100_000

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGENT_IDS = torch.eye(N_BS, dtype=torch.float32).to(DEVICE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _intf_norm(intf_caused):
    """Normalise intf_caused [N_BS] → [-1,1] using log2 scale (consistent with SINR norm)."""
    return np.clip(np.log2(1.0 + np.maximum(intf_caused, 0.0)) / 10.0, -1, 1).astype(np.float32)


def _build_gkpm_ext(gkpm, intf_caused):
    """Append intf_norm to global KPM → [N_BS, ENC_KPM_DIM=8]."""
    return np.concatenate([gkpm, _intf_norm(intf_caused).reshape(N_BS, 1)], axis=1).astype(np.float32)


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
    """DeepSet: [B, N_BS, ENC_KPM_DIM=8] → z [B, Z_DIM=16]. Permutation-invariant."""
    def __init__(self):
        super().__init__()
        self.cell = nn.Sequential(
            nn.Linear(ENC_KPM_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32),           nn.ReLU(),
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

    def push(self, kpm, gkpm_ext, acts, rews, kpm_n, gkpm_ext_n, done):
        self.buf.append((
            kpm.astype(np.float32),
            gkpm_ext.astype(np.float32),      # [N_BS, ENC_KPM_DIM=8]
            acts.astype(np.float32),
            rews.astype(np.float32),
            kpm_n.astype(np.float32),
            gkpm_ext_n.astype(np.float32),
            np.float32(done),
        ))

    def sample(self, n):
        idx   = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.stack([b[j] for b in batch]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


# ── Runner ────────────────────────────────────────────────────────────────────

class CCHASACv14Runner:
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

        # best-checkpoint tracking
        self.best_sr   = -float("inf")
        self.best_ckpt = None

    # ── snapshot-pool helpers ─────────────────────────────────────────────────

    def _reset_from_pool(self, env, pool, idx):
        As, assoc = pool[idx]
        env.As, env.assoc = As.copy(), assoc.copy()
        env._step = 0
        P0 = np.full((N_BS, N_RB), env.Pmax * 0.5)
        m0 = metrics(P0, env.As, env.assoc, env.cfg, env.nw)
        env._kpm = env._build_kpm(m0, P0)
        kpm = env._kpm.copy()
        gkpm_ext = _build_gkpm_ext(kpm, m0['intf_caused'])
        return kpm, gkpm_ext

    def _reset_train(self):
        return self._reset_from_pool(self.env, self._train_pool,
                                     np.random.randint(N_POOL))

    def _reset_eval(self, idx):
        return self._reset_from_pool(self.eval_env, self._eval_pool, idx)

    # ── helpers ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_z(self, gkpm_ext_np):
        t = torch.FloatTensor(gkpm_ext_np).unsqueeze(0).to(DEVICE)
        return self.encoder(t).squeeze(0).cpu().numpy()

    def _actor_forward(self, worker_obs_t):
        B = worker_obs_t.shape[0]
        flat = worker_obs_t.reshape(B * N_BS, WORKER_OBS)
        out  = self.actor(flat)
        mean, log_std = out[:, :N_RB], out[:, N_RB:]
        log_std = torch.clamp(log_std, -5, 2)
        std = log_std.exp()
        zs  = mean + std * torch.randn_like(mean)
        a   = torch.tanh(zs)
        lp  = (
            -((zs - mean) ** 2) / (2 * std ** 2 + 1e-8)
            - log_std - 0.5 * np.log(2 * np.pi)
            - torch.log(1 - a.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        a = (a + 1) / 2
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
        """kpm_t: [B, N_BS, KPM_DIM=7]; worker obs excludes intf."""
        ids = AGENT_IDS.unsqueeze(0).expand(kpm_t.shape[0], -1, -1)
        z_e = z_t.unsqueeze(1).expand(-1, N_BS, -1)
        return torch.cat([kpm_t, ids, z_e], dim=-1)

    def _build_share_obs(self, kpm_t, z_t):
        flat_kpm = kpm_t.reshape(kpm_t.shape[0], -1)
        return torch.cat([flat_kpm, z_t], dim=-1)

    def _save_ckpt(self):
        self.best_ckpt = {
            'encoder': deepcopy(self.encoder.state_dict()),
            'actor':   deepcopy(self.actor.state_dict()),
        }

    def _restore_ckpt(self):
        if self.best_ckpt is not None:
            self.encoder.load_state_dict(self.best_ckpt['encoder'])
            self.actor.load_state_dict(self.best_ckpt['actor'])
            print(f"  [ckpt] Restored best model (peak_sr={self.best_sr:.4f})", flush=True)

    # ── BC pre-training (grid_opt targets) ───────────────────────────────────

    def pre_train_bc(self):
        """Supervise encoder+actor with grid_opt actions (near-global optimum, N_BS=3)."""
        cfg, nw, Pmax = self.env.cfg, self.env.nw, self.env.Pmax

        grid_acts, init_kpms_ext = [], []
        for As, assoc in self._train_pool:
            P_g = bl_grid(As, cfg, nw, Pmax)                     # [N_BS, N_RB]
            grid_acts.append((P_g / Pmax).astype(np.float32))

            P0  = np.full((N_BS, N_RB), Pmax * 0.5)
            m0  = metrics(P0, As, assoc, cfg, nw)
            self.env.As, self.env.assoc = As.copy(), assoc.copy()
            kpm = self.env._build_kpm(m0, P0)                    # [N_BS, 7]
            gkpm_ext = _build_gkpm_ext(kpm, m0['intf_caused'])   # [N_BS, 8]
            init_kpms_ext.append(gkpm_ext.astype(np.float32))

        grid_t  = torch.FloatTensor(np.stack(grid_acts)).to(DEVICE)       # [N_POOL, N_BS, N_RB]
        kpms_t  = torch.FloatTensor(np.stack(init_kpms_ext)).to(DEVICE)   # [N_POOL, N_BS, 8]
        wkpm_t  = kpms_t[:, :, :KPM_DIM]                                  # [N_POOL, N_BS, 7]

        bc_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=BC_LR,
        )

        print("── BC pre-training (grid_opt targets, 5000 steps) ─────────────", flush=True)
        for step in range(1, BC_STEPS + 1):
            idx   = np.random.choice(len(self._train_pool), BC_BATCH, replace=True)
            kpm_b = kpms_t[idx]      # [B, N_BS, 8]  — encoder input
            wkp_b = wkpm_t[idx]      # [B, N_BS, 7]  — worker obs
            tgt_b = grid_t[idx]      # [B, N_BS, N_RB]
            B = len(idx)

            z    = self.encoder(kpm_b)
            wobs = self._build_worker_obs(wkp_b, z)
            flat = wobs.reshape(B * N_BS, WORKER_OBS)
            out  = self.actor(flat)
            mean = out[:, :N_RB]
            pred = ((torch.tanh(mean) + 1) / 2).reshape(B, N_BS, N_RB)

            bc_loss = ((pred - tgt_b) ** 2).mean()
            bc_opt.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.actor.parameters()), 5.0)
            bc_opt.step()

            if step % 1000 == 0 or step == BC_STEPS:
                print(f"  [BC] step {step:5d}/{BC_STEPS}  MSE={bc_loss.item():.6f}", flush=True)

        # BC sanity eval
        sr_bc = []
        for ep in range(5):
            kpm, gkpm_ext = self._reset_eval(ep)
            z_np = self._get_z(gkpm_ext)
            sr_ep, steps = 0.0, 0
            while True:
                acts = self._get_actions(kpm, z_np)
                kpm, _, done, m = self.eval_env.step(acts)
                sr_ep += m["sum_rate"]; steps += 1
                if steps % K_HOLD == 0:
                    gkpm = self.eval_env.get_global_kpm()
                    gkpm_ext = _build_gkpm_ext(gkpm, m['intf_caused'])
                    z_np = self._get_z(gkpm_ext)
                if done:
                    break
            sr_bc.append(sr_ep / steps)
        print(f"  BC sanity eval (5 eps): sum_rate={np.mean(sr_bc):.4f} bps/Hz", flush=True)

    # ── RL update ─────────────────────────────────────────────────────────────

    def update(self, freeze_enc=False):
        kpm, gkpm_ext, acts, rews, kpm_n, gkpm_ext_n, dones = self.buf.sample(BATCH_SIZE)

        kpm_t        = torch.FloatTensor(kpm).to(DEVICE)
        gkpm_ext_t   = torch.FloatTensor(gkpm_ext).to(DEVICE)
        acts_t       = torch.FloatTensor(acts).to(DEVICE)
        rews_t       = torch.FloatTensor(rews).to(DEVICE)
        kpm_n_t      = torch.FloatTensor(kpm_n).to(DEVICE)
        gkpm_ext_n_t = torch.FloatTensor(gkpm_ext_n).to(DEVICE)
        done_t       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        alpha = self.log_alpha.exp().detach()
        sum_r = rews_t.sum(dim=1, keepdim=True)

        z     = self.encoder(gkpm_ext_t)
        z_det = z.detach()
        with torch.no_grad():
            z_next = self.encoder(gkpm_ext_n_t)

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
            kpm, gkpm_ext = self._reset_eval(ep)
            z = np.zeros(Z_DIM, dtype=np.float32) if zero_z else self._get_z(gkpm_ext)
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
                    gkpm_ext = _build_gkpm_ext(gkpm, m['intf_caused'])
                    z = (np.zeros(Z_DIM, dtype=np.float32) if zero_z
                         else self._get_z(gkpm_ext))
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
        log_path = os.path.join(RESULTS_DIR, "cc_hasac_v14_log.txt")
        rew_path = os.path.join(RESULTS_DIR, "cc_hasac_v14_rewards.npy")
        stdout_path = os.path.join(RESULTS_DIR, "cc_hasac_v14_stdout.txt")

        header = (f"cc-HASAC v14 | device={DEVICE} | "
                  f"ENC_KPM={ENC_KPM_DIM} KPM={KPM_DIM} z={Z_DIM} obs={WORKER_OBS} | "
                  f"pool={N_POOL} K_HOLD={K_HOLD} BC_STEPS={BC_STEPS} "
                  f"ENC_FREEZE={ENC_FREEZE_STEPS}")
        print(header, flush=True)

        self.pre_train_bc()

        kpm, gkpm_ext = self._reset_train()
        z = self._get_z(gkpm_ext)

        ep_rew, done_eps, log_lines = 0.0, [], []
        print(f"\n{'Step':>8}  {'AvgEpRew':>10}  {'SumRate':>10}  "
              f"{'DoneEps':>8}  {'EncFrz':>6}  {'EncLR':>8}  {'BestSR':>10}", flush=True)

        for step in range(1, NUM_STEPS + 1):
            freeze_enc = step <= ENC_FREEZE_STEPS

            acts = self._get_actions(kpm, z)

            kpm_n, rews, done, m_info = self.env.step(acts)
            gkpm    = self.env.get_global_kpm()
            gkpm_ext_n = _build_gkpm_ext(gkpm, m_info['intf_caused'])
            ep_rew += rews.sum()

            self.buf.push(kpm, gkpm_ext, acts, rews, kpm_n, gkpm_ext_n, float(done))
            kpm, gkpm_ext = kpm_n, gkpm_ext_n

            if step % K_HOLD == 0:
                z = self._get_z(gkpm_ext)

            if done:
                done_eps.append(ep_rew)
                ep_rew = 0.0
                kpm, gkpm_ext = self._reset_train()
                z = self._get_z(gkpm_ext)

            if step % TRAIN_EVERY == 0 and len(self.buf) >= BATCH_SIZE:
                self.update(freeze_enc=freeze_enc)

            if step == ENC_FREEZE_STEPS:
                for g in self.enc_opt.param_groups:
                    g["lr"] = ENC_LR_FINETUNE
                print(f"  [step {step}] Encoder unfrozen — cosine LR decay starts "
                      f"(enc_lr_init={ENC_LR_FINETUNE:.0e})", flush=True)

            # Cosine LR decay after unfreeze
            if step > ENC_FREEZE_STEPS and step % LOG_EVERY == 0:
                progress = (step - ENC_FREEZE_STEPS) / (NUM_STEPS - ENC_FREEZE_STEPS)
                cosine_lr = ENC_LR_FINETUNE * 0.5 * (1 + math.cos(math.pi * progress))
                for g in self.enc_opt.param_groups:
                    g["lr"] = max(cosine_lr, 1e-7)

            if step % LOG_EVERY == 0:
                avg_rew, avg_sr = self.evaluate(n_episodes=5)
                frz_tag    = "Y" if freeze_enc else "N"
                cur_enc_lr = self.enc_opt.param_groups[0]["lr"]
                if avg_sr > self.best_sr:
                    self.best_sr = avg_sr
                    self._save_ckpt()
                    best_tag = f"{self.best_sr:.4f}*"
                else:
                    best_tag = f"{self.best_sr:.4f}"
                line = (f"{step:8d}  {avg_rew:10.2f}  {avg_sr:10.4f}  "
                        f"{len(done_eps):8d}  {frz_tag:>6}  {cur_enc_lr:.1e}  {best_tag}")
                print(line, flush=True)
                log_lines.append(line)

        np.save(rew_path, np.array(done_eps))
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        # Restore best checkpoint before final ablation
        print(f"\n  [ckpt] Training done. Restoring best checkpoint.", flush=True)
        self._restore_ckpt()

        # z-Ablation
        print("\n── z-Ablation Probe (20 held-out snapshots) ──", flush=True)
        r_z,    sr_z    = self.evaluate(n_episodes=20, zero_z=False)
        r_0,    sr_0    = self.evaluate(n_episodes=20, zero_z=True)
        r_shuf, sr_shuf = self.evaluate(n_episodes=20, shuffle_z=True)
        print(f"  With z    : ep_rew={r_z:.2f}  sum_rate={sr_z:.4f}", flush=True)
        print(f"  z ← 0     : ep_rew={r_0:.2f}  sum_rate={sr_0:.4f}  "
              f"Δsr={sr_z-sr_0:+.4f}", flush=True)
        print(f"  z shuffle : ep_rew={r_shuf:.2f}  sum_rate={sr_shuf:.4f}  "
              f"Δsr={sr_z-sr_shuf:+.4f}", flush=True)

        # Baseline Comparison
        print("\n── Baseline Comparison (same 20 held-out snapshots) ──", flush=True)
        bl = self._baseline_eval()
        print(f"  full_power  : {bl['full_power']:.4f} bps/Hz", flush=True)
        print(f"  wmmse       : {bl['wmmse']:.4f} bps/Hz", flush=True)
        print(f"  pf_wmmse    : {bl['pf_wmmse']:.4f} bps/Hz", flush=True)
        print(f"  cc-HASAC v14: {sr_z:.4f} bps/Hz  "
              f"(gap vs wmmse = {100*(bl['wmmse']-sr_z)/bl['wmmse']:+.1f}%)", flush=True)
        print(f"\n  Reference  : Ind-SAC A=28.1  v13=34.21  v13-peak=39.34  WMMSE=85.0",
              flush=True)

        # Save stdout summary
        all_output = [header, f"  peak_sr={self.best_sr:.4f}",
                      f"  final_sr={sr_z:.4f}",
                      f"  z0_sr={sr_0:.4f}  delta={sr_z-sr_0:+.4f}",
                      f"  zshuffle_sr={sr_shuf:.4f}  delta={sr_z-sr_shuf:+.4f}",
                      f"  wmmse={bl['wmmse']:.4f}  pf_wmmse={bl['pf_wmmse']:.4f}"]
        with open(stdout_path, "w") as f:
            f.write("\n".join(all_output))

        np.save(os.path.join(RESULTS_DIR, "cc_hasac_v14_ablation.npy"),
                np.array([r_z, r_0, r_shuf, sr_z, sr_0, sr_shuf]))
        return done_eps


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CCHASACv14Runner().run()
