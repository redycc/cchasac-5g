"""
cc-HASAC v2 — built on baseline.py's metrics() (per HANDOFF.MD).

Key fixes vs v1:
  • R2 obs = KPM-only [tput, prb_util, n_ue] (3-dim), NO SINR
  • Reward from baseline.metrics() — single source of truth
  • Parameter-shared actor + agent-id one-hot (HANDOFF §7.2)
  • Running value normaliser (stabilises Q-learning, was missing in v1)
  • Centralized Q_ψ(share_obs, a_1..a_N) for CTDE
  • z-probe: z←0 AND z←shuffle at eval

Architecture:
  Encoder f_θ  : [N_BS, KPM_DIM] → z ∈ R^Z_DIM  (DeepSet, perm-invariant)
  Actor  π_φ   : [KPM_DIM + N_BS + Z_DIM] → [N_RB*2]  (parameter-shared)
  Critic Q_ψ   : [N_BS*(KPM_DIM+Z_DIM) + N_BS*N_RB] → 1  (centralised)
"""
import sys, os
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

from envs.cc_env import CCEnv
from baseline import (Cfg, make_snapshot, metrics,
                      bl_wmmse, bl_pf_wmmse, bl_full_power,
                      dbm_to_w, noise_w_per_rb)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
N_BS        = 3
N_RB        = 4
KPM_DIM     = CCEnv.KPM_DIM          # 3
Z_DIM       = 8
WORKER_OBS  = KPM_DIM + N_BS + Z_DIM  # 3+3+8 = 14  (kpm + agent-id onehot + z)
SHARE_OBS   = N_BS * KPM_DIM + Z_DIM  # 9+8 = 17  (all-kpm concat + z)
HIDDEN      = 128
LR          = 3e-4
GAMMA       = 0.99
POLYAK      = 0.005
ALPHA_INIT  = 0.01
BUFFER_SIZE = 100_000
BATCH_SIZE  = 256
WARMUP      = 2_000
TRAIN_EVERY = 10
K_HOLD      = 10
NUM_STEPS   = 300_000
LOG_EVERY   = 10_000
EP_LEN      = 200
SEED        = 42
RESULTS_DIR = "/home/hyc1014/DL/FinalProject/results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent-id one-hot matrix  [N_BS, N_BS]
AGENT_IDS = torch.eye(N_BS, dtype=torch.float32).to(DEVICE)

# ── Modules ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(128, 128)):
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

    def forward(self, kpm):          # kpm: [B, N_BS, KPM_DIM]
        h = self.cell(kpm)           # [B, N_BS, 32]
        z = self.proj(h.mean(dim=1)) # [B, Z_DIM]
        return z


class ValueNorm(nn.Module):
    """Running mean/std normaliser for Q-learning targets (from HARL)."""
    def __init__(self, eps=1e-5, beta=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var",  torch.ones(1))
        self.eps   = eps
        self.beta  = beta  # EMA decay

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

class CCBufferV2:
    """Stores per-step: (kpm, global_kpm, actions, rewards, kpm_next, global_kpm_next, done)"""
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

class CCHASACv2Runner:
    def __init__(self):
        self.encoder  = GlobalContextEncoder().to(DEVICE)
        self.actor    = MLP(WORKER_OBS, N_RB * 2).to(DEVICE)   # parameter-shared
        self.q1       = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q2       = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q1_tgt   = deepcopy(self.q1)
        self.q2_tgt   = deepcopy(self.q2)
        self.vnorm    = ValueNorm().to(DEVICE)

        self.buf = CCBufferV2()
        self.env = CCEnv(seed=SEED)
        self.eval_env = CCEnv(seed=SEED + 9999)

        enc_params    = list(self.encoder.parameters())
        actor_params  = list(self.actor.parameters())
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())

        self.enc_actor_opt = torch.optim.Adam(enc_params + actor_params, lr=LR)
        self.critic_opt    = torch.optim.Adam(critic_params, lr=LR)

        self.log_alpha = torch.tensor(np.log(ALPHA_INIT),
                                      requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -float(N_RB)

    # ── helpers ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_z(self, gkpm_np):
        t = torch.FloatTensor(gkpm_np).unsqueeze(0).to(DEVICE)
        return self.encoder(t).squeeze(0).cpu().numpy()

    def _actor_forward(self, worker_obs_t):
        """worker_obs_t: [B, N_BS, WORKER_OBS] → actions [B, N_BS, N_RB], logp [B, N_BS, 1]"""
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
        a   = (a + 1) / 2                               # → [0,1]
        return a.reshape(B, N_BS, N_RB), lp.reshape(B, N_BS, 1)

    @torch.no_grad()
    def _get_actions(self, kpm_np, z_np):
        ids = AGENT_IDS.cpu().numpy()                   # [N_BS, N_BS]
        z_bc = np.tile(z_np, (N_BS, 1))                # [N_BS, Z_DIM]
        wobs = np.concatenate([kpm_np, ids, z_bc], -1) # [N_BS, WORKER_OBS]
        t = torch.FloatTensor(wobs).unsqueeze(0).to(DEVICE)  # [1, N_BS, WO]
        a, _ = self._actor_forward(t)
        return a.squeeze(0).cpu().numpy()               # [N_BS, N_RB]

    def _build_worker_obs(self, kpm_t, z_t):
        """kpm_t:[B,N_BS,KPM], z_t:[B,Z_DIM] → [B,N_BS,WORKER_OBS]"""
        ids = AGENT_IDS.unsqueeze(0).expand(kpm_t.shape[0], -1, -1)
        z_e = z_t.unsqueeze(1).expand(-1, N_BS, -1)
        return torch.cat([kpm_t, ids, z_e], dim=-1)

    def _build_share_obs(self, kpm_t, z_t):
        """[B,N_BS,KPM], [B,Z_DIM] → [B, SHARE_OBS]"""
        flat_kpm = kpm_t.reshape(kpm_t.shape[0], -1)   # [B, N_BS*KPM_DIM]
        return torch.cat([flat_kpm, z_t], dim=-1)       # [B, SHARE_OBS]

    # ── update ───────────────────────────────────────────────────────────────

    def update(self):
        kpm, gkpm, acts, rews, kpm_n, gkpm_n, dones = self.buf.sample(BATCH_SIZE)

        kpm_t   = torch.FloatTensor(kpm).to(DEVICE)
        gkpm_t  = torch.FloatTensor(gkpm).to(DEVICE)
        acts_t  = torch.FloatTensor(acts).to(DEVICE)          # [B, N_BS, N_RB]
        rews_t  = torch.FloatTensor(rews).to(DEVICE)          # [B, N_BS]
        kpm_n_t = torch.FloatTensor(kpm_n).to(DEVICE)
        gkpm_n_t= torch.FloatTensor(gkpm_n).to(DEVICE)
        done_t  = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        alpha  = self.log_alpha.exp().detach()
        sum_r  = rews_t.sum(dim=1, keepdim=True)              # [B, 1] total reward

        # ── Recompute z (differentiable for actor, detached for critic) ──────
        z      = self.encoder(gkpm_t)                         # [B, Z_DIM] — grad
        z_det  = z.detach()
        with torch.no_grad():
            z_next = self.encoder(gkpm_n_t)

        # Critic uses z detached (stable)
        sobs      = self._build_share_obs(kpm_t,   z_det)    # [B, SHARE_OBS]
        sobs_next = self._build_share_obs(kpm_n_t, z_next)
        wobs_next = self._build_worker_obs(kpm_n_t, z_next)

        # ── Critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            na, nlp = self._actor_forward(wobs_next)
            na_flat = na.reshape(BATCH_SIZE, -1)               # [B, N_BS*N_RB]
            nlp_sum = nlp.sum(dim=1)                           # [B, 1]
            tgt = torch.min(
                self.q1_tgt(torch.cat([sobs_next, na_flat], -1)),
                self.q2_tgt(torch.cat([sobs_next, na_flat], -1)),
            ) - alpha * nlp_sum
            self.vnorm.update(tgt)
            tgt_norm = self.vnorm.normalize(tgt)
            backup = self.vnorm.normalize(
                sum_r + GAMMA * (1 - done_t) * self.vnorm.denormalize(tgt_norm))

        acts_flat = acts_t.reshape(BATCH_SIZE, -1)             # [B, N_BS*N_RB]
        q1_l = ((self.q1(torch.cat([sobs, acts_flat], -1)) - backup) ** 2).mean()
        q2_l = ((self.q2(torch.cat([sobs, acts_flat], -1)) - backup) ** 2).mean()
        if not (q1_l + q2_l).isnan():
            self.critic_opt.zero_grad()
            (q1_l + q2_l).backward()
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
            self.critic_opt.step()

        # ── Actor + Encoder update (z with gradient) ─────────────────────────
        wobs_grad = self._build_worker_obs(kpm_t, z)          # z non-detached
        sobs_grad = self._build_share_obs(kpm_t,  z)

        new_a, lp = self._actor_forward(wobs_grad)
        new_a_flat = new_a.reshape(BATCH_SIZE, -1)
        lp_sum     = lp.sum(dim=1)
        q_val = torch.min(
            self.q1(torch.cat([sobs_grad, new_a_flat], -1)),
            self.q2(torch.cat([sobs_grad, new_a_flat], -1)),
        )
        a_loss = (alpha * lp_sum - q_val).mean()
        if not a_loss.isnan():
            self.enc_actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.actor.parameters()), 10.0)
            self.enc_actor_opt.step()

        # ── Alpha update ──────────────────────────────────────────────────────
        with torch.no_grad():
            _, lp_det = self._actor_forward(wobs_grad.detach())
            lp_det_sum = lp_det.sum(dim=1)
        al = -(self.log_alpha * (lp_det_sum + self.target_entropy * N_BS)).mean()
        self.alpha_opt.zero_grad(); al.backward(); self.alpha_opt.step()

        # Soft update targets
        for p, tp in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, n_episodes=20, zero_z=False, shuffle_z=False):
        """Returns (mean_ep_rew, mean_sum_rate_bps_hz)."""
        ep_rews, sum_rates = [], []
        for _ in range(n_episodes):
            kpm = self.eval_env.reset()
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
                    z = np.zeros(Z_DIM, dtype=np.float32) if zero_z else self._get_z(gkpm)
                    if shuffle_z:
                        z = z[np.random.permutation(Z_DIM)]
                if done:
                    break
            ep_rews.append(ep_r)
            sum_rates.append(sr_sum / steps)
        return float(np.mean(ep_rews)), float(np.mean(sum_rates))

    def _wmmse_baseline_eval(self, n_snapshots=50):
        """Compare against WMMSE / pf_wmmse on held-out snapshots."""
        cfg  = self.env.cfg
        nw   = self.env.nw
        Pmax = self.env.Pmax
        rng  = np.random.default_rng(999)
        results = {"wmmse": [], "pf_wmmse": [], "full_power": []}
        for _ in range(n_snapshots):
            As, assoc = make_snapshot(cfg, rng)
            results["wmmse"].append(
                metrics(bl_wmmse(As, cfg, nw, Pmax, rng),
                        As, assoc, cfg, nw)['sum_rate'])
            results["pf_wmmse"].append(
                metrics(bl_pf_wmmse(As, assoc, cfg, nw, Pmax, rng),
                        As, assoc, cfg, nw)['sum_rate'])
            results["full_power"].append(
                metrics(bl_full_power(As, cfg, nw, Pmax),
                        As, assoc, cfg, nw)['sum_rate'])
        return {k: float(np.mean(v)) for k, v in results.items()}

    # ── Training Loop ─────────────────────────────────────────────────────────

    def run(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "cc_hasac_v2_log.txt")
        rew_path = os.path.join(RESULTS_DIR, "cc_hasac_v2_rewards.npy")

        kpm  = self.env.reset()
        gkpm = self.env.get_global_kpm()
        z    = self._get_z(gkpm)

        ep_rew, done_eps, log_lines = 0.0, [], []
        print(f"cc-HASAC v2 | device={DEVICE} | KPM={KPM_DIM} z={Z_DIM} "
              f"obs={WORKER_OBS}")
        print(f"{'Step':>8}  {'AvgEpRew':>10}  {'SumRate':>10}  {'DoneEps':>8}")

        for step in range(1, NUM_STEPS + 1):
            # ── Collect ──────────────────────────────────────────────────────
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
                kpm    = self.env.reset()
                gkpm   = self.env.get_global_kpm()
                z      = self._get_z(gkpm)

            # ── Train ─────────────────────────────────────────────────────────
            if step > WARMUP and step % TRAIN_EVERY == 0 and len(self.buf) >= BATCH_SIZE:
                self.update()

            # ── Log ───────────────────────────────────────────────────────────
            if step % LOG_EVERY == 0:
                avg_rew, avg_sr = self.evaluate(n_episodes=5)
                line = (f"{step:8d}  {avg_rew:10.2f}  {avg_sr:10.4f}  "
                        f"{len(done_eps):8d}")
                print(line)
                log_lines.append(line)

        np.save(rew_path, np.array(done_eps))
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        # ── z-Ablation Probe ─────────────────────────────────────────────────
        print("\n── z-Ablation Probe (20 episodes) ──")
        r_z,    sr_z    = self.evaluate(n_episodes=20, zero_z=False)
        r_0,    sr_0    = self.evaluate(n_episodes=20, zero_z=True)
        r_shuf, sr_shuf = self.evaluate(n_episodes=20, shuffle_z=True)
        print(f"  With z    : ep_rew={r_z:.2f}  sum_rate={sr_z:.4f}")
        print(f"  z ← 0     : ep_rew={r_0:.2f}  sum_rate={sr_0:.4f}  Δ={r_z-r_0:+.2f}")
        print(f"  z shuffle : ep_rew={r_shuf:.2f}  sum_rate={sr_shuf:.4f}  Δ={r_z-r_shuf:+.2f}")

        # ── WMMSE Comparison (held-out 50 snapshots) ─────────────────────────
        print("\n── Baseline Comparison (50 held-out snapshots) ──")
        bl = self._wmmse_baseline_eval(n_snapshots=50)
        print(f"  full_power : {bl['full_power']:.4f} bps/Hz")
        print(f"  wmmse      : {bl['wmmse']:.4f} bps/Hz")
        print(f"  pf_wmmse   : {bl['pf_wmmse']:.4f} bps/Hz")
        print(f"  cc-HASAC   : {sr_z:.4f} bps/Hz  "
              f"(gap vs wmmse = {100*(bl['wmmse']-sr_z)/bl['wmmse']:+.1f}%)")

        np.save(os.path.join(RESULTS_DIR, "cc_hasac_v2_ablation.npy"),
                np.array([r_z, r_0, r_shuf, sr_z, sr_0, sr_shuf]))
        return done_eps


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    runner = CCHASACv2Runner()
    runner.run()
