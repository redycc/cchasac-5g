"""
cc-HASAC: Context-Conditioned HASAC.

Architecture:
  GlobalContextEncoder f_θ  (DeepSet, permutation-invariant)
    global_KPM [N_BS, kpm_dim] → z ∈ R^Z_DIM
  Worker π_φ_i(o_i ‖ z) → a_i   (local R2 obs + shared z)

Training:
  - z recomputed from global_kpm each update (not stored stale)
  - Encoder receives gradients from sum of all worker actor losses
  - No manager reward, no manager replay, no sub-goal design

Comparison:
  Run after train_r2_flat.py to isolate the contribution of z.
  z-ablation probe runs automatically at end of training.
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque

sys.path.insert(0, "/home/hyc1014/DL/FinalProject/HARL")
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")

from envs.fiveg_env import FiveGEnv

# ─── Hyperparameters ──────────────────────────────────────────────────────────

N_BS          = 3
N_UE          = 10
N_RB          = 4
KPM_DIM       = N_RB + 4          # 8: sinr×4, load, intf, tpt, n_ue
Z_DIM         = 8                  # latent context dim
WORKER_OBS    = KPM_DIM + Z_DIM   # 16
HIDDEN        = 128
LR            = 3e-4
GAMMA         = 0.99
POLYAK        = 0.005
ALPHA_INIT    = 0.01
BUFFER_SIZE   = 100_000
BATCH_SIZE    = 256
WARMUP        = 2_000
TRAIN_EVERY   = 10
K_HOLD        = 10                 # steps between z refreshes
NUM_STEPS     = 300_000
LOG_EVERY     = 10_000
EP_LENGTH     = 200
SEED          = 42
RESULTS_DIR   = "/home/hyc1014/DL/FinalProject/results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Modules ─────────────────────────────────────────────────────────────────

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
    """
    DeepSet encoder: permutation-invariant over BS cells.
    Input:  global_kpm [B, N_BS, kpm_dim]
    Output: z          [B, Z_DIM]
    """
    def __init__(self, kpm_dim=KPM_DIM, z_dim=Z_DIM, hidden=32):
        super().__init__()
        self.cell_enc = nn.Sequential(
            nn.Linear(kpm_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.proj = nn.Linear(hidden, z_dim)

    def forward(self, kpm):          # [B, N_BS, kpm_dim]
        h = self.cell_enc(kpm)       # [B, N_BS, hidden]
        z = self.proj(h.mean(dim=1)) # [B, Z_DIM]
        return z


class WorkerSAC:
    """Minimal SAC for one worker. obs_dim = WORKER_OBS = 16, act_dim = N_RB = 4."""

    def __init__(self, obs_dim=WORKER_OBS, act_dim=N_RB, device=DEVICE):
        self.device  = device
        self.act_dim = act_dim

        self.actor  = MLP(obs_dim, act_dim * 2).to(device)
        self.q1     = MLP(obs_dim + act_dim, 1).to(device)
        self.q2     = MLP(obs_dim + act_dim, 1).to(device)
        self.q1_tgt = deepcopy(self.q1)
        self.q2_tgt = deepcopy(self.q2)

        self.q_opt     = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.log_alpha = torch.tensor(np.log(ALPHA_INIT), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -act_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _policy(self, obs):
        out   = self.actor(obs)
        mean, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = torch.clamp(log_std, -5, 2)
        std   = log_std.exp()
        z     = mean + std * torch.randn_like(mean)
        act   = torch.tanh(z)
        lp    = (
            -((z - mean) ** 2) / (2 * std ** 2 + 1e-8)
            - log_std - 0.5 * np.log(2 * np.pi)
            - torch.log(1 - act.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return (act + 1) / 2, lp   # actions in [0,1]

    @torch.no_grad()
    def get_action(self, obs_np):
        obs = torch.FloatTensor(obs_np).unsqueeze(0).to(self.device)
        out = self.actor(obs)
        mean, log_std = out[:, :self.act_dim], out[:, self.act_dim:]
        log_std = torch.clamp(log_std, -5, 2)
        act = torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
        return ((act + 1) / 2).squeeze(0).cpu().numpy()

    def soft_update(self):
        for p, tp in zip(self.q1.parameters(), self.q1_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_tgt.parameters()):
            tp.data.mul_(1 - POLYAK).add_(POLYAK * p.data)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

class CCBuffer:
    """
    Stores: (local_kpm, global_kpm, actions, rewards, local_kpm_next, global_kpm_next, done)
    local_kpm:  [N_BS, KPM_DIM]
    global_kpm: [N_BS, KPM_DIM]  (same here; kept separate for clarity/future asymmetry)
    actions:    [N_BS, N_RB]
    rewards:    [N_BS]
    done:       scalar bool
    """
    FIELDS = 7

    def __init__(self, cap=BUFFER_SIZE):
        self.buf = deque(maxlen=cap)

    def push(self, local_kpm, global_kpm, actions, rewards,
             local_kpm_next, global_kpm_next, done):
        self.buf.append((
            local_kpm.astype(np.float32),
            global_kpm.astype(np.float32),
            actions.astype(np.float32),
            rewards.astype(np.float32),
            local_kpm_next.astype(np.float32),
            global_kpm_next.astype(np.float32),
            np.float32(done),
        ))

    def sample(self, n):
        idx   = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.stack([b[j] for b in batch]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


# ─── cc-HASAC Runner ─────────────────────────────────────────────────────────

class CCHASACRunner:
    def __init__(self):
        self.encoder = GlobalContextEncoder().to(DEVICE)
        self.workers = [WorkerSAC() for _ in range(N_BS)]
        self.buf     = CCBuffer()
        self.enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=LR)

        env_args = {
            "n_bs": N_BS, "n_ue": N_UE, "n_rb": N_RB,
            "episode_length": EP_LENGTH,
            "hierarchical": False,
            "channel_source": "deepmimo",
            "obs_mode": "r2",
            "deepmimo_cache": "/home/hyc1014/DL/FinalProject/deepmimo_cache",
        }
        self.env      = FiveGEnv(env_args)
        self.eval_env = FiveGEnv(env_args)
        self.env.seed(SEED)
        self.eval_env.seed(SEED + 100)

    @torch.no_grad()
    def _get_z(self, global_kpm_np):
        kpm_t = torch.FloatTensor(global_kpm_np).unsqueeze(0).to(DEVICE)
        return self.encoder(kpm_t).squeeze(0).cpu().numpy()

    def _build_worker_obs(self, local_kpm_np, z_np):
        """Concatenate per-BS local KPM with shared z: list of N_BS arrays (16-dim)."""
        return [np.concatenate([local_kpm_np[i], z_np]) for i in range(N_BS)]

    def update(self):
        (lkpm, gkpm, acts, rews,
         lkpm_n, gkpm_n, dones) = self.buf.sample(BATCH_SIZE)

        # Convert to tensors
        lkpm_t   = torch.FloatTensor(lkpm).to(DEVICE)    # [B, N_BS, kpm]
        gkpm_t   = torch.FloatTensor(gkpm).to(DEVICE)    # [B, N_BS, kpm]
        gkpm_n_t = torch.FloatTensor(gkpm_n).to(DEVICE)
        lkpm_n_t = torch.FloatTensor(lkpm_n).to(DEVICE)
        acts_t   = torch.FloatTensor(acts).to(DEVICE)    # [B, N_BS, N_RB]
        rews_t   = torch.FloatTensor(rews).to(DEVICE)    # [B, N_BS]
        done_t   = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)  # [B, 1]

        # ── Recompute z (differentiable) ──
        z      = self.encoder(gkpm_t)         # [B, Z_DIM]  — grad flows here
        z_det  = z.detach()                   # for critic (stable)
        with torch.no_grad():
            z_next = self.encoder(gkpm_n_t)   # [B, Z_DIM]

        z_exp      = z_det.unsqueeze(1).expand(-1, N_BS, -1)     # [B, N_BS, Z_DIM]
        z_next_exp = z_next.unsqueeze(1).expand(-1, N_BS, -1)
        z_grad_exp = z.unsqueeze(1).expand(-1, N_BS, -1)         # for actor loss

        # worker obs: [B, N_BS, WORKER_OBS]
        wobs      = torch.cat([lkpm_t,   z_exp],      dim=-1)  # detached z
        wobs_grad = torch.cat([lkpm_t,   z_grad_exp], dim=-1)  # gradient z
        wobs_next = torch.cat([lkpm_n_t, z_next_exp], dim=-1)

        total_actor_loss = torch.tensor(0.0, device=DEVICE)

        for i, worker in enumerate(self.workers):
            obs_i      = wobs[:, i, :]       # [B, WORKER_OBS]  critic input
            obs_grad_i = wobs_grad[:, i, :]  # [B, WORKER_OBS]  actor input (grad to enc)
            obs_next_i = wobs_next[:, i, :]
            act_i      = acts_t[:, i, :]
            rew_i      = rews_t[:, i:i+1]

            # ── Critic update (z detached) ──────────────────────────────────
            with torch.no_grad():
                na, nlp = worker._policy(obs_next_i)
                tgt = torch.min(
                    worker.q1_tgt(torch.cat([obs_next_i, na], -1)),
                    worker.q2_tgt(torch.cat([obs_next_i, na], -1)),
                ) - worker.alpha.detach() * nlp
                backup = rew_i + GAMMA * (1 - done_t) * tgt

            q1_l = ((worker.q1(torch.cat([obs_i, act_i], -1)) - backup) ** 2).mean()
            q2_l = ((worker.q2(torch.cat([obs_i, act_i], -1)) - backup) ** 2).mean()
            if not (q1_l + q2_l).isnan():
                worker.q_opt.zero_grad()
                (q1_l + q2_l).backward()
                nn.utils.clip_grad_norm_(
                    list(worker.q1.parameters()) + list(worker.q2.parameters()), 10.0)
                worker.q_opt.step()

            # ── Actor loss (z gradient-connected to encoder) ─────────────────
            new_a, lp = worker._policy(obs_grad_i)
            a_l = (worker.alpha.detach() * lp - torch.min(
                worker.q1(torch.cat([obs_i.detach(), new_a], -1)),
                worker.q2(torch.cat([obs_i.detach(), new_a], -1)),
            )).mean()
            if not a_l.isnan():
                total_actor_loss = total_actor_loss + a_l

            # ── Alpha update ────────────────────────────────────────────────
            with torch.no_grad():
                _, lp_det = worker._policy(obs_grad_i)
            al = -(worker.log_alpha * (lp_det + worker.target_entropy)).mean()
            worker.alpha_opt.zero_grad(); al.backward(); worker.alpha_opt.step()

            worker.soft_update()

        # ── Encoder update via total actor loss ──────────────────────────────
        if not total_actor_loss.isnan():
            self.enc_opt.zero_grad()
            for w in self.workers:
                w.actor_opt.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 10.0)
            for w in self.workers:
                nn.utils.clip_grad_norm_(w.actor.parameters(), 10.0)
            self.enc_opt.step()
            for w in self.workers:
                w.actor_opt.step()

    def evaluate(self, n_episodes=5, zero_z=False):
        """Evaluate mean episode reward. zero_z=True runs ablation (z←0)."""
        ep_rewards = []
        for _ in range(n_episodes):
            obs, _, _ = self.eval_env.reset()
            local_kpm = self.eval_env.get_global_kpm()
            z = np.zeros(Z_DIM, dtype=np.float32) if zero_z else self._get_z(local_kpm)
            ep_rew, step = 0.0, 0
            while True:
                wobs = self._build_worker_obs(local_kpm, z)
                actions = np.array([self.workers[i].get_action(wobs[i]) for i in range(N_BS)])
                obs, _, rews, dones, _, _ = self.eval_env.step(actions)
                local_kpm = self.eval_env.get_global_kpm()
                ep_rew += sum(r[0] for r in rews)
                step += 1
                if step % K_HOLD == 0:
                    z = np.zeros(Z_DIM, dtype=np.float32) if zero_z else self._get_z(local_kpm)
                if dones[0]:
                    break
            ep_rewards.append(ep_rew)
        return float(np.mean(ep_rewards))

    def run(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_path = os.path.join(RESULTS_DIR, "cc_hasac_log.txt")
        rew_path = os.path.join(RESULTS_DIR, "cc_hasac_rewards.npy")

        obs, _, _ = self.env.reset()
        local_kpm = self.env.get_global_kpm()
        z = self._get_z(local_kpm)

        ep_rew      = 0.0
        ep_step     = 0
        done_eps    = []
        log_lines   = []

        print(f"cc-HASAC training | device={DEVICE} | KPM={KPM_DIM} z={Z_DIM}")
        print(f"{'Step':>8}  {'Ep Reward':>12}  {'Done EPs':>9}")

        for step in range(1, NUM_STEPS + 1):

            # ── Collect ──────────────────────────────────────────────────────
            if step <= WARMUP:
                actions = np.array([
                    self.env.action_space[i].sample() for i in range(N_BS)])
            else:
                wobs    = self._build_worker_obs(local_kpm, z)
                actions = np.array([
                    self.workers[i].get_action(wobs[i]) for i in range(N_BS)])

            next_obs, _, rews, dones, _, _ = self.env.step(actions)
            next_kpm = self.env.get_global_kpm()

            ep_rew  += sum(r[0] for r in rews)
            ep_step += 1

            self.buf.push(
                local_kpm, local_kpm,      # local ≡ global in current 3-BS setup
                actions,
                np.array([r[0] for r in rews]),
                next_kpm, next_kpm,
                float(dones[0]),
            )

            obs       = next_obs
            local_kpm = next_kpm

            # Refresh z every K steps
            if step % K_HOLD == 0:
                z = self._get_z(local_kpm)

            if dones[0]:
                done_eps.append(ep_rew)
                ep_rew = ep_step = 0
                obs, _, _  = self.env.reset()
                local_kpm  = self.env.get_global_kpm()
                z          = self._get_z(local_kpm)

            # ── Train ─────────────────────────────────────────────────────────
            if step > WARMUP and step % TRAIN_EVERY == 0 and len(self.buf) >= BATCH_SIZE:
                self.update()

            # ── Log ───────────────────────────────────────────────────────────
            if step % LOG_EVERY == 0:
                avg = float(np.mean(done_eps[-20:])) if done_eps else float("nan")
                line = f"{step:8d}  {avg:12.2f}  {len(done_eps):9d}"
                print(line)
                log_lines.append(line)

        # ── Save ─────────────────────────────────────────────────────────────
        np.save(rew_path, np.array(done_eps))
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))
        print(f"\nTraining done. Rewards saved to {rew_path}")

        # ── z-Ablation Probe ─────────────────────────────────────────────────
        print("\n── z-Ablation Probe (20 episodes) ──")
        r_with_z = self.evaluate(n_episodes=20, zero_z=False)
        r_zero_z = self.evaluate(n_episodes=20, zero_z=True)
        delta    = r_with_z - r_zero_z
        print(f"  With z  : {r_with_z:.2f}")
        print(f"  z ← 0   : {r_zero_z:.2f}")
        print(f"  Δ       : {delta:+.2f}  ({'z is useful ✓' if delta > 5 else 'z marginal — check R2 obs'})")

        ablation_path = os.path.join(RESULTS_DIR, "cc_hasac_ablation.npy")
        np.save(ablation_path, np.array([r_with_z, r_zero_z, delta]))
        return done_eps


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    runner = CCHASACRunner()
    done_eps = runner.run()
    final_raw_ep = float(np.mean(done_eps[-50:])) if len(done_eps) >= 50 else float(np.mean(done_eps)) if done_eps else float("nan")
    import sys, os as _os; sys.path.insert(0, _os.path.dirname(__file__))
    from log_experiment import log_experiment
    log_experiment("cc-HASAC", final_raw_ep, note="DeepSet encoder, z broadcast")
