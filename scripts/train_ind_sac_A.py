"""
Independent SAC baseline (NO z) — standalone PyTorch.

Diagnostic control for cc-HASAC: identical CCEnv (R2 KPM-only obs), identical
reward from baseline.metrics(), parameter-shared actor, agent-id one-hot.
NO encoder, NO context z.

Fix applied (shared with cc-HASAC v3): SNAPSHOT POOL.
  Instead of generating a fresh random channel snapshot every episode, we
  pre-generate N_POOL=50 snapshots at init and sample from the pool on reset.
  Eval uses a SEPARATE held-out pool (different seed, N=20).

Worker obs = [kpm(3), agent_id_onehot(3)] = 6-dim.
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
KPM_DIM     = CCEnv.KPM_DIM            # 7  (R1-partial)
WORKER_OBS  = KPM_DIM + N_BS          # 7 + 3 = 10  (obs + agent-id onehot)
SHARE_OBS   = N_BS * KPM_DIM          # 21 (all-obs concat, no z)
HIDDEN      = 128
LR          = 3e-4
GAMMA       = 0.99
POLYAK      = 0.005
ALPHA_INIT  = 0.01
BUFFER_SIZE = 100_000
BATCH_SIZE  = 256
WARMUP      = 2_000
TRAIN_EVERY = 10
NUM_STEPS   = 300_000
LOG_EVERY   = 10_000
EP_LEN      = 200
N_POOL      = 50
SEED        = 42
RESULTS_DIR = "/home/hyc1014/DL/FinalProject/results"

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

class IndBuffer:
    """Stores per-step: (kpm, actions, rewards, kpm_next, done)"""
    FIELDS = 5

    def __init__(self, cap=BUFFER_SIZE):
        self.buf = deque(maxlen=cap)

    def push(self, kpm, acts, rews, kpm_n, done):
        self.buf.append((
            kpm.astype(np.float32),
            acts.astype(np.float32),
            rews.astype(np.float32),
            kpm_n.astype(np.float32),
            np.float32(done),
        ))

    def sample(self, n):
        idx   = np.random.choice(len(self.buf), n, replace=False)
        batch = [self.buf[i] for i in idx]
        return [np.stack([b[j] for b in batch]) for j in range(self.FIELDS)]

    def __len__(self):
        return len(self.buf)


# ── Runner ────────────────────────────────────────────────────────────────────

class IndSACRunner:
    def __init__(self):
        self.actor  = MLP(WORKER_OBS, N_RB * 2).to(DEVICE)        # parameter-shared
        self.q1     = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q2     = MLP(SHARE_OBS + N_BS * N_RB, 1).to(DEVICE)
        self.q1_tgt = deepcopy(self.q1)
        self.q2_tgt = deepcopy(self.q2)
        self.vnorm  = ValueNorm().to(DEVICE)

        self.buf      = IndBuffer()
        self.env      = CCEnv(seed=SEED)
        self.eval_env = CCEnv(seed=SEED + 9999)

        # ── Snapshot pools (FIX) ───────────────────────────────────────────────
        cfg = self.env.cfg
        rng_pool        = np.random.default_rng(SEED + 1)
        self._train_pool = [make_snapshot(cfg, rng_pool) for _ in range(N_POOL)]
        self._eval_pool  = [make_snapshot(cfg, np.random.default_rng(9999))
                            for _ in range(20)]

        actor_params  = list(self.actor.parameters())
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.actor_opt  = torch.optim.Adam(actor_params,  lr=LR)
        self.critic_opt = torch.optim.Adam(critic_params, lr=LR)

        self.log_alpha = torch.tensor(np.log(ALPHA_INIT),
                                      requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)
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

    # ── actor helpers ──────────────────────────────────────────────────────────

    def _actor_forward(self, worker_obs_t):
        """[B, N_BS, WORKER_OBS] → actions [B,N_BS,N_RB], logp [B,N_BS,1]"""
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
    def _get_actions(self, kpm_np):
        ids  = AGENT_IDS.cpu().numpy()
        wobs = np.concatenate([kpm_np, ids], -1)            # [N_BS, WORKER_OBS]
        t = torch.FloatTensor(wobs).unsqueeze(0).to(DEVICE)
        a, _ = self._actor_forward(t)
        return a.squeeze(0).cpu().numpy()

    def _build_worker_obs(self, kpm_t):
        ids = AGENT_IDS.unsqueeze(0).expand(kpm_t.shape[0], -1, -1)
        return torch.cat([kpm_t, ids], dim=-1)

    def _build_share_obs(self, kpm_t):
        return kpm_t.reshape(kpm_t.shape[0], -1)            # [B, N_BS*KPM_DIM]

    # ── update ───────────────────────────────────────────────────────────────

    def update(self):
        kpm, acts, rews, kpm_n, dones = self.buf.sample(BATCH_SIZE)

        kpm_t   = torch.FloatTensor(kpm).to(DEVICE)
        acts_t  = torch.FloatTensor(acts).to(DEVICE)
        rews_t  = torch.FloatTensor(rews).to(DEVICE)
        kpm_n_t = torch.FloatTensor(kpm_n).to(DEVICE)
        done_t  = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        alpha = self.log_alpha.exp().detach()
        sum_r = rews_t.sum(dim=1, keepdim=True)

        sobs      = self._build_share_obs(kpm_t)
        sobs_next = self._build_share_obs(kpm_n_t)
        wobs_next = self._build_worker_obs(kpm_n_t)

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

        # ── Actor update ───────────────────────────────────────────────────────
        wobs_grad = self._build_worker_obs(kpm_t)
        sobs_grad = self._build_share_obs(kpm_t)
        new_a, lp = self._actor_forward(wobs_grad)
        new_a_flat = new_a.reshape(BATCH_SIZE, -1)
        lp_sum     = lp.sum(dim=1)
        q_val = torch.min(
            self.q1(torch.cat([sobs_grad, new_a_flat], -1)),
            self.q2(torch.cat([sobs_grad, new_a_flat], -1)),
        )
        a_loss = (alpha * lp_sum - q_val).mean()
        if not a_loss.isnan():
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()), 10.0)
            self.actor_opt.step()

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

    def evaluate(self, n_episodes=20):
        """Eval on held-out snapshot pool. Returns (mean_ep_rew, mean_sum_rate)."""
        ep_rews, sum_rates = [], []
        n_episodes = min(n_episodes, len(self._eval_pool))
        for ep in range(n_episodes):
            kpm = self._reset_eval(ep)
            ep_r, sr_sum, steps = 0.0, 0.0, 0
            while True:
                acts = self._get_actions(kpm)
                kpm, rews, done, m = self.eval_env.step(acts)
                ep_r   += rews.sum()
                sr_sum += m['sum_rate']
                steps  += 1
                if done:
                    break
            ep_rews.append(ep_r)
            sum_rates.append(sr_sum / steps)
        return float(np.mean(ep_rews)), float(np.mean(sum_rates))

    def _baseline_eval(self):
        """WMMSE / pf_wmmse / full_power on the SAME held-out eval pool."""
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
        log_path = os.path.join(RESULTS_DIR, "ind_sac_A_log.txt")
        rew_path = os.path.join(RESULTS_DIR, "ind_sac_A_rewards.npy")

        kpm = self._reset_train()
        ep_rew, done_eps, log_lines = 0.0, [], []
        print(f"Independent SAC A (R1-partial, no z) | device={DEVICE} | obs={WORKER_OBS} "
              f"| pool={N_POOL}", flush=True)
        print(f"{'Step':>8}  {'AvgEpRew':>10}  {'SumRate':>10}  {'DoneEps':>8}",
              flush=True)

        for step in range(1, NUM_STEPS + 1):
            if step <= WARMUP:
                acts = np.random.uniform(0, 1, (N_BS, N_RB)).astype(np.float32)
            else:
                acts = self._get_actions(kpm)

            kpm_n, rews, done, _ = self.env.step(acts)
            ep_rew += rews.sum()
            self.buf.push(kpm, acts, rews, kpm_n, float(done))
            kpm = kpm_n

            if done:
                done_eps.append(ep_rew)
                ep_rew = 0.0
                kpm = self._reset_train()

            if step > WARMUP and step % TRAIN_EVERY == 0 and len(self.buf) >= BATCH_SIZE:
                self.update()

            if step % LOG_EVERY == 0:
                avg_rew, avg_sr = self.evaluate(n_episodes=5)
                line = (f"{step:8d}  {avg_rew:10.2f}  {avg_sr:10.4f}  "
                        f"{len(done_eps):8d}")
                print(line, flush=True)
                log_lines.append(line)

        np.save(rew_path, np.array(done_eps))
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

        # ── Final held-out evaluation (20 snapshots) ────────────────────────────
        print("\n── Final Held-out Evaluation (20 snapshots, 200 steps) ──",
              flush=True)
        r, sr = self.evaluate(n_episodes=20)
        print(f"  Independent SAC : ep_rew={r:.2f}  sum_rate={sr:.4f} bps/Hz",
              flush=True)

        print("\n── Baseline Comparison (same 20 held-out snapshots) ──",
              flush=True)
        bl = self._baseline_eval()
        print(f"  full_power : {bl['full_power']:.4f} bps/Hz", flush=True)
        print(f"  wmmse      : {bl['wmmse']:.4f} bps/Hz", flush=True)
        print(f"  pf_wmmse   : {bl['pf_wmmse']:.4f} bps/Hz", flush=True)
        print(f"  ind-SAC    : {sr:.4f} bps/Hz", flush=True)
        print(f"\n  Reference  : WMMSE=87.9  full_power=27.3 (project constants)",
              flush=True)
        print(f"  gap vs wmmse(eval) = "
              f"{100*(bl['wmmse']-sr)/bl['wmmse']:+.1f}%", flush=True)

        np.save(os.path.join(RESULTS_DIR, "ind_sac_A_final.npy"),
                np.array([r, sr, bl['full_power'], bl['wmmse'], bl['pf_wmmse']]))
        return done_eps


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    IndSACRunner().run()
