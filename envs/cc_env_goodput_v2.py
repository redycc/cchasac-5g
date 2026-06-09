"""
CCEnvGoodputV2: QoS-aware dynamic env with drift-plus-penalty reward.

Changes from cc_env_goodput.py (v1):
  1. Reward = log(1+thr_i) - β*(Q_i/Q_ref) - η*power_i  (Lyapunov drift-plus-penalty)
     vs old: goodput_i - lambda*intf + 0.1*jain
  2. OBS adds per-BS HOL delay (9-dim vs 8-dim)
  3. Exposes bs_pos for encoder distance-biased attention
  4. Reports P50/P99 delay in step info

Observation per BS i (9-dim):
  [sinr_norm_rb0..3, load, goodput_norm, buf_fullness, n_ue, hol_delay_norm]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from baseline import (Cfg, gen_topology, path_gain, associate,
                      dbm_to_w, noise_w_per_rb)

_log2_1e4 = np.log2(1.0 + 1e4)
EMA_ALPHA  = 0.1


class CCEnvGoodputV2:
    OBS_DIM = 9        # sinr×4 + load + goodput + buf_fullness + n_ue + hol_delay
    KPM_DIM = OBS_DIM

    ARRIVE_RATE = 3.0
    BUF_MAX     = 30.0
    WALK_SPEED  = 3.0

    # Lyapunov reward params
    BETA        = 0.3   # backlog penalty weight
    ETA         = 0.01  # power cost weight
    Q_REF       = 15.0  # reference queue level (half of BUF_MAX)

    def __init__(self, cfg=None, episode_length=200, seed=42):
        self.cfg   = cfg if cfg is not None else Cfg(freq_selective=True)
        self.elen  = episode_length
        self.rng   = np.random.default_rng(seed)
        self.nw    = noise_w_per_rb()
        self.Pmax  = dbm_to_w(self.cfg.Pmax_dBm)
        self._norm = self.cfg.N_RB * _log2_1e4

        self.bs_pos  = None
        self.ue_pos  = None
        self.G       = None
        self.assoc   = None
        self.buf     = None
        self.age     = None    # per-UE HOL delay (steps)
        self.rate_avg = None   # per-UE EMA rate
        self._kpm    = np.zeros((self.cfg.N_BS, self.OBS_DIM), dtype=np.float32)
        self._step   = 0

    # ── MARL interface ─────────────────────────────────────────────────────────

    def reset(self):
        cfg = self.cfg
        self.bs_pos, self.ue_pos = gen_topology(cfg, self.rng)
        self.G     = path_gain(cfg, self.bs_pos, self.ue_pos, self.rng)
        self.assoc = associate(self.G)
        self.buf   = np.zeros(cfg.N_UE, dtype=np.float32)
        self.age   = np.zeros(cfg.N_UE, dtype=np.float32)
        self.rate_avg = np.ones(cfg.N_UE, dtype=np.float32) * self.ARRIVE_RATE
        self._step = 0
        P0 = np.full((cfg.N_BS, cfg.N_RB), self.Pmax * 0.5)
        self._kpm = self._build_obs(P0, np.zeros(cfg.N_BS, dtype=np.float32),
                                    np.zeros(cfg.N_BS, dtype=np.float32))
        return self._kpm.copy()

    def get_bs_pos(self):
        """Return BS positions [N_BS, 2] for distance-biased attention."""
        return self.bs_pos.copy()

    def step(self, actions):
        cfg = self.cfg
        P   = np.clip(actions, 0.0, 1.0) * self.Pmax

        rate_ue = self._per_ue_rate(P)

        # HOL age update: +1 if buffer non-empty, else 0
        self.age = np.where(self.buf > 0, self.age + 1, 0.0).astype(np.float32)

        # Serve from buffer
        served = np.minimum(rate_ue, self.buf)

        # Update EMA rate
        self.rate_avg = ((1 - EMA_ALPHA) * self.rate_avg
                         + EMA_ALPHA * served).astype(np.float32)

        # Buffer update
        arrivals = self.rng.poisson(self.ARRIVE_RATE, size=cfg.N_UE).astype(np.float32)
        self.buf = np.clip(self.buf - served + arrivals, 0.0, self.BUF_MAX)

        # Per-BS goodput
        goodput_per_bs = np.zeros(cfg.N_BS, dtype=np.float32)
        age_per_bs     = np.zeros(cfg.N_BS, dtype=np.float32)
        for i in range(cfg.N_BS):
            mask = (self.assoc == i)
            if mask.any():
                goodput_per_bs[i] = served[mask].sum()
                age_per_bs[i]     = self.age[mask].mean()

        # Drift-plus-penalty reward
        rews = self._reward(P, goodput_per_bs, age_per_bs)

        # UE walk
        self.ue_pos += self.rng.normal(0.0, self.WALK_SPEED,
                                       self.ue_pos.shape).astype(np.float32)
        self.ue_pos  = np.clip(self.ue_pos, 0.0, cfg.area)
        self.G       = path_gain(cfg, self.bs_pos, self.ue_pos, self.rng)
        self.assoc   = associate(self.G)

        obs = self._build_obs(P, goodput_per_bs, age_per_bs)
        self._kpm  = obs
        self._step += 1
        done = self._step >= self.elen

        info = {
            'goodput':  float(goodput_per_bs.sum()),
            'sum_rate': float(rate_ue.sum()),
            'buf_mean': float(self.buf.mean()),
            'hol_mean': float(self.age.mean()),
            'hol_p99':  float(np.percentile(self.age, 99)) if len(self.age) > 0 else 0.0,
        }
        return obs, rews, done, info

    def get_global_kpm(self):
        return self._kpm.copy()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _per_ue_rate(self, P):
        cfg  = self.cfg
        n_ue = cfg.N_UE
        received = P[:, :, None] * self.G[:, None, :]    # [N_BS, N_RB, N_UE]
        total    = received.sum(axis=0)
        desired  = received[self.assoc, :, np.arange(n_ue)].T
        intf     = total - desired
        sinr     = desired / (self.nw + intf + 1e-20)

        rate = np.zeros(n_ue, dtype=np.float64)
        for i in range(cfg.N_BS):
            mask = (self.assoc == i)
            n_i  = mask.sum()
            if n_i == 0: continue
            rb_frac     = cfg.N_RB / float(n_i)
            rate_per_rb = np.log2(1.0 + np.maximum(sinr[:, mask], 0.0))
            rate[mask]  = rb_frac * rate_per_rb.mean(axis=0)
        return rate.astype(np.float32)

    def _build_obs(self, P, goodput_per_bs, age_per_bs):
        cfg = self.cfg
        obs = np.empty((cfg.N_BS, self.OBS_DIM), dtype=np.float32)
        for i in range(cfg.N_BS):
            mask   = (self.assoc == i)
            n_ue_i = mask.sum()

            sinr_vals = np.zeros(cfg.N_RB, dtype=np.float64)
            if n_ue_i > 0:
                G_des = self.G[i, mask].mean()
                for rb in range(cfg.N_RB):
                    sig  = P[i, rb] * G_des
                    intf = sum(P[j, rb] * self.G[j, mask].mean()
                               for j in range(cfg.N_BS) if j != i)
                    sinr_vals[rb] = sig / (self.nw + intf + 1e-20)
            sinr_norm = np.clip(np.log2(1.0 + sinr_vals) / 10.0, -1.0, 1.0)

            load_n  = float(np.clip((P[i] / self.Pmax).mean(), 0, 1)) * 2 - 1
            gput_n  = float(np.clip(goodput_per_bs[i] / self._norm, 0, 1)) * 2 - 1
            buf_n   = (float(self.buf[mask].mean() / self.BUF_MAX)
                       if n_ue_i > 0 else 0.0) * 2 - 1
            n_ue_n  = float(np.clip(n_ue_i / cfg.N_UE, 0, 1)) * 2 - 1
            hol_n   = float(np.clip(age_per_bs[i] / (self.elen + 1), 0, 1)) * 2 - 1

            obs[i] = np.concatenate([sinr_norm,
                                      [load_n, gput_n, buf_n, n_ue_n, hol_n]])
        return obs

    def _reward(self, P, goodput_per_bs, age_per_bs):
        cfg = self.cfg
        rews = np.zeros(cfg.N_BS, dtype=np.float32)
        for i in range(cfg.N_BS):
            mask = (self.assoc == i)
            # Utility: proportional-fair throughput
            utility = float(np.log(1.0 + goodput_per_bs[i] + 1e-6))
            # Backlog penalty: queue pressure
            q_i     = (self.buf[mask].mean() / self.Q_REF
                       if mask.any() else 0.0)
            # Power cost
            power_i = float(P[i].mean() / self.Pmax)
            rews[i] = utility - self.BETA * q_i - self.ETA * power_i
        return rews.astype(np.float32)

    def compute_bs_distances(self):
        """Pairwise BS distances [N_BS, N_BS], normalised by area diagonal."""
        n = self.cfg.N_BS
        d = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                d[i, j] = float(np.linalg.norm(self.bs_pos[i] - self.bs_pos[j]))
        diag = float(self.cfg.area) * 1.414
        return d / (diag + 1e-6)
