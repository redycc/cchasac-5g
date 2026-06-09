"""
CCEnvGoodput: Dynamic environment with UE random walk + per-UE traffic buffer.

Why this matters:
  Static snapshot + sum-rate → RL has no sequential credit-assignment advantage
  over WMMSE. With dynamic channel and buffer, RL can learn to anticipate future
  congestion and coordinate proactively — giving it genuine room to improve over
  myopic baselines.

Key differences from cc_env_r1partial.py:
  - UE positions evolve via Gaussian random walk each step → dynamic channel
  - Per-UE traffic buffer: Poisson arrivals, served up to instantaneous rate
  - Reward = goodput (bits actually delivered) not instantaneous sum-rate
  - Observation adds buf_fullness → OBS_DIM = 8 (vs 7)

Observation per BS i (8-dim):
  [sinr_norm_rb0..3, load_norm, goodput_norm, buf_fullness_norm, n_ue_norm]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from baseline import (Cfg, gen_topology, path_gain, associate,
                      dbm_to_w, noise_w_per_rb)

_log2_1e4 = np.log2(1.0 + 1e4)


class CCEnvGoodput:
    OBS_DIM = 8        # sinr×4 + load + goodput + buf_fullness + n_ue
    KPM_DIM = OBS_DIM

    ARRIVE_RATE = 3.0  # bits/slot/UE (Poisson mean); ~50% UEs capacity-limited with RB sharing
    BUF_MAX     = 30.0 # max buffer per UE (~10 steps of arrivals)
    WALK_SPEED  = 3.0  # Gaussian std (metres/step); pedestrian ≈ 1.4 m/s

    def __init__(self, cfg=None, episode_length=200, seed=42):
        self.cfg   = cfg if cfg is not None else Cfg(freq_selective=True)
        self.elen  = episode_length
        self.rng   = np.random.default_rng(seed)
        self.nw    = noise_w_per_rb()
        self.Pmax  = dbm_to_w(self.cfg.Pmax_dBm)
        self._norm = self.cfg.N_RB * _log2_1e4

        self.bs_pos = None   # [N_BS, 2]  fixed per episode
        self.ue_pos = None   # [N_UE, 2]  moves each step
        self.G      = None   # [N_BS, N_UE] current path gains
        self.assoc  = None   # [N_UE]
        self.buf    = None   # [N_UE] current buffer (bits, normalised)
        self._kpm   = np.zeros((self.cfg.N_BS, self.OBS_DIM), dtype=np.float32)
        self._step  = 0

    # ── MARL interface ────────────────────────────────────────────────────────

    def reset(self):
        cfg = self.cfg
        self.bs_pos, self.ue_pos = gen_topology(cfg, self.rng)
        self.G     = path_gain(cfg, self.bs_pos, self.ue_pos, self.rng)
        self.assoc = associate(self.G)
        self.buf   = np.zeros(cfg.N_UE, dtype=np.float32)
        self._step = 0
        P0 = np.full((cfg.N_BS, cfg.N_RB), self.Pmax * 0.5)
        self._kpm  = self._build_obs(P0, np.zeros(cfg.N_BS, dtype=np.float32))
        return self._kpm.copy()

    def step(self, actions):
        cfg = self.cfg
        P   = np.clip(actions, 0.0, 1.0) * self.Pmax   # [N_BS, N_RB]

        # 1. Per-UE achievable rate on current channel
        rate_ue = self._per_ue_rate(P)   # [N_UE] bits/slot

        # 2. Serve from buffer (can't serve more than buffer holds)
        served = np.minimum(rate_ue, self.buf)

        # 3. Buffer update: subtract served, add Poisson arrivals, clip
        arrivals   = self.rng.poisson(self.ARRIVE_RATE, size=cfg.N_UE).astype(np.float32)
        self.buf   = np.clip(self.buf - served + arrivals, 0.0, self.BUF_MAX)

        # 4. Per-BS goodput (for reward + obs)
        goodput_per_bs = np.zeros(cfg.N_BS, dtype=np.float32)
        for i in range(cfg.N_BS):
            mask = (self.assoc == i)
            if mask.any():
                goodput_per_bs[i] = served[mask].sum()

        # 5. Shaped reward
        rews = self._per_agent_reward(P, goodput_per_bs)

        # 6. Move UEs (Gaussian random walk, reflecting boundary)
        self.ue_pos += self.rng.normal(0.0, self.WALK_SPEED,
                                       self.ue_pos.shape).astype(np.float32)
        self.ue_pos  = np.clip(self.ue_pos, 0.0, cfg.area)

        # 7. Recompute channel with new positions
        self.G     = path_gain(cfg, self.bs_pos, self.ue_pos, self.rng)
        self.assoc = associate(self.G)

        # 8. Build obs with updated channel + buffer
        obs = self._build_obs(P, goodput_per_bs)
        self._kpm  = obs
        self._step += 1
        done = self._step >= self.elen

        info = {
            'goodput':  float(goodput_per_bs.sum()),
            'sum_rate': float(rate_ue.sum()),
            'buf_mean': float(self.buf.mean()),
            'buf_full': float((self.buf >= self.BUF_MAX * 0.95).mean()),
        }
        return obs, rews, done, info

    def get_global_kpm(self):
        return self._kpm.copy()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _per_ue_rate(self, P):
        """Per-UE achievable rate with equal intra-cell RB sharing.

        Each UE in BS i gets N_RB/N_UE_i fraction of the RBs (round-robin
        equivalent). This gives realistic per-UE capacity ~2-4 bits/step and
        puts the system in a capacity-limited regime with ARRIVE_RATE=3.

        Returns [N_UE] bits/slot.
        """
        cfg  = self.cfg
        n_ue = cfg.N_UE

        # received[j, rb, k] = P[j, rb] * G[j, k]
        received = P[:, :, None] * self.G[:, None, :]    # [N_BS, N_RB, N_UE]
        total    = received.sum(axis=0)                   # [N_RB, N_UE]

        # desired signal from serving BS
        desired = received[self.assoc, :, np.arange(n_ue)].T  # [N_RB, N_UE]

        intf = total - desired
        sinr = desired / (self.nw + intf + 1e-20)         # [N_RB, N_UE]

        # Equal RB sharing: each UE gets N_RB / n_ue_per_bs fraction of RBs
        rate = np.zeros(n_ue, dtype=np.float64)
        for i in range(cfg.N_BS):
            mask  = (self.assoc == i)
            n_i   = mask.sum()
            if n_i == 0:
                continue
            rb_frac  = cfg.N_RB / float(n_i)          # expected RBs per UE
            rate_per_rb = np.log2(1.0 + np.maximum(sinr[:, mask], 0.0))
            # average over RBs (since no per-RB fading, all RBs equal)
            rate[mask] = rb_frac * rate_per_rb.mean(axis=0)
        return rate.astype(np.float32)

    def _build_obs(self, P, goodput_per_bs):
        cfg = self.cfg
        obs = np.empty((cfg.N_BS, self.OBS_DIM), dtype=np.float32)

        for i in range(cfg.N_BS):
            mask = (self.assoc == i)
            n_ue_i = mask.sum()

            # Per-RB SINR for BS i (averaged over its UEs)
            sinr_vals = np.zeros(cfg.N_RB, dtype=np.float64)
            if n_ue_i > 0:
                G_des  = self.G[i, mask].mean()
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

            obs[i] = np.concatenate([sinr_norm, [load_n, gput_n, buf_n, n_ue_n]])
        return obs

    def _per_agent_reward(self, P, goodput_per_bs):
        cfg = self.cfg
        # Interference caused by BS i to other BSs' UEs (summed over RBs)
        intf_caused = np.zeros(cfg.N_BS, dtype=np.float32)
        for i in range(cfg.N_BS):
            for j in range(cfg.N_BS):
                if j == i:
                    continue
                mask = (self.assoc == j)
                if mask.any():
                    intf_caused[i] += (P[i] * self.G[i, mask].mean()).sum()

        # Jain fairness over per-UE goodput
        n_ue_per_bs  = np.array([(self.assoc == i).sum()
                                  for i in range(cfg.N_BS)], dtype=float)
        n_ue_per_bs  = np.maximum(n_ue_per_bs, 1.0)
        per_ue_gput  = goodput_per_bs / n_ue_per_bs
        jain_num     = per_ue_gput.sum() ** 2
        jain_den     = cfg.N_BS * (per_ue_gput ** 2).sum() + 1e-20
        jain         = jain_num / jain_den

        r = (goodput_per_bs
             - cfg.lam * intf_caused
             + 0.1 * jain / cfg.N_BS) / self._norm
        return r.astype(np.float32)
