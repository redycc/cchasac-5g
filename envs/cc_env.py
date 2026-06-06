"""
CCEnv: R2-regime environment for Context-Conditioned HASAC.

Observation (R2 KPM-only, per HANDOFF.MD §6.1):
  [tput_norm, prb_util_norm, n_ue_norm]  =  3-dim  (NO SINR, NO CSI)
  "上輪 DRB.UEThpDl / RRU.PrbTot / RRC.ConnMean"

Reward: from baselines.baseline.metrics() — the single source of truth.

Channel: baselines.make_snapshot() (formula UMi), new snapshot each episode.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from baseline import Cfg, make_snapshot, metrics, dbm_to_w, noise_w_per_rb

_log2_1e4 = np.log2(1.0 + 1e4)


class CCEnv:
    KPM_DIM = 3   # [tput_norm, prb_util_norm, n_ue_norm]

    def __init__(self, cfg=None, episode_length=200, seed=42):
        self.cfg   = cfg if cfg is not None else Cfg()
        self.elen  = episode_length
        self.rng   = np.random.default_rng(seed)
        self.nw    = noise_w_per_rb()
        self.Pmax  = dbm_to_w(self.cfg.Pmax_dBm)
        self._norm = self.cfg.N_RB * _log2_1e4

        self.As        = None
        self.assoc     = None
        self._kpm      = np.zeros((self.cfg.N_BS, self.KPM_DIM), dtype=np.float32)
        self._step     = 0

    # ── MARL interface ────────────────────────────────────────────────────────

    def reset(self):
        self.As, self.assoc = make_snapshot(self.cfg, self.rng)
        self._step = 0
        P0 = np.full((self.cfg.N_BS, self.cfg.N_RB), self.Pmax * 0.5)
        m0 = metrics(P0, self.As, self.assoc, self.cfg, self.nw)
        self._kpm = self._build_kpm(m0, P0)
        return self._kpm.copy()       # [N_BS, KPM_DIM]

    def step(self, actions):
        """
        actions: np.ndarray [N_BS, N_RB] in [0, 1]; scaled to [0, Pmax] internally.
        returns: kpm [N_BS, KPM_DIM], rewards [N_BS], done bool, info dict
        """
        P   = np.clip(actions, 0.0, 1.0) * self.Pmax
        m   = metrics(P, self.As, self.assoc, self.cfg, self.nw)
        kpm = self._build_kpm(m, P)
        self._kpm  = kpm
        self._step += 1
        done = self._step >= self.elen
        return kpm, self._per_agent_reward(m), done, m

    def get_global_kpm(self):
        """Cached KPM from last step: [N_BS, KPM_DIM]."""
        return self._kpm.copy()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_kpm(self, m, P):
        """Map metrics output → normalised R2 KPM obs [N_BS, 3]."""
        kpm = np.empty((self.cfg.N_BS, self.KPM_DIM), dtype=np.float32)
        for i in range(self.cfg.N_BS):
            tput_n  = float(np.clip(m['per_bs_rate'][i] / self._norm, 0, 1)) * 2 - 1
            prb_n   = float(np.clip((P[i] / self.Pmax).mean(),         0, 1)) * 2 - 1
            n_ue_i  = float((self.assoc == i).sum())
            n_ue_n  = float(np.clip(n_ue_i / self.cfg.N_UE,            0, 1)) * 2 - 1
            kpm[i]  = [tput_n, prb_n, n_ue_n]
        return kpm

    def _per_agent_reward(self, m):
        """
        Per-agent shaped reward consistent with metrics():
          r_i = (rate_i - lam * log2(1+intf_i) + 0.1*jain/N) / norm
        """
        intf  = m['intf_caused']
        jain  = m['jain']
        r = (m['per_bs_rate']
             - self.cfg.lam * np.log2(1.0 + np.maximum(intf, 0.0))
             + 0.1 * jain / self.cfg.N_BS) / self._norm
        return r.astype(np.float32)
