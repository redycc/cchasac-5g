"""
CCEnvR1Partial: R1-partial observation environment for Context-Conditioned HASAC.

Motivation (vs R2 KPM-only cc_env.py):
  R2 KPM (tput, prb_util, n_ue = 3-dim) is too indirect — a worker cannot tell
  "low SINR from a bad channel" apart from "low SINR from cross-cell
  interference". R1-partial gives each agent its OWN per-RB SINR (so it can see
  its own channel quality) but withholds the neighbour-interference term — that
  missing cross-cell coupling is exactly the coordination gap that context z is
  meant to fill.

Observation (R1-partial, per agent i):
  [sinr_norm_rb0..rb3, load_norm, tput_norm, n_ue_norm]  =  7-dim  (N_RB=4 + 3)
  NO intf_norm.

Channel: MUST use Cfg(freq_selective=True) so per-RB SINRs differ; otherwise the
  4 SINR values collapse to one number and the 7-dim obs degenerates.

Reward: from baseline.metrics() — the single source of truth (identical to
  cc_env.py; we do NOT touch the objective).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from baseline import (Cfg, make_snapshot, metrics, sinr_rb,
                      dbm_to_w, noise_w_per_rb)

_log2_1e4 = np.log2(1.0 + 1e4)


class CCEnvR1Partial:
    OBS_DIM = 4 + 3   # N_RB sinr + load + tput + n_ue = 7
    KPM_DIM = OBS_DIM  # alias for scripts that reference KPM_DIM

    def __init__(self, cfg=None, episode_length=200, seed=42):
        # default to freq_selective=True so per-RB SINRs differ
        self.cfg   = cfg if cfg is not None else Cfg(freq_selective=True)
        self.elen  = episode_length
        self.rng   = np.random.default_rng(seed)
        self.nw    = noise_w_per_rb()
        self.Pmax  = dbm_to_w(self.cfg.Pmax_dBm)
        self._norm = self.cfg.N_RB * _log2_1e4

        self.As        = None
        self.assoc     = None
        self._kpm      = np.zeros((self.cfg.N_BS, self.OBS_DIM), dtype=np.float32)
        self._step     = 0

    # ── MARL interface ────────────────────────────────────────────────────────

    def reset(self):
        self.As, self.assoc = make_snapshot(self.cfg, self.rng)
        self._step = 0
        P0 = np.full((self.cfg.N_BS, self.cfg.N_RB), self.Pmax * 0.5)
        m0 = metrics(P0, self.As, self.assoc, self.cfg, self.nw)
        self._kpm = self._build_obs(m0, P0)
        return self._kpm.copy()       # [N_BS, OBS_DIM]

    def step(self, actions):
        """
        actions: np.ndarray [N_BS, N_RB] in [0, 1]; scaled to [0, Pmax] internally.
        returns: obs [N_BS, OBS_DIM], rewards [N_BS], done bool, info dict (metrics)
        """
        P   = np.clip(actions, 0.0, 1.0) * self.Pmax
        m   = metrics(P, self.As, self.assoc, self.cfg, self.nw)
        obs = self._build_obs(m, P)
        self._kpm  = obs
        self._step += 1
        done = self._step >= self.elen
        return obs, self._per_agent_reward(m), done, m

    def get_global_kpm(self):
        """Cached obs from last step: [N_BS, OBS_DIM]."""
        return self._kpm.copy()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_obs(self, m, P):
        """Map metrics output + per-RB SINR → R1-partial obs [N_BS, 7]."""
        # per-RB SINR for every BS: [N_BS, N_RB]
        sinr_all = np.stack(
            [sinr_rb(P[:, rb], self.As[rb], self.nw) for rb in range(self.cfg.N_RB)],
            axis=1,
        )
        sinr_norm = np.clip(np.log2(1.0 + sinr_all) / 10.0, -1, 1)  # [N_BS, N_RB]

        obs = np.empty((self.cfg.N_BS, self.OBS_DIM), dtype=np.float32)
        for i in range(self.cfg.N_BS):
            tput_n = float(np.clip(m['per_bs_rate'][i] / self._norm, 0, 1)) * 2 - 1
            load_n = float(np.clip((P[i] / self.Pmax).mean(),        0, 1)) * 2 - 1
            n_ue_i = float((self.assoc == i).sum())
            n_ue_n = float(np.clip(n_ue_i / self.cfg.N_UE,           0, 1)) * 2 - 1
            obs[i] = np.concatenate([sinr_norm[i], [load_n, tput_n, n_ue_n]])
        return obs

    # alias so snapshot-pool reset helpers (which call _build_kpm) work unchanged
    def _build_kpm(self, m, P):
        return self._build_obs(m, P)

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
