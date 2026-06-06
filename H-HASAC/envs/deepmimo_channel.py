"""
DeepMIMO channel model for FiveGEnv.

Pre-loads O1 3.5GHz scenario channel gains (incoherent path power sum)
for multiple BSes and caches to .npy for fast reload.

Usage:
    ch = DeepMIMOChannel(n_bs=3, cache_dir='./deepmimo_cache')
    ue_idx, ue_pos = ch.sample_ues(n_ue=10, rng=np.random.default_rng(0))
    gains = ch.get_gains(ue_idx)   # shape (n_bs, n_ue), linear channel gain
"""

import os
import numpy as np

SCENARIO = 'o1_3p5'
TX_IDS = [3, 4, 5]      # 3 BSes from O1
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), '..', 'deepmimo_cache')
TX_POWER_DBM = 30.0      # assumed BS TX power (dBm), used to scale gains


class DeepMIMOChannel:
    def __init__(
        self,
        n_bs: int = 3,
        scenario: str = SCENARIO,
        tx_ids: list = None,
        cache_dir: str = DEFAULT_CACHE,
        ue_pool_size: int = 10_000,
    ):
        self.n_bs = n_bs
        self.scenario = scenario
        self.tx_ids = (tx_ids or TX_IDS)[:n_bs]
        self.cache_dir = os.path.abspath(cache_dir)
        self.ue_pool_size = ue_pool_size

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_or_build_cache()

    # ── Public API ──────────────────────────────────────────────────────────

    def sample_ues(self, n_ue: int, rng: np.random.Generator):
        """Sample n_ue positions from the UE pool.
        Returns:
            ue_idx:  (n_ue,) indices into pool
            ue_pos:  (n_ue, 2) xy positions in metres
        """
        idx = rng.choice(self.pool_size, n_ue, replace=False)
        return idx, self.rx_pos[idx, :2]

    def get_gains(self, ue_idx: np.ndarray) -> np.ndarray:
        """Linear channel gain for each BS-UE pair.
        Args:
            ue_idx: (n_ue,) indices into pool
        Returns:
            gains: (n_bs, n_ue) linear channel gain (unitless, not normalised by TX power)
        """
        return self.gains[:, ue_idx]   # (n_bs, n_ue)

    @property
    def bs_pos(self) -> np.ndarray:
        """BS positions in metres, shape (n_bs, 2)."""
        return self._bs_pos

    @property
    def pool_size(self) -> int:
        return len(self.rx_pos)

    # ── Cache management ────────────────────────────────────────────────────

    def _cache_paths(self):
        tag = f"{self.scenario}_tx{'_'.join(str(t) for t in self.tx_ids)}_pool{self.ue_pool_size}"
        gains_path = os.path.join(self.cache_dir, f"{tag}_gains.npy")
        rxpos_path = os.path.join(self.cache_dir, f"{tag}_rxpos.npy")
        txpos_path = os.path.join(self.cache_dir, f"{tag}_txpos.npy")
        return gains_path, rxpos_path, txpos_path

    def _load_or_build_cache(self):
        gains_path, rxpos_path, txpos_path = self._cache_paths()

        if all(os.path.exists(p) for p in [gains_path, rxpos_path, txpos_path]):
            print(f"[DeepMIMOChannel] Loading cached gains from {self.cache_dir}")
            self.gains = np.load(gains_path)       # (n_bs, n_ue_pool)
            self.rx_pos = np.load(rxpos_path)       # (n_ue_pool, 3)
            self._bs_pos = np.load(txpos_path)      # (n_bs, 3)
            print(f"[DeepMIMOChannel] Loaded: {self.gains.shape} gains, {len(self.rx_pos)} UE positions")
            return

        print(f"[DeepMIMOChannel] Building cache (first run, may take ~30s)...")
        import deepmimo as dm

        dataset = dm.generate(
            self.scenario,
            load_params={'tx_sets': self.tx_ids, 'rx_sets': [0]},
        )

        n_positions = len(dataset[0].rx_pos)
        # Uniformly subsample UE pool to ue_pool_size
        if self.ue_pool_size < n_positions:
            pool_idx = np.linspace(0, n_positions - 1, self.ue_pool_size, dtype=int)
        else:
            pool_idx = np.arange(n_positions)

        self.rx_pos = dataset[0].rx_pos[pool_idx]          # (pool, 3)
        self._bs_pos = np.array([
            dataset[bs].tx_pos[0] for bs in range(self.n_bs)
        ])                                                   # (n_bs, 3)

        # Incoherent power sum: linear gain = sum of |path powers|
        # dataset[bs].power: (n_positions, n_paths) in dBW at 0 dBm TX
        gains = np.zeros((self.n_bs, len(pool_idx)), dtype=np.float32)
        for bs in range(self.n_bs):
            power_dbw = dataset[bs].power[pool_idx]        # (pool, n_paths)
            # Replace -inf (no path / masked) with very large negative dB
            power_dbw = np.where(np.isfinite(power_dbw), power_dbw, -300.0)
            power_w = 10 ** (power_dbw / 10)               # linear watts
            gains[bs] = power_w.sum(axis=1)                # sum over paths

        self.gains = gains

        # Save cache
        np.save(gains_path, self.gains)
        np.save(rxpos_path, self.rx_pos)
        np.save(txpos_path, self._bs_pos)
        print(f"[DeepMIMOChannel] Cache saved. Gains shape: {self.gains.shape}")
