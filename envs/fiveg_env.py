"""
Simple 5G multi-cell environment for validating Hierarchical HASAC.

Scenario:
  - N_BS small-cell base stations (workers/agents)
  - N_UE user equipments randomly placed in a square area
  - N_RB resource blocks per BS
  - Centralized manager can inject per-BS power budgets (sub-goals)

Action: each BS outputs power fraction per RB  ∈ [0, 1]^N_RB
        actual Tx power = action * P_max_i (clipped by sub-goal in hierarchical mode)

Observation per BS i:
  [sinr_per_rb (N_RB), load (1), inter-cell-interference (1),
   n_connected_ues (1), throughput_last (1),
   sub_goal: (p_max_frac, i_thresh_frac, rb_share) (3)]  <- zeros in flat mode
  total = N_RB + 4 + 3

Share obs: concatenation of all per-BS observations  (N_BS * obs_dim,)

Reward per BS: Shannon throughput - lambda * interference caused
"""

import numpy as np
from gymnasium import spaces


class FiveGEnv:
    # Physical constants
    NOISE_DBM = -174 + 10 * np.log10(180e3)  # thermal noise per RB (180 kHz), ~-121 dBm
    NOISE_W = 10 ** ((NOISE_DBM - 30) / 10)

    def __init__(self, args: dict):
        self.n_bs = int(args.get("n_bs", 3))           # number of small-cell BSes
        self.n_ue = int(args.get("n_ue", 10))          # number of UEs
        self.n_rb = int(args.get("n_rb", 4))           # resource blocks per BS
        self.area = float(args.get("area", 500.0))     # square area side length (m)
        self.p_max_dbm = float(args.get("p_max_dbm", 30.0))  # max BS TX power (dBm)
        self.p_max_w = 10 ** ((self.p_max_dbm - 30) / 10)
        self.lambda_intf = float(args.get("lambda_intf", 0.1))  # interference penalty
        self.episode_length = int(args.get("episode_length", 200))
        self.hierarchical = bool(args.get("hierarchical", False))

        # DeepMIMO channel source (optional)
        self.channel_source = args.get("channel_source", "formula")  # "formula" or "deepmimo"
        self._deepmimo = None
        if self.channel_source == "deepmimo":
            from envs.deepmimo_channel import DeepMIMOChannel
            self._deepmimo = DeepMIMOChannel(
                n_bs=self.n_bs,
                ue_pool_size=int(args.get("ue_pool_size", 10_000)),
                cache_dir=args.get("deepmimo_cache", "./deepmimo_cache"),
            )
            # Override BS positions from DeepMIMO
            self._dm_bs_pos = self._deepmimo.bs_pos[:, :2]  # (n_bs, 2)
        self._dm_ue_idx = None  # current UE indices in deepmimo pool

        # obs_mode: "full" includes sub_goal (3-dim); "r2" is local KPM only
        self.obs_mode = args.get("obs_mode", "full")
        kpm_dim = self.n_rb + 4
        self.obs_dim = kpm_dim if self.obs_mode == "r2" else kpm_dim + 3
        self.n_agents = self.n_bs
        self._last_global_kpm = np.zeros((self.n_bs, kpm_dim), dtype=np.float32)

        # HARL-required spaces (lists of length n_agents)
        agent_obs = spaces.Box(-1.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        share_obs = spaces.Box(-1.0, 1.0, shape=(self.n_bs * self.obs_dim,), dtype=np.float32)
        agent_act = spaces.Box(0.0, 1.0, shape=(self.n_rb,), dtype=np.float32)

        self.observation_space = [agent_obs] * self.n_agents
        self.share_observation_space = [share_obs] * self.n_agents
        self.action_space = [agent_act] * self.n_agents

        self._seed = 0
        self._step = 0
        self._rng = np.random.default_rng(self._seed)

        # positions (set in reset)
        self.bs_pos = None   # (n_bs, 2)
        self.ue_pos = None   # (n_ue, 2)
        self.path_loss = None  # (n_bs, n_ue) linear gain
        self.power_w = None    # (n_bs, n_rb)  current TX power
        self.sub_goals = np.zeros((self.n_bs, 3), dtype=np.float32)  # manager sub-goals

    # ------------------------------------------------------------------
    # HARL interface
    # ------------------------------------------------------------------

    def seed(self, seed: int):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self):
        self._step = 0
        self.sub_goals = np.zeros((self.n_bs, 3), dtype=np.float32)

        if self.channel_source == "deepmimo":
            # BS positions fixed from DeepMIMO scenario
            self.bs_pos = self._dm_bs_pos.copy()
            # Sample UE positions from DeepMIMO pool
            self._dm_ue_idx, self.ue_pos = self._deepmimo.sample_ues(self.n_ue, self._rng)
        else:
            # Place BSes on a grid + small jitter
            n_side = int(np.ceil(np.sqrt(self.n_bs)))
            grid = np.array([[i, j] for i in range(n_side) for j in range(n_side)], dtype=float)
            grid = grid[:self.n_bs] * (self.area / (n_side + 1)) + self.area / (n_side + 1)
            grid += self._rng.uniform(-20, 20, grid.shape)
            self.bs_pos = np.clip(grid, 10, self.area - 10)
            # Place UEs uniformly
            self.ue_pos = self._rng.uniform(0, self.area, (self.n_ue, 2))

        self._update_path_loss()
        self.power_w = np.ones((self.n_bs, self.n_rb)) * self.p_max_w * 0.1

        obs, share_obs = self._get_obs()
        return obs, share_obs, None  # avail_actions = None (continuous)

    def step(self, actions):
        """
        actions: list of n_bs arrays, each shape (n_rb,), values in [0,1]
        """
        self._step += 1

        # Set powers (action * p_max, clipped by sub-goal if hierarchical)
        actions = np.array(actions)   # (n_bs, n_rb)
        p_max_frac = np.ones(self.n_bs)
        if self.hierarchical:
            p_max_frac = np.clip(self.sub_goals[:, 0], 0.05, 1.0)  # P_max fraction

        for i in range(self.n_bs):
            self.power_w[i] = np.clip(actions[i], 0, 1) * self.p_max_w * p_max_frac[i]

        # Move UEs (random walk for formula mode; resample from pool for deepmimo)
        if self.channel_source == "deepmimo":
            # Resample a subset of UEs to simulate mobility
            n_mobile = max(1, self.n_ue // 5)
            mobile_idx = self._rng.choice(self.n_ue, n_mobile, replace=False)
            new_pool_idx, new_pos = self._deepmimo.sample_ues(n_mobile, self._rng)
            self._dm_ue_idx[mobile_idx] = new_pool_idx
            self.ue_pos[mobile_idx] = new_pos
        else:
            self.ue_pos += self._rng.normal(0, 1.0, self.ue_pos.shape)
            self.ue_pos = np.clip(self.ue_pos, 0, self.area)
        self._update_path_loss()

        rewards, info = self._compute_rewards()
        obs, share_obs = self._get_obs()

        done = self._step >= self.episode_length
        dones = [done] * self.n_bs
        infos = [info] * self.n_bs
        if done:
            for d in infos:
                d["bad_transition"] = True

        return obs, share_obs, [[r] for r in rewards], dones, infos, None

    def get_global_kpm(self) -> np.ndarray:
        """Return cached per-BS local KPM (R2 regime) from last step: [n_bs, n_rb+4]."""
        return self._last_global_kpm.copy()

    def set_sub_goals(self, sub_goals: np.ndarray):
        """Called by hierarchical wrapper to inject manager sub-goals.
        sub_goals: (n_bs, 3)  [p_max_frac, i_thresh_frac, rb_share_frac]
        """
        self.sub_goals = np.clip(np.nan_to_num(sub_goals, nan=0.5), 0, 1).astype(np.float32)

    def close(self):
        pass

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_path_loss(self):
        """Linear channel gain (n_bs, n_ue). Uses DeepMIMO or formula."""
        if self.channel_source == "deepmimo":
            # Ray-tracing channel gains from pre-loaded DeepMIMO data
            # gains shape: (n_bs, n_ue), already linear, normalized to 0-dBm TX
            self.path_loss = self._deepmimo.get_gains(self._dm_ue_idx).astype(np.float64)
            # Scale from 0-dBm reference to actual p_max
            tx_p_w = self.p_max_w
            # DeepMIMO power is referenced to 0 dBm = 1e-3 W; rescale
            self.path_loss = self.path_loss / 1e-3  # now gain is unitless (W/W at p_max)
        else:
            # 3GPP UMi path loss formula
            diff = self.bs_pos[:, None, :] - self.ue_pos[None, :, :]  # (n_bs, n_ue, 2)
            dist_m = np.linalg.norm(diff, axis=-1)
            dist_m = np.clip(dist_m, 1.0, None)
            pl_db = 32.4 + 21 * np.log10(dist_m) + 20 * np.log10(3.5)
            pl_db += self._rng.normal(0, 4.0, pl_db.shape)
            self.path_loss = 10 ** (-pl_db / 10)  # (n_bs, n_ue)

    def _compute_sinr(self):
        """SINR per BS per RB (averaged over UEs served by that BS).
        Returns sinr: (n_bs, n_rb)
        """
        # Each UE associates with the BS with highest path loss gain
        assoc = np.argmax(self.path_loss, axis=0)  # (n_ue,) -> which BS each UE belongs to

        sinr_per_bs_rb = np.zeros((self.n_bs, self.n_rb))
        intf_caused = np.zeros(self.n_bs)  # total interference caused to others

        for i in range(self.n_bs):
            ue_idx = np.where(assoc == i)[0]
            if len(ue_idx) == 0:
                continue
            for rb in range(self.n_rb):
                signal = self.power_w[i, rb] * self.path_loss[i, ue_idx].mean()
                # Interference from other BSes on this RB
                intf = sum(
                    self.power_w[j, rb] * self.path_loss[j, ue_idx].mean()
                    for j in range(self.n_bs) if j != i
                )
                intf_caused[i] += intf  # interference I cause (others receive from me)
                sinr_per_bs_rb[i, rb] = signal / (intf + self.NOISE_W + 1e-20)

        # interference caused by BS i = sum over other BSes j of (p_i * gain to j's UEs)
        for i in range(self.n_bs):
            intf_caused[i] = 0.0
            for j in range(self.n_bs):
                if j == i:
                    continue
                ue_j = np.where(assoc == j)[0]
                if len(ue_j) == 0:
                    continue
                for rb in range(self.n_rb):
                    intf_caused[i] += self.power_w[i, rb] * self.path_loss[i, ue_j].mean()

        return sinr_per_bs_rb, intf_caused, assoc

    def _compute_rewards(self):
        sinr, intf_caused, assoc = self._compute_sinr()
        # Shannon throughput per RB (bps/Hz)
        tput = np.log2(1 + np.clip(sinr, 0, None))  # (n_bs, n_rb)
        tput_per_bs = tput.sum(axis=1)  # (n_bs,)

        # Jain fairness over BS throughputs
        n_ue_per_bs = np.array([(assoc == i).sum() for i in range(self.n_bs)], dtype=float)
        n_ue_per_bs = np.maximum(n_ue_per_bs, 1.0)
        per_ue_tput = tput_per_bs / n_ue_per_bs

        jain_num = per_ue_tput.sum() ** 2
        jain_den = self.n_bs * (per_ue_tput ** 2).sum() + 1e-20
        fairness = jain_num / jain_den

        rewards = tput_per_bs - self.lambda_intf * intf_caused + 0.1 * fairness
        # Normalize to ~[-1, 1] range
        rewards = rewards / (self.n_rb * np.log2(1 + 1e4) + 1e-8)

        info = {
            "throughput": float(tput_per_bs.sum()),
            "fairness": float(fairness),
            "interference": float(intf_caused.sum()),
        }
        return rewards.astype(np.float32), info

    def _get_obs(self):
        sinr, intf_caused, assoc = self._compute_sinr()
        n_ue_per_bs = np.array([(assoc == i).sum() for i in range(self.n_bs)], dtype=float)
        tput = np.log2(1 + np.clip(sinr, 0, None)).sum(axis=1)

        obs_list = []
        kpm_list = []
        for i in range(self.n_bs):
            # SINR per RB (log-scale, normalized to ~[-1,1])
            sinr_norm = np.clip(np.log2(1 + sinr[i]) / 10.0, -1, 1)
            load = np.clip(n_ue_per_bs[i] / self.n_ue, 0, 1) * 2 - 1
            intf_norm = np.clip(intf_caused[i] / (self.p_max_w * self.n_rb + 1e-20), 0, 1) * 2 - 1
            tput_norm = np.clip(tput[i] / (self.n_rb * 10), 0, 1) * 2 - 1
            n_ue_norm = np.clip(n_ue_per_bs[i] / self.n_ue, 0, 1) * 2 - 1
            sub_goal = self.sub_goals[i] * 2 - 1  # [0,1] -> [-1,1]

            kpm_i = np.array([*sinr_norm, load, intf_norm, tput_norm, n_ue_norm], dtype=np.float32)
            kpm_list.append(kpm_i)

            if self.obs_mode == "r2":
                obs_i = kpm_i
            else:
                obs_i = np.concatenate([kpm_i, sub_goal]).astype(np.float32)
            obs_list.append(obs_i)

        self._last_global_kpm = np.array(kpm_list, dtype=np.float32)
        share_obs_flat = np.concatenate(obs_list).astype(np.float32)
        share_obs_list = [share_obs_flat] * self.n_agents
        return obs_list, share_obs_list
