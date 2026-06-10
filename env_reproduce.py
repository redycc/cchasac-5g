"""
env_reproduce.py — faithful implementation of tasks/files/REPRODUCE.md (§2, §4, §5).

Single source of truth: channel → SINR → rate → delivered goodput → reward.
- N_BS=4 cells, N_UE=8 users, area 1200 m, fc=3.5 GHz, Pmax=−15 dBm, shadow σ=4 dB
- per-cell scalar power level, equal split among backlogged UEs (intra-cell orthogonal)
- traffic queues (q_max=80, arrivals 8/slot to hot cells), PF weights w=1/(R̄+ε), β=0.05
- rewards: team (r_i = G = Σ w·delivered, canonical HASAC) or difference (counterfactual mute)
- oracles: full-CSI grid search (ceiling), spatial (large-scale-gain) grid search (gate target)
- baselines: equal (floor), round-robin
- regimes: fixed_topology (topo_seed, canonical comparison) or random topology
Units: mW.
"""
import numpy as np


def dbm_to_mw(dbm):
    return 10.0 ** (dbm / 10.0)


class Cfg:
    N_BS = 4
    N_UE = 8
    area = 1200.0
    fc_ghz = 3.5
    Pmax_dBm = -15.0
    sigma_shadow = 4.0
    N0 = 1e-12            # mW; N0 sweep matched spec §8 exactly: floor 5.27≈5.3, oracle 8.95≈9.1, RR<=floor
    q_max = 80.0
    arr_bits = 8.0
    pf_beta = 0.05
    pf_eps = 1e-2
    fading = True         # Gauss-Markov small-scale, rho=0.9
    rho_fading = 0.9
    hot_init = 1.0        # full-buffer default: always hot
    p_on_off = 0.0
    p_off_on = 1.0
    walk_std = 0.0        # optional UE mobility (m/slot)
    grid_K = 5            # oracle power levels {0, .25, .5, .75, 1}
    fixed_topology = True
    topo_seed = 12345
    geom_file = None      # path to authors' geom npz (g_ls/serv/bs/ue/N0/Pmax) for exact match
    ep_len = 20


# --------------------------- scenario ------------------------------
def _pathloss_db(d, fc_ghz, shadow):
    return 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc_ghz) + shadow


def gen_scenario(cfg, rng):
    """BS/UE placement → g_ls [N_BS,N_UE], serv [N_UE], plus positions/distances."""
    bs = rng.uniform(0, cfg.area, size=(cfg.N_BS, 2))
    ue = rng.uniform(0, cfg.area, size=(cfg.N_UE, 2))
    d = np.maximum(np.linalg.norm(bs[:, None, :] - ue[None, :, :], axis=-1), 1.0)
    shadow = rng.normal(0, cfg.sigma_shadow, size=d.shape)   # one draw per link, fixed for trial
    g_ls = 10.0 ** (-_pathloss_db(d, cfg.fc_ghz, shadow) / 10.0)
    serv = g_ls.argmax(axis=0)
    # guarantee >=1 UE per cell: donor must keep >=1 UE (never empty a cell)
    for _ in range(cfg.N_UE):
        counts = np.bincount(serv, minlength=cfg.N_BS)
        empties = np.where(counts == 0)[0]
        if len(empties) == 0:
            break
        j = empties[0]
        donors = np.where(counts > 1)[0]
        cand = [u for u in range(cfg.N_UE) if serv[u] in donors]
        u_best = max(cand, key=lambda u: g_ls[j, u])
        serv[u_best] = j
    return bs, ue, d, g_ls, serv


# --------------------------- physics -------------------------------
def rates_from_levels(levels, g, serv, Q, cfg, pmax):
    """levels [N_BS] in [0,1] → per-UE power (equal split among backlogged UEs) → rate [N_UE].
    Intra-cell orthogonal: interference is inter-cell only."""
    n_bs, n_ue = cfg.N_BS, cfg.N_UE
    P_bs = levels * pmax                                   # [N_BS]
    p_ue = np.zeros(n_ue)
    for j in range(n_bs):
        mine = np.where(serv == j)[0]
        backlogged = mine[Q[mine] > 0]
        tgt = backlogged if len(backlogged) > 0 else mine
        if len(tgt) > 0:
            p_ue[tgt] = P_bs[j] / len(tgt)
    P_bs_eff = np.array([p_ue[serv == j].sum() for j in range(n_bs)])
    total_rx = P_bs_eff @ g                                # [N_UE]
    sig = p_ue * g[serv, np.arange(n_ue)]
    intf = total_rx - P_bs_eff[serv] * g[serv, np.arange(n_ue)]
    sinr = sig / (intf + cfg.N0)
    return np.log2(1.0 + sinr), p_ue


def delivered_goodput(levels, g, serv, Q, cfg, pmax):
    rate, p_ue = rates_from_levels(levels, g, serv, Q, cfg, pmax)
    return np.minimum(Q, rate), rate, p_ue


# --------------------------- oracles / baselines -------------------
def oracle_levels(g, serv, Q, cfg, pmax):
    """Full grid search over per-cell levels maximizing per-slot Σ min(Q, rate)."""
    K = cfg.grid_K
    lv = np.linspace(0.0, 1.0, K)
    best, best_levels = -1.0, np.ones(cfg.N_BS)
    for idx in np.ndindex(*([K] * cfg.N_BS)):
        levels = lv[list(idx)]
        d, _, _ = delivered_goodput(levels, g, serv, Q, cfg, pmax)
        s = d.sum()
        if s > best:
            best, best_levels = s, levels.copy()
    return best_levels


def bl_equal(cfg):
    return np.ones(cfg.N_BS)


def bl_round_robin(t, cfg):
    levels = np.zeros(cfg.N_BS)
    levels[t % cfg.N_BS] = 1.0
    return levels


# --------------------------- env -----------------------------------
class Env:
    def __init__(self, cfg, reward_mode="team", seed=0):
        assert reward_mode in ("team", "difference")
        self.cfg, self.reward_mode = cfg, reward_mode
        self.rng = np.random.default_rng(seed)
        self.pmax = dbm_to_mw(cfg.Pmax_dBm)
        self._topo = None
        if getattr(cfg, "geom_file", None):
            # exact-geometry mode: load g_ls/serv/bs/ue + constants from the authors' dump,
            # skipping our own RNG draws entirely (draw-order differences made same-seed
            # topologies diverge between implementations)
            d = np.load(cfg.geom_file)
            bs, ue = d["bs"].astype(float), d["ue"].astype(float)
            g_ls = d["g_ls"].astype(float)
            serv = d["serv"].astype(int)
            dist = np.maximum(np.linalg.norm(bs[:, None, :] - ue[None, :, :], axis=-1), 1.0)
            self._topo = (bs, ue, dist, g_ls, serv)
            if "N0" in d:
                cfg.N0 = float(d["N0"]) * 1e3        # file stores Watts; env uses mW
            if "Pmax" in d:
                self.pmax = float(d["Pmax"]) * 1e3   # W -> mW
        elif cfg.fixed_topology:
            topo_rng = np.random.default_rng(cfg.topo_seed)
            self._topo = gen_scenario(cfg, topo_rng)
        self.t = 0

    def reset(self):
        cfg = self.cfg
        if self._topo is not None:
            self.bs, self.ue, self.d, self.g_ls, self.serv = (x.copy() for x in self._topo)
        else:
            self.bs, self.ue, self.d, self.g_ls, self.serv = gen_scenario(cfg, self.rng)
        self.g_ls0, self.d0 = self.g_ls.copy(), self.d.copy()
        self.h = (self.rng.normal(size=self.g_ls.shape) +
                  1j * self.rng.normal(size=self.g_ls.shape)) / np.sqrt(2.0)
        self.Q = np.zeros(cfg.N_UE)
        self.hot = (self.rng.random(cfg.N_BS) < cfg.hot_init).astype(float)
        self._arrivals()                                       # initial backlog
        self.Rbar = np.zeros(cfg.N_UE)
        self.levels_prev = np.ones(cfg.N_BS)                   # start at equal power
        rate, p_ue = rates_from_levels(self.levels_prev, self.gain(), self.serv,
                                       self.Q, cfg, self.pmax)
        self.rate_prev, self.p_ue_prev = rate, p_ue
        self.t = 0
        return self._obs()

    def gain(self):
        return self.g_ls * np.abs(self.h) ** 2 if self.cfg.fading else self.g_ls

    def _arrivals(self):
        cfg = self.cfg
        for j in range(cfg.N_BS):
            if self.hot[j] > 0:
                mine = self.serv == j
                self.Q[mine] = np.minimum(self.Q[mine] + cfg.arr_bits, cfg.q_max)

    def _weights(self):
        return 1.0 / (self.Rbar + self.cfg.pf_eps)

    def step(self, levels):
        """levels [N_BS] in [0,1]. Returns obs, r [N_BS], done(False), info."""
        cfg = self.cfg
        levels = np.clip(np.asarray(levels, dtype=float), 0.0, 1.0)
        g = self.gain()
        w = self._weights()
        delivered, rate, p_ue = delivered_goodput(levels, g, self.serv, self.Q, cfg, self.pmax)
        G = float((w * delivered).sum())

        if self.reward_mode == "team":
            r = np.full(cfg.N_BS, G)
        else:                                                  # difference: r_i = G − G(cell i muted)
            r = np.zeros(cfg.N_BS)
            for i in range(cfg.N_BS):
                lv_mut = levels.copy(); lv_mut[i] = 0.0
                d_mut, _, _ = delivered_goodput(lv_mut, g, self.serv, self.Q, cfg, self.pmax)
                r[i] = G - float((w * d_mut).sum())

        # state updates
        self.Rbar = (1 - cfg.pf_beta) * self.Rbar + cfg.pf_beta * delivered
        self.Q = np.minimum(self.Q - delivered, cfg.q_max)
        self._toggle_hot()
        self._arrivals()
        if cfg.fading:
            n = (self.rng.normal(size=self.h.shape) +
                 1j * self.rng.normal(size=self.h.shape)) / np.sqrt(2.0)
            self.h = cfg.rho_fading * self.h + np.sqrt(1 - cfg.rho_fading ** 2) * n
        if cfg.walk_std > 0:
            self.ue = np.clip(self.ue + self.rng.normal(0, cfg.walk_std, self.ue.shape),
                              0, cfg.area)
            d_new = np.maximum(np.linalg.norm(self.bs[:, None, :] - self.ue[None, :, :],
                                              axis=-1), 1.0)
            self.g_ls = self.g_ls0 * (self.d0 / d_new) ** 2.1
            self.d = d_new
        self.levels_prev, self.rate_prev, self.p_ue_prev = levels, rate, p_ue
        self.t += 1
        info = dict(delivered=delivered, goodput=float(delivered.sum()), rate=rate, G=G)
        return self._obs(), r, False, info

    def _toggle_hot(self):
        cfg = self.cfg
        for j in range(cfg.N_BS):
            if self.hot[j] > 0 and self.rng.random() < cfg.p_on_off:
                self.hot[j] = 0.0
            elif self.hot[j] == 0 and self.rng.random() < cfg.p_off_on:
                self.hot[j] = 1.0

    # ---- observations (three-tier) ----
    def obs_local(self):
        """(A) per-UE deployable features [N_UE, 4]: [rate, log(w), prev_p/Pmax, log(1+Q)]."""
        w = self._weights()
        return np.stack([self.rate_prev, np.log(w),
                         self.p_ue_prev / self.pmax, np.log1p(self.Q)], axis=1).astype(np.float32)

    def obs_kpm(self):
        """(B) per-cell KPM [N_BS, 3]: [load(#UE), mean R̄, mean Q]. No CSI."""
        cfg = self.cfg
        kpm = np.zeros((cfg.N_BS, 3), np.float32)
        for j in range(cfg.N_BS):
            mine = self.serv == j
            kpm[j] = [mine.sum(), self.Rbar[mine].mean(), self.Q[mine].mean()]
        return kpm

    def obs_share(self):
        """(C) privileged share_obs: [log-gain, p_ue/Pmax, serv-onehot, w, rate, Q]."""
        cfg = self.cfg
        g = self.gain()
        g_log = (np.log10(g.flatten()) + 14.0) / 6.0           # roughly in [0,1]
        onehot = np.zeros((cfg.N_UE, cfg.N_BS), np.float32)
        onehot[np.arange(cfg.N_UE), self.serv] = 1.0
        return np.concatenate([g_log, self.p_ue_prev / self.pmax, onehot.flatten(),
                               self._weights(), self.rate_prev, self.Q]).astype(np.float32)

    def _obs(self):
        return dict(local=self.obs_local(), kpm=self.obs_kpm(), share=self.obs_share())

    # ---- oracle helpers (need current internal state) ----
    def oracle_action(self):
        return oracle_levels(self.gain(), self.serv, self.Q, self.cfg, self.pmax)

    def spatial_oracle_action(self):
        return oracle_levels(self.g_ls, self.serv, self.Q, self.cfg, self.pmax)


SHARE_DIM = lambda cfg: cfg.N_BS * cfg.N_UE + cfg.N_UE + cfg.N_UE * cfg.N_BS + 3 * cfg.N_UE


# --------------------------- sanity --------------------------------
if __name__ == "__main__":
    cfg = Cfg()
    pmax = dbm_to_mw(cfg.Pmax_dBm)

    def run_policy(policy, n_trials=5, T=150, seed0=20000):
        means = []
        for k in range(n_trials):
            env = Env(cfg, seed=seed0 + k)
            env.reset()
            gp = []
            for t in range(T):
                _, _, _, info = env.step(policy(env, t))
                gp.append(info["goodput"])
            means.append(np.mean(gp[T // 3:]))                 # steady-state
        return float(np.mean(means)), float(np.std(means))

    eq, eqs = run_policy(lambda e, t: bl_equal(cfg))
    rr, rrs = run_policy(lambda e, t: bl_round_robin(t, cfg))
    orc, orcs = run_policy(lambda e, t: e.oracle_action(), n_trials=3)
    print(f"equal (floor)  : {eq:.3f} ± {eqs:.3f}")
    print(f"round-robin    : {rr:.3f} ± {rrs:.3f}")
    print(f"oracle (ceiling): {orc:.3f} ± {orcs:.3f}")

    # interference-limited check at equal power
    env = Env(cfg, seed=1); env.reset()
    g = env.gain()
    _, p_ue = rates_from_levels(bl_equal(cfg), g, env.serv, env.Q, cfg, pmax)
    P_bs = np.array([p_ue[env.serv == j].sum() for j in range(cfg.N_BS)])
    total = P_bs @ g
    sig = p_ue * g[env.serv, np.arange(cfg.N_UE)]
    intf = total - P_bs[env.serv] * g[env.serv, np.arange(cfg.N_UE)]
    print(f"median interference/N0 = {np.median(intf) / cfg.N0:.1f}  (>>1 → interference-limited)")
    print(f"sanity: harm check — oracle >= equal: {orc >= eq}")
