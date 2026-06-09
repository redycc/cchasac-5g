"""
C-HASAC 環境核心 + difference reward + full-CSI 天花板  (numpy only)

對應 HANDOFF_CHASAC_IMPL.md §3–§4–§7。
這是「唯一真相來源」: channel -> SINR -> rate -> reward。RL (HASAC/C-HASAC) 接 Env 即可。

模型:
  multi-cell, reuse-1。每個 UE 由最強 BS 服務 (自然產生 cell-edge UE)。
  intra-cell 正交 (同 BS 的 UE 互不干擾);inter-cell 干擾 = 其他 BS 的「總發射功率 × 增益」。
  動作 = per-UE 功率,per-BS sum-power 約束 Σ_{u∈i} p_u ≤ P_max。

提供:
  - metrics 核心: rates_from_power / pf utility / jain
  - reward: difference_reward (per-BS, 反事實) + team_reward
  - 觀測: 三層 (actor 本地 o_i / encoder kpm / critic share_obs)
  - baselines: equal_power (floor) / pf_wsr_ceiling (full-CSI 天花板)
  - Env class: reset() / step()
直接 `python env_chasac.py` 會跑 sanity check。
"""
import numpy as np
from itertools import product


# ----------------------------- config -----------------------------
class Cfg:
    N_BS = 3
    N_UE = 12            # 總 UE 數固定 (per-BS 數量由 association 決定, 會變動)
    area = 500.0
    fc_GHz = 3.5
    Pmax_dBm = 30.0
    shadow_dB = 4.0
    pf_beta = 0.05       # PF running-average rate 的 EMA 係數
    ceiling_grid = 6     # 天花板: 每 BS 總功率的 grid 等分數


def dbm_to_w(x):
    return 10.0 ** ((x - 30.0) / 10.0)


N0 = dbm_to_w(-174.0 + 10.0 * np.log10(180e3))   # 每 RB 熱雜訊
PMAX = None  # 由 cfg 設定, 見下


# --------------------------- channel ------------------------------
def gen_scenario(cfg, rng):
    """佈點 + 關聯。回傳 g[N_BS,N_UE], serv[N_UE](服務 BS index)。"""
    bs = rng.uniform(0, cfg.area, size=(cfg.N_BS, 2))
    ue = rng.uniform(0, cfg.area, size=(cfg.N_UE, 2))
    d = np.linalg.norm(bs[:, None, :] - ue[None, :, :], axis=-1)
    d = np.maximum(d, 1.0)
    PL = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(cfg.fc_GHz)
    PL = PL + rng.normal(0.0, cfg.shadow_dB, size=PL.shape)
    g = 10.0 ** (-PL / 10.0)                      # [N_BS, N_UE] power gain
    serv = np.argmax(g, axis=0)                   # best-signal association
    # 確保每個 BS 至少 1 個 UE (否則 bl_equal_power 除以零 / critic 維度問題)。
    # 注意: 不能單純對空 BS 借一個 UE —— 若從只有 1 個 UE 的 BS 借走會製造新的空 BS,
    # 且前向掃描不會回頭重檢。改用迴圈: 每次從「UE 數最多」的 BS (必 ≥2) 借出對空 BS
    # 增益最強的 UE, 直到所有 BS 非空 (空 BS 數量單調遞減, 保證終止)。
    counts = np.bincount(serv, minlength=cfg.N_BS)
    while np.any(counts == 0):
        j = int(np.argmin(counts))                # 某個空 BS
        donor = int(np.argmax(counts))            # UE 最多的 BS (此時必 ≥2)
        cand = np.where(serv == donor)[0]
        u = cand[int(np.argmax(g[j, cand]))]      # 借出對 j 增益最強的 UE
        serv[u] = j
        counts = np.bincount(serv, minlength=cfg.N_BS)
    return g, serv, bs, ue


# ----------------------- objective (唯一真相) ----------------------
def rates_from_power(p, g, serv, N_BS, isolated=False):
    """
    p:[N_UE] per-UE 功率; 回傳 rate[N_UE](bps/Hz), sinr[N_UE], P_bs[N_BS] 各 BS 總功率。
    intra-cell 正交; inter-cell 干擾 = 其他 BS 總功率 × 增益。
    isolated=True: 零化 inter-cell 干擾（等效 N_BS=1，curriculum pretrain 用）。
    """
    N_UE = len(p)
    P_bs = np.zeros(N_BS)
    for j in range(N_BS):
        P_bs[j] = p[serv == j].sum()
    g_serv = g[serv, np.arange(N_UE)]                 # 服務增益 g[serv[u], u]
    signal = p * g_serv
    if isolated:
        interference = np.zeros(N_UE)
    else:
        total_rx = (P_bs[:, None] * g).sum(axis=0)        # Σ_j P_bs[j] g[j,u]
        serv_contrib = P_bs[serv] * g_serv                # 服務 BS「總功率」貢獻 (要扣掉=inter-cell)
        interference = total_rx - serv_contrib
    sinr = signal / (interference + N0)
    rate = np.log2(1.0 + sinr)
    return rate, sinr, P_bs


def pf_utility(rate, w):
    return float((w * rate).sum())


def jain(per_bs_or_ue_rate):
    x = np.asarray(per_bs_or_ue_rate, dtype=float)
    s = x.sum()
    return float(s * s / (len(x) * (x * x).sum() + 1e-12)) if s > 0 else 0.0


# --------------------------- rewards ------------------------------
def difference_reward(p, g, serv, w, N_BS, isolated=False):
    """
    per-BS difference reward (偷自 Nasir-Guo, 只當訓練訊號, 用全資訊算):
        r_i = Σ_{u∈i} w_u rate_u  −  Σ_{u∉i} w_u ( rate_u^{i靜音} − rate_u^{實際} )
    回傳 r[N_BS], rate_act[N_UE]。
    """
    rate_act, _, _ = rates_from_power(p, g, serv, N_BS, isolated=isolated)
    C_act = w * rate_act
    r = np.zeros(N_BS)
    for i in range(N_BS):
        p_cf = p.copy()
        p_cf[serv == i] = 0.0
        rate_cf, _, _ = rates_from_power(p_cf, g, serv, N_BS, isolated=isolated)
        C_cf = w * rate_cf
        other = (serv != i)
        harm_i = float((C_cf[other] - C_act[other]).sum())     # ≥ 0
        own_i = float(C_act[serv == i].sum())
        r[i] = own_i - harm_i
    return r, rate_act


def team_reward(p, g, serv, w, N_BS, isolated=False):
    rate_act, _, _ = rates_from_power(p, g, serv, N_BS, isolated=isolated)
    return float((w * rate_act).sum()), rate_act


# --------------------------- observations -------------------------
def obs_local(p, g, serv, w, N_BS, isolated=False):
    """(A) actor 的 per-BS 本地觀測 (deployable)。回傳 list, 每個 BS 一個 [n_ue_i, F] 陣列。"""
    rate, sinr, _ = rates_from_power(p, g, serv, N_BS, isolated=isolated)
    out = []
    for i in range(N_BS):
        idx = np.where(serv == i)[0]
        # 每個 UE: [achievable_rate(=本地可量 CQI proxy), PF權重, 上一步功率]
        feats = np.stack([rate[idx], w[idx], p[idx]], axis=1)   # [n_ue_i, 3]
        out.append(feats)
    return out


def obs_kpm(p, g, serv, N_BS, bs=None, isolated=False):
    """(B) encoder 輸入: 各 cell 的 KPM [N_BS, 3+(N_BS-1)] = [load, throughput, P_bs, d_to_others...]。
    若提供 bs[N_BS,2]，額外附加各 BS 到其他 BS 的正規化距離，讓 encoder z 編碼空間鄰域關係。"""
    rate, _, P_bs = rates_from_power(p, g, serv, N_BS, isolated=isolated)
    kpm_base = np.zeros((N_BS, 3), dtype=np.float32)
    for i in range(N_BS):
        idx = np.where(serv == i)[0]
        kpm_base[i] = [len(idx), rate[idx].sum(), P_bs[i]]
    if bs is None:
        return kpm_base
    # per-BS distances to every other BS, ordered by j index (j != i)
    dists = np.array([
        [np.linalg.norm(bs[i] - bs[j]) / Cfg.area for j in range(N_BS) if j != i]
        for i in range(N_BS)
    ], dtype=np.float32)                         # [N_BS, N_BS-1]
    return np.concatenate([kpm_base, dists], axis=1)  # [N_BS, 3+(N_BS-1)]


def obs_share(p, g, serv, bs=None):
    """(C) critic 的 share_obs (sim 特權): 攤平的全域真實狀態。
    若提供 bs[N_BS,2]，額外附加 BS pairwise 正規化距離 (÷ area=500m)。"""
    base = np.concatenate([g.reshape(-1), p, serv.astype(float)])
    if bs is None:
        return base
    N_BS = bs.shape[0]
    dists = [np.linalg.norm(bs[i] - bs[j]) / Cfg.area
             for i in range(N_BS) for j in range(i + 1, N_BS)]
    return np.concatenate([base, np.array(dists, dtype=np.float32)])


# --------------------------- baselines ----------------------------
def _weighted_waterfill(w, a, budget):
    """max Σ w*log(1+a*p) s.t. Σ p ≤ budget, p≥0。回傳 p。bisection on water level。"""
    if budget <= 0 or len(w) == 0:
        return np.zeros(len(w))
    lo, hi = 1e-18, 1e18
    for _ in range(80):
        mu = np.sqrt(lo * hi)
        p = np.maximum(0.0, w / mu - 1.0 / a)
        if p.sum() > budget:
            lo = mu
        else:
            hi = mu
    mu = np.sqrt(lo * hi)
    p = np.maximum(0.0, w / mu - 1.0 / a)
    if p.sum() > 0:
        p = p * min(1.0, budget / p.sum())
    return p


def bl_equal_power(g, serv, w, N_BS):
    """floor: 每 BS 用滿 P_max, 平均分給自己的 UE。"""
    p = np.zeros(g.shape[1])
    for i in range(N_BS):
        idx = np.where(serv == i)[0]
        if len(idx) > 0:                          # 防禦: 空 BS 不分配 (理論上 gen_scenario 已保證非空)
            p[idx] = PMAX / len(idx)
    return p


def pf_wsr_ceiling(g, serv, w, N_BS, grid=6):
    """
    full-CSI 天花板 (cooperative): 對「每 BS 的總功率」做 coarse grid 搜尋,
    每個組合內各 BS 用 weighted water-filling 把預算分給自己的 UE, 取全域 PF-utility 最佳。
    捕捉合作式的 power back-off (某些 BS 降功率以減干擾) —— 這才是 social optimum 的關鍵,
    區別於 selfish 全功率。grid^N_BS 個組合, N_BS≤6 可接受。
    """
    levels = np.linspace(0.0, PMAX, grid)          # 每 BS 總功率候選
    idx_by_bs = [np.where(serv == i)[0] for i in range(N_BS)]
    best_p, best_obj = None, -np.inf
    for combo in product(range(grid), repeat=N_BS):
        B = np.array([levels[c] for c in combo])    # 各 BS 總功率
        # 給定各 BS 總功率 -> 算每個 UE 受到的 inter-cell 干擾 (與 intra 分配無關)
        # I_u = Σ_{j≠serv[u]} B[j] g[j,u]
        total = (B[:, None] * g).sum(axis=0)         # Σ_j B[j] g[j,u]
        p = np.zeros(g.shape[1])
        for i in range(N_BS):
            idx = idx_by_bs[i]
            I_u = total[idx] - B[i] * g[i, idx] + N0  # 扣掉自己 BS 的貢獻 = inter-cell
            a_u = g[i, idx] / I_u
            p[idx] = _weighted_waterfill(w[idx], a_u, B[i])
        rate, _, _ = rates_from_power(p, g, serv, N_BS)
        obj = pf_utility(rate, w)
        if obj > best_obj:
            best_obj, best_p = obj, p.copy()
    return best_p


# --------------------------- Env (for RL) -------------------------
class Env:
    """
    full-buffer, per-slot。reset() 開新 scenario;step(actions) 執行 + 回 reward。
    actions: list,每個 BS 一個 [n_ue_i] 的功率向量 (會投影到 sum ≤ PMAX)。
    """
    def __init__(self, cfg, reward_mode="difference", seed=0):
        global PMAX
        PMAX = dbm_to_w(cfg.Pmax_dBm)
        self.cfg = cfg
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)
        self.Rbar = np.full(cfg.N_UE, 1e-3)         # PF running-average rate
        self.isolated = False                         # curriculum: zero inter-cell interference

    def _weights(self):
        return 1.0 / (self.Rbar + 1e-3)

    def reset(self):
        self.g, self.serv, self.bs, self.ue = gen_scenario(self.cfg, self.rng)
        self.p = bl_equal_power(self.g, self.serv, self._weights(), self.cfg.N_BS)  # 初始功率
        return self._obs()

    def _obs(self):
        w = self._weights()
        return dict(
            local=obs_local(self.p, self.g, self.serv, w, self.cfg.N_BS, isolated=self.isolated),
            kpm=obs_kpm(self.p, self.g, self.serv, self.cfg.N_BS, self.bs, isolated=self.isolated),
            share=obs_share(self.p, self.g, self.serv, self.bs),
        )

    def _project(self, actions):
        """把各 BS 動作投影到 [0,?], Σ ≤ PMAX。"""
        p = np.zeros(self.cfg.N_UE)
        for i in range(self.cfg.N_BS):
            idx = np.where(self.serv == i)[0]
            a = np.clip(np.asarray(actions[i], dtype=float), 0.0, None)
            s = a.sum()
            if s > PMAX:
                a = a * (PMAX / s)
            p[idx] = a
        return p

    def step(self, actions):
        self.p = self._project(actions)
        w = self._weights()
        if self.reward_mode == "difference":
            r, rate = difference_reward(self.p, self.g, self.serv, w, self.cfg.N_BS, isolated=self.isolated)
            self.Rbar = (1 - self.cfg.pf_beta) * self.Rbar + self.cfg.pf_beta * rate
        elif self.reward_mode == "logpf":
            # potential-based PF: r = ΔΦ, Φ=Σ_u log(R̄_u+ε); telescopes to the eval objective.
            rate, _, _ = rates_from_power(self.p, self.g, self.serv, self.cfg.N_BS, isolated=self.isolated)
            eps = 1e-2
            Rbar_new = (1 - self.cfg.pf_beta) * self.Rbar + self.cfg.pf_beta * rate
            dlog = np.log(Rbar_new + eps) - np.log(self.Rbar + eps)        # [N_UE]
            # per-BS PF credit (each BS rewarded for its own UEs' running-rate progress)
            r = np.array([dlog[self.serv == i].sum() for i in range(self.cfg.N_BS)])
            self.Rbar = Rbar_new
        else:
            rt, rate = team_reward(self.p, self.g, self.serv, w, self.cfg.N_BS, isolated=self.isolated)
            r = np.full(self.cfg.N_BS, rt)
            self.Rbar = (1 - self.cfg.pf_beta) * self.Rbar + self.cfg.pf_beta * rate
        info = dict(pf_utility=pf_utility(rate, w), rate=rate,
                    jain=jain([rate[self.serv == i].sum() for i in range(self.cfg.N_BS)]))
        return self._obs(), r, False, info   # full-buffer: 無 terminal


# --------------------------- sanity check -------------------------
def _sanity(cfg, n=200, seed=1):
    global PMAX
    PMAX = dbm_to_w(cfg.Pmax_dBm)
    rng = np.random.default_rng(seed)
    res = {"equal": [], "ceiling": []}
    jain_res = {"equal": [], "ceiling": []}
    harm_ok = True
    for _ in range(n):
        g, serv, _, _ = gen_scenario(cfg, rng)
        Rbar = np.full(cfg.N_UE, 1e-3)
        # 用幾步暖機讓 PF 權重不全相等
        w = 1.0 / (Rbar + 1e-3)
        for name, fn in [("equal", bl_equal_power),
                         ("ceiling", lambda g, s, w, N: pf_wsr_ceiling(g, s, w, N, cfg.ceiling_grid))]:
            p = fn(g, serv, w, cfg.N_BS)
            rate, _, _ = rates_from_power(p, g, serv, cfg.N_BS)
            res[name].append(pf_utility(rate, w))
            jain_res[name].append(jain([rate[serv == i].sum() for i in range(cfg.N_BS)]))
        # 檢查 difference reward 的 harm_i ≥ 0
        p = bl_equal_power(g, serv, w, cfg.N_BS)
        for i in range(cfg.N_BS):
            rate_act, _, _ = rates_from_power(p, g, serv, cfg.N_BS)
            p_cf = p.copy(); p_cf[serv == i] = 0.0
            rate_cf, _, _ = rates_from_power(p_cf, g, serv, cfg.N_BS)
            other = serv != i
            harm = (w[other] * (rate_cf[other] - rate_act[other])).sum()
            if harm < -1e-6:
                harm_ok = False

    print(f"=== sanity ({n} scenarios, N_BS={cfg.N_BS}, N_UE={cfg.N_UE}) ===")
    print(f"{'method':<10}{'PF-utility (mean±std)':>26}{'Jain':>10}")
    for name in ["equal", "ceiling"]:
        a = np.array(res[name]); j = np.array(jain_res[name])
        print(f"{name:<10}{a.mean():>16.2f} ±{a.std():>6.2f}{j.mean():>10.3f}")
    print(f"difference-reward harm_i ≥ 0 : {'OK' if harm_ok else 'FAILED'}")
    print(f"ceiling > equal (PF-utility) : "
          f"{'OK' if np.mean(res['ceiling']) > np.mean(res['equal']) else 'FAILED'}")


if __name__ == "__main__":
    _sanity(Cfg())
