"""
Multi-cell power-control experiment: shared objective core + baselines.

設計原則: channel -> SINR -> rate/reward 是「唯一真相來源」(metrics()),
baseline / RL training / eval 三邊全部呼叫它, 才能公平比較.

依賴: 只需 numpy. 直接 `python baselines.py` 會跑一組 snapshot 的對照表.

對照組:
  full_power : 全 RB 開滿 P_max  (≈ 3GPP round-robin / naive floor)
  wmmse      : WSRM 的 strong baseline (Shi 2011 / Sun 2018 的 SISO 版)
  grid_opt   : 小規模 (3-BS) 的 near-global 參考, 用來量 WMMSE 的 optimality gap

注意: formula channel 是 frequency-flat -> 預設 per-RB cap 下 4 個 RB 退化成同一子問題.
      設 freq_selective=True 給每個 RB 獨立 fading, RB 維度才有意義.
"""
import numpy as np
from itertools import product
import time


# ----------------------------- config -----------------------------
class Cfg:
    N_BS = 3
    N_UE = 10
    N_RB = 4
    area = 500.0          # m, 正方形邊長
    fc_GHz = 3.5
    Pmax_dBm = 30.0       # per-RB 功率上限 (per-RB cap 模式)
    shadow_dB = 4.0
    freq_selective = False # True: 每個 RB 加獨立 fading, 讓 RB 不退化
    lam = 0.1             # report 的干擾懲罰權重 (只用在 shaped reward, 不影響 sum-rate)

    def __init__(self, **overrides):
        # 允許 Cfg(freq_selective=True) 等覆寫 (class 預設值仍保留為 fallback)
        for k, v in overrides.items():
            if not hasattr(type(self), k):
                raise TypeError(f"Cfg got unknown attribute '{k}'")
            setattr(self, k, v)


def dbm_to_w(x):
    return 10.0 ** ((x - 30.0) / 10.0)


def noise_w_per_rb():
    # 每個 180 kHz RB 的熱雜訊: -174 + 10log10(180e3) ≈ -121 dBm
    return dbm_to_w(-174.0 + 10.0 * np.log10(180e3))


# --------------------------- channel ------------------------------
def gen_topology(cfg, rng):
    bs = rng.uniform(0, cfg.area, size=(cfg.N_BS, 2))
    ue = rng.uniform(0, cfg.area, size=(cfg.N_UE, 2))
    return bs, ue


def path_gain(cfg, bs, ue, rng):
    """3GPP UMi 路徑損耗 (頻率以 GHz 計, 即 report 修過的 bug). 回傳線性 power gain [N_BS, N_UE]."""
    d = np.linalg.norm(bs[:, None, :] - ue[None, :, :], axis=-1)
    d = np.maximum(d, 1.0)
    PL = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(cfg.fc_GHz)
    PL = PL + rng.normal(0.0, cfg.shadow_dB, size=PL.shape)
    return 10.0 ** (-PL / 10.0)


def associate(G):
    """best-signal association: 每個 UE 接最強 BS."""
    return np.argmax(G, axis=0)


def effective_gains(G, assoc, N_BS):
    """
    A[i,i] = BS i 對「自己 UE 群」的平均增益 (desired)
    A[i,j] = BS j 對「i 的 UE 群」的平均增益 (j 對 i 的干擾)
    這完全複製 report 的 per-BS 平均 SINR 定義, 確保 baseline 與其 env 同目標.
    """
    A = np.zeros((N_BS, N_BS))
    for i in range(N_BS):
        ues = np.where(assoc == i)[0]
        src = G if len(ues) == 0 else G[:, ues]
        A[i, :] = src.mean(axis=1)
    return A


def make_snapshot(cfg, rng):
    """產生一個完整 channel snapshot: 每個 RB 一個 A 矩陣 [N_RB, N_BS, N_BS]."""
    bs, ue = gen_topology(cfg, rng)
    G = path_gain(cfg, bs, ue, rng)
    assoc = associate(G)
    A0 = effective_gains(G, assoc, cfg.N_BS)
    As = []
    for _ in range(cfg.N_RB):
        if cfg.freq_selective:
            # 每 RB 獨立 Rayleigh power fading (exponential mean=1), 打破 RB 退化
            fade = rng.exponential(1.0, size=(cfg.N_BS, cfg.N_BS))
            As.append(A0 * fade)
        else:
            As.append(A0.copy())
    return np.stack(As, axis=0), assoc  # [N_RB, N_BS, N_BS], [N_UE]


# ----------------------- objective (唯一真相) ----------------------
def sinr_rb(P_rb, A, nw):
    """P_rb:[N_BS] 該 RB 各 BS 功率; A:[N_BS,N_BS]. 回傳 [N_BS] SINR."""
    desired = P_rb * np.diag(A)
    total = A @ P_rb                # total[i] = Σ_j A[i,j] P[j]
    intf = total - desired
    return desired / (intf + nw)


def metrics(P, As, assoc, cfg, nw):
    """
    P:[N_BS,N_RB] 功率分配. 回傳所有實驗共用的指標 dict.
    sum_rate    : Σ_bs Σ_rb log2(1+SINR)  [bps/Hz]  <- 真正的 objective
    per_bs_rate : 每個 BS 的總 rate
    jain        : 以「每 UE 平均 rate」算的 Jain fairness
    reward      : report 那條 shaped reward (供對照, 不是優化目標)
    """
    N_BS, N_RB = P.shape
    per_bs = np.zeros(N_BS)
    intf_caused = np.zeros(N_BS)
    for rb in range(N_RB):
        A = As[rb]
        s = sinr_rb(P[:, rb], A, nw)
        per_bs += np.log2(1.0 + s)
        # BS i 對別人造成的干擾 (供 shaped reward)
        for i in range(N_BS):
            intf_caused[i] += sum(A[k, i] * P[i, rb] for k in range(N_BS) if k != i)
    sum_rate = per_bs.sum()

    n_ue = np.array([max((assoc == i).sum(), 1) for i in range(N_BS)])
    thr_per_ue = per_bs / n_ue
    jain = thr_per_ue.sum() ** 2 / (N_BS * (thr_per_ue ** 2).sum() + 1e-12)

    norm = N_RB * np.log2(1.0 + 1e4)
    reward = (per_bs - cfg.lam * _to_bits(intf_caused) + 0.1 * jain).sum() / norm
    return dict(sum_rate=sum_rate, per_bs_rate=per_bs, jain=jain,
                reward=reward, intf_caused=intf_caused)


def _to_bits(x):
    return np.log2(1.0 + np.maximum(x, 0.0))


# --------------------------- baselines ----------------------------
def bl_full_power(As, cfg, nw, Pmax):
    """全部 RB 開滿. ≈ 3GPP round-robin floor."""
    return np.full((cfg.N_BS, cfg.N_RB), Pmax)


def wmmse_rb(A, nw, Pmax, n_iter=100, n_init=8, rng=None):
    """
    單一 RB 的 SISO WSRM power control (Sun et al. 2018, Alg.1 的 scalar 形式).
    A[i,j] = 干擾 power gain (tx j -> rx i); A[i,i] = desired.
    回傳該 RB 的 [N_BS] 最佳功率 (per-RB cap = Pmax).
    """
    N = A.shape[0]
    Hd = np.sqrt(np.diag(A))      # desired amplitude
    best_p, best_o = None, -np.inf
    inits = []
    inits.append(np.full(N, np.sqrt(Pmax)))                 # full-power init
    if rng is not None:
        for _ in range(n_init - 1):
            inits.append(np.sqrt(rng.uniform(0, Pmax, size=N)))
    for v in inits:
        v = v.copy()
        for _ in range(n_iter):
            denom = (A * (v ** 2)[None, :]).sum(axis=1) + nw   # Σ_j A[i,j] v_j² + σ²
            u = Hd * v / denom
            w = 1.0 / np.maximum(1.0 - u * Hd * v, 1e-9)
            num = w * u * Hd
            den = (A * (w * u ** 2)[:, None]).sum(axis=0)      # Σ_j A[j,i] (w u²)_j
            v = np.clip(num / np.maximum(den, 1e-12), 0.0, np.sqrt(Pmax))
        p = v ** 2
        o = np.log2(1.0 + sinr_rb(p, A, nw)).sum()
        if o > best_o:
            best_o, best_p = o, p
    return best_p


def bl_wmmse(As, cfg, nw, Pmax, rng):
    """per-RB 獨立跑 WMMSE (per-RB cap 模式)."""
    P = np.zeros((cfg.N_BS, cfg.N_RB))
    for rb in range(cfg.N_RB):
        P[:, rb] = wmmse_rb(As[rb], nw, Pmax, rng=rng)
    return P


def wmmse_rb_weighted(A, nw, Pmax, w_pf, n_iter=100, n_init=8, rng=None):
    """PF-weighted WMMSE for a single RB. w_pf[i] = PF weight for BS i."""
    N = A.shape[0]
    Hd = np.sqrt(np.diag(A))
    best_p, best_o = None, -np.inf
    inits = [np.full(N, np.sqrt(Pmax))]
    if rng is not None:
        for _ in range(n_init - 1):
            inits.append(np.sqrt(rng.uniform(0, Pmax, size=N)))
    for v in inits:
        v = v.copy()
        for _ in range(n_iter):
            denom = (A * (v ** 2)[None, :]).sum(axis=1) + nw
            u = Hd * v / denom
            w_mm = w_pf / np.maximum(1.0 - u * Hd * v, 1e-9)
            num = w_mm * u * Hd
            den = (A * (w_mm * u ** 2)[:, None]).sum(axis=0)
            v = np.clip(num / np.maximum(den, 1e-12), 0.0, np.sqrt(Pmax))
        p = v ** 2
        sinr = sinr_rb(p, A, nw)
        o = (w_pf * np.log2(1.0 + sinr)).sum()
        if o > best_o:
            best_o, best_p = o, p
    return best_p


def bl_pf_wmmse(As, assoc, cfg, nw, Pmax, rng, n_outer=5):
    """Proportional-Fair WMMSE: outer loop updates PF weights, inner runs weighted WMMSE."""
    N_BS, N_RB = cfg.N_BS, cfg.N_RB
    P = np.full((N_BS, N_RB), Pmax)
    w_pf = np.ones(N_BS)
    for _ in range(n_outer):
        m = metrics(P, As, assoc, cfg, nw)
        tput = np.maximum(m['per_bs_rate'], 1e-6)
        w_pf = 1.0 / tput
        w_pf /= w_pf.sum()
        for rb in range(N_RB):
            P[:, rb] = wmmse_rb_weighted(As[rb], nw, Pmax, w_pf, rng=rng)
    return P


def grid_opt_rb(A, nw, Pmax, G=13):
    """3-BS 暴力 grid, 量 WMMSE 的 optimality gap 用 (僅小規模)."""
    levels = np.linspace(0, Pmax, G)
    best_p, best_o = None, -np.inf
    for combo in product(levels, repeat=A.shape[0]):
        p = np.array(combo)
        o = np.log2(1.0 + sinr_rb(p, A, nw)).sum()
        if o > best_o:
            best_o, best_p = o, p
    return best_p


def bl_grid(As, cfg, nw, Pmax):
    P = np.zeros((cfg.N_BS, cfg.N_RB))
    for rb in range(cfg.N_RB):
        P[:, rb] = grid_opt_rb(As[rb], nw, Pmax)
    return P


# ------------------------- eval harness ---------------------------
def run_eval(cfg, n_snapshots=200, seed=42):
    """在同一批 held-out snapshot 上跑所有 baseline, 印對照表."""
    rng = np.random.default_rng(seed)
    nw = noise_w_per_rb()
    Pmax = dbm_to_w(cfg.Pmax_dBm)

    methods = {
        "full_power": lambda As, a: bl_full_power(As, cfg, nw, Pmax),
        "wmmse":      lambda As, a: bl_wmmse(As, cfg, nw, Pmax, rng),
        "pf_wmmse":   lambda As, a: bl_pf_wmmse(As, a, cfg, nw, Pmax, rng),
        "grid_opt":   lambda As, a: bl_grid(As, cfg, nw, Pmax),
    }
    agg = {m: dict(sr=[], jain=[], t=0.0) for m in methods}

    for _ in range(n_snapshots):
        As, assoc = make_snapshot(cfg, rng)
        for m, fn in methods.items():
            t0 = time.perf_counter()
            P = fn(As, assoc)
            agg[m]["t"] += time.perf_counter() - t0
            mt = metrics(P, As, assoc, cfg, nw)
            agg[m]["sr"].append(mt["sum_rate"])
            agg[m]["jain"].append(mt["jain"])

    print(f"\n=== {n_snapshots} snapshots | freq_selective={cfg.freq_selective} "
          f"| N_BS={cfg.N_BS} N_UE={cfg.N_UE} N_RB={cfg.N_RB} ===")
    print(f"{'method':<12}{'sum-rate(bps/Hz)':>20}{'Jain':>10}{'gap%':>10}{'ms/snap':>10}")
    ref = np.mean(agg["grid_opt"]["sr"])
    for m in methods:
        sr = np.array(agg[m]["sr"])
        ja = np.array(agg[m]["jain"])
        gap = 100.0 * (ref - sr.mean()) / ref
        ms = 1000.0 * agg[m]["t"] / n_snapshots
        print(f"{m:<12}{sr.mean():>13.2f} ±{sr.std():>4.1f}{ja.mean():>10.3f}"
              f"{gap:>10.1f}{ms:>10.2f}")
    return agg


if __name__ == "__main__":
    cfg = Cfg()
    # 預設 frequency-flat: 會看到 grid≈wmmse, full_power 明顯較差 (interference-limited)
    run_eval(cfg, n_snapshots=200)

    # 開 frequency-selective: RB 維度才有差異
    cfg.freq_selective = True
    run_eval(cfg, n_snapshots=200)
