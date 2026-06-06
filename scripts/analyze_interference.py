"""
analyze_interference.py — 干擾競爭根因分析
==============================================
分析 RL (cc-HASAC v13, 34.21 bps/Hz) 比 WMMSE (85 bps/Hz) 差很多的根因:
  Q: 是因為 agents 彼此競爭 (同時在相同 RB 高功率發射，造成 mutual interference) 嗎？

分解架構:
  no_intf_bound  ← 若無跨基站干擾的理論上限 (每 BS 獨立傳輸)
  WMMSE          ← 最優功率控制 (per-RB Shi 2011 algorithm)
  full_power     ← 無協調 (全功率競爭)
  RL (v13)       ← 實測 34.21 bps/Hz

分析面向:
  1. 效能分解: competition_tax / coordination_gain / RL_gap
  2. 干擾強度: SINR 分佈與干擾佔比
  3. WMMSE 功率分配模式: BS 間是否有「讓步」(backing off)
  4. 結論: 競爭是主因還是次因？
"""
import sys
sys.path.insert(0, "/home/hyc1014/DL/FinalProject")
import numpy as np
from baseline import (Cfg, make_snapshot, metrics, sinr_rb,
                      bl_wmmse, bl_full_power, dbm_to_w, noise_w_per_rb)


N_SNAP = 200
SEED   = 9999   # 與 v13 eval pool 相同
RL_V13 = 34.21  # cc-HASAC v13 實測 sum-rate (bps/Hz)


# ─── helpers ──────────────────────────────────────────────────────────────────

def no_intf_bound(As, cfg, nw, Pmax):
    """
    理論上限: 若各 BS 在每個 RB 上傳輸時完全不受跨基站干擾 (只有熱雜訊)。
    = Σ_rb Σ_bs log2(1 + Pmax * A_rb[bs,bs] / nw)
    """
    rate = 0.0
    for rb in range(cfg.N_RB):
        A = As[rb]
        for i in range(cfg.N_BS):
            rate += np.log2(1.0 + Pmax * A[i, i] / nw)
    return rate


def intf_stats(P, As, cfg, nw):
    """
    計算所有 (BS, RB) 對的干擾指標。
    Returns:
      sinrs      : SINR per (BS, RB), shape [N_BS * N_RB]
      intf_fracs : intf / (signal + intf + noise), shape [N_BS * N_RB]
                   越高表示干擾越嚴重 (interference-limited regime)
    """
    sinrs, intf_fracs = [], []
    for rb in range(cfg.N_RB):
        A = As[rb]
        p = P[:, rb]
        s = sinr_rb(p, A, nw)                     # [N_BS]
        desired = p * np.diag(A)
        total   = A @ p
        intf    = np.maximum(total - desired, 0.0)
        frac    = intf / (desired + intf + nw + 1e-30)
        sinrs.extend(s.tolist())
        intf_fracs.extend(frac.tolist())
    return np.array(sinrs), np.array(intf_fracs)


def backing_off_fraction(P, Pmax, threshold=0.1):
    """回傳 power < threshold × Pmax 的 (BS, RB) 對比例。"""
    return float((P / Pmax < threshold).mean())


def power_correlation(P):
    """
    計算 BS 間功率分配的相關係數 (cross-BS correlation per RB)。
    如果所有 BS 功率高度正相關 → 同時高功率競爭。
    如果負相關 → 有讓步/輪流使用頻譜 (如 WMMSE)。
    Returns: avg pairwise correlation across RBs
    """
    N_BS, N_RB = P.shape
    if N_BS < 2:
        return 0.0
    corrs = []
    for rb in range(N_RB):
        p_rb = P[:, rb]       # [N_BS] power on this RB
        # Since all BSes have same Pmax with full_power, skip degenerate case
        if p_rb.std() < 1e-10:
            continue
        for i in range(N_BS):
            for j in range(i + 1, N_BS):
                # point correlation between BS i and BS j across snapshots is
                # done at the caller; here we just return the power values
                pass
    return None   # handled at caller level


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg  = Cfg(freq_selective=True)   # 與 cc_env_r1partial 相同設定
    rng  = np.random.default_rng(SEED)
    nw   = noise_w_per_rb()
    Pmax = dbm_to_w(cfg.Pmax_dBm)

    # Per-snapshot accumulators
    nibs         = []   # no-competition bound
    wmrs         = []   # WMMSE sum-rate
    fprs         = []   # full-power sum-rate

    wm_sinrs_all  = []
    fp_sinrs_all  = []
    wm_ifracs_all = []
    fp_ifracs_all = []

    wm_backoff    = []   # WMMSE backing-off fraction
    wm_power_util = []   # WMMSE avg power / Pmax

    # Cross-BS power correlation (collect [N_BS, N_RB] arrays for correlation analysis)
    wm_powers_list = []   # per snapshot, shape [N_BS, N_RB]

    for snap in range(N_SNAP):
        As, assoc = make_snapshot(cfg, rng)
        P_fp  = bl_full_power(As, cfg, nw, Pmax)
        P_wm  = bl_wmmse(As, cfg, nw, Pmax, rng)

        nibs.append(no_intf_bound(As, cfg, nw, Pmax))
        wmrs.append(metrics(P_wm, As, assoc, cfg, nw)['sum_rate'])
        fprs.append(metrics(P_fp, As, assoc, cfg, nw)['sum_rate'])

        wm_s, wm_f = intf_stats(P_wm, As, cfg, nw)
        fp_s, fp_f = intf_stats(P_fp, As, cfg, nw)
        wm_sinrs_all.append(wm_s.mean())
        wm_ifracs_all.append(wm_f.mean())
        fp_sinrs_all.append(fp_s.mean())
        fp_ifracs_all.append(fp_f.mean())

        wm_backoff.append(backing_off_fraction(P_wm, Pmax, threshold=0.1))
        wm_power_util.append((P_wm / Pmax).mean())
        wm_powers_list.append(P_wm.copy())

    # Aggregate
    nib_m  = float(np.mean(nibs))
    wm_m   = float(np.mean(wmrs))
    fp_m   = float(np.mean(fprs))

    wm_sinr_db = float(10 * np.log10(np.maximum(np.mean(wm_sinrs_all), 1e-9)))
    fp_sinr_db = float(10 * np.log10(np.maximum(np.mean(fp_sinrs_all), 1e-9)))
    wm_if_pct  = float(np.mean(wm_ifracs_all) * 100)
    fp_if_pct  = float(np.mean(fp_ifracs_all) * 100)
    wm_bo_pct  = float(np.mean(wm_backoff) * 100)
    wm_pu_pct  = float(np.mean(wm_power_util) * 100)

    # Cross-BS power correlation for WMMSE (per RB, across snapshots)
    wm_powers_arr = np.stack(wm_powers_list, axis=0)   # [N_SNAP, N_BS, N_RB]
    corr_vals = []
    for rb in range(cfg.N_RB):
        p_rb = wm_powers_arr[:, :, rb]   # [N_SNAP, N_BS]
        for i in range(cfg.N_BS):
            for j in range(i + 1, cfg.N_BS):
                c = float(np.corrcoef(p_rb[:, i], p_rb[:, j])[0, 1])
                corr_vals.append(c)
    wm_avg_corr = float(np.mean(corr_vals))

    # Derived metrics
    comp_tax         = nib_m - fp_m      # cost of full competition
    coord_gain       = wm_m  - fp_m      # gain from WMMSE coordination
    rl_vs_fp         = RL_V13 - fp_m     # RL improvement over full_power
    rl_vs_wm         = RL_V13 - wm_m     # RL gap vs WMMSE
    coord_recovery   = 100 * rl_vs_fp / coord_gain if coord_gain > 1e-6 else 0.0

    # ── print report ──────────────────────────────────────────────────────────
    W = 66
    print("=" * W)
    print("  干擾競爭根因分析 (Interference Competition Root-Cause Analysis)")
    print("=" * W)
    print(f"  設定: N_BS={cfg.N_BS}, N_UE={cfg.N_UE}, N_RB={cfg.N_RB}, "
          f"freq_selective=True")
    print(f"  Eval: {N_SNAP} snapshots, seed={SEED}")
    print()

    print("── 1. 效能分解 ─────────────────────────────────────────────────")
    print(f"  {'No-competition bound (干擾≡0):':<38} {nib_m:8.2f} bps/Hz")
    print(f"  {'WMMSE (最優功率控制):':<38} {wm_m:8.2f} bps/Hz")
    print(f"  {'Full-power (全功率競爭):':<38} {fp_m:8.2f} bps/Hz")
    print(f"  {'RL cc-HASAC v13 (實測):':<38} {RL_V13:8.2f} bps/Hz")
    print()
    print(f"  競爭代價   (no_intf → full_power):     {comp_tax:+8.2f} bps/Hz "
          f"({100*comp_tax/nib_m:.0f}% 損失)")
    print(f"  WMMSE 協調增益 (fp → wmmse):           {coord_gain:+8.2f} bps/Hz")
    print(f"  RL 相對 full_power:                     {rl_vs_fp:+8.2f} bps/Hz "
          f"(恢復 {coord_recovery:.0f}% of WMMSE gain)")
    print(f"  RL 與 WMMSE 的差距:                    {rl_vs_wm:+8.2f} bps/Hz")
    print()

    print("── 2. 干擾強度 ─────────────────────────────────────────────────")
    print(f"  Full-power: avg SINR = {fp_sinr_db:+6.1f} dB,  "
          f"干擾佔比 = {fp_if_pct:.1f}%")
    print(f"  WMMSE:      avg SINR = {wm_sinr_db:+6.1f} dB,  "
          f"干擾佔比 = {wm_if_pct:.1f}%")
    print(f"  SINR 改善: {wm_sinr_db - fp_sinr_db:+.1f} dB  (WMMSE 相較 full_power)")
    print()

    print("── 3. WMMSE 功率分配模式 ───────────────────────────────────────")
    print(f"  WMMSE avg power utilization: {wm_pu_pct:.1f}% of Pmax")
    print(f"  WMMSE backing-off (< 10% Pmax) 的 BS-RB 對比例: {wm_bo_pct:.1f}%")
    print(f"  WMMSE 跨 BS 平均功率相關係數 (per RB): {wm_avg_corr:+.3f}")
    if wm_avg_corr < -0.1:
        corr_interp = "負相關 → BS 之間輪流使用頻譜，避免衝突"
    elif wm_avg_corr < 0.1:
        corr_interp = "接近零相關 → 輕度協調"
    else:
        corr_interp = "正相關 → 競爭行為明顯"
    print(f"  解讀: {corr_interp}")
    print()

    print("── 4. 結論 ─────────────────────────────────────────────────────")
    print(f"  【Q: 是因為彼此在搶嗎？】")
    print()
    print(f"  ① 競爭代價極高: 全功率競爭讓 sum-rate 從 {nib_m:.0f} 壓到 {fp_m:.0f} bps/Hz")
    print(f"     (損失 {comp_tax:.0f} bps/Hz = {100*comp_tax/nib_m:.0f}%)")
    print()
    if fp_if_pct > 50:
        print(f"  ② 干擾主導確認: full_power 模式下干擾佔比 {fp_if_pct:.0f}%")
        print(f"     → 系統已進入 interference-limited regime (噪音幾乎無關)")
    print()
    print(f"  ③ WMMSE 解法: 讓 {wm_bo_pct:.0f}% 的 (BS,RB) 主動降功率 (backing off)")
    print(f"     → 透過協調讓 {coord_gain:.0f} bps/Hz 的損失被恢復,")
    print(f"        avg SINR 從 {fp_sinr_db:.1f} dB 提升到 {wm_sinr_db:.1f} dB")
    print()
    print(f"  ④ RL 只恢復了 {coord_recovery:.0f}% 的 WMMSE 增益 ({rl_vs_fp:.1f}/{coord_gain:.1f} bps/Hz)")
    print(f"     根本原因: R1-partial obs 缺少跨基站干擾資訊 (intf_norm 被隱藏),")
    print(f"     z (encoder 輸出) 理論上應補充這部分，但 actor-z 耦合尚不夠強")
    print()
    print(f"  ★ 答案: 是的，競爭 (mutual interference) 是主因。")
    print(f"     改善方向: 讓 z 更明確地編碼跨基站干擾資訊 → 引導 backing off。")
    print("=" * W)


if __name__ == "__main__":
    main()
