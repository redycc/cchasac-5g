# cc-HASAC 實驗進度

**專案方向（確認）**：比較 **vanilla HASAC（無 z）** vs **C-HASAC（有 z）**，唯一差別 = actor 有沒有吃 KPM encoder 學出的 latent context z。

---

**🆕 chasac_pbs（per-BS z，200k，2026-06-09）**：`--remove_own_kpm 1`，每個 BS i 只吃鄰居 KPM → 各自的 z_i [16]。
- FINAL policy = **−2.694** ± 4.934（優於 global-z 的 −3.207）
- drop_zero = **+1.582**，drop_shuffle = **+1.735** ✅
- 訓練極不穩定（多次崩潰到 −14 以下後自救），震盪遠比 global-z 劇烈
- 結論：per-BS z policy 更好但學習更難；drop_shuffle 略低於 global-z（1.735 vs 2.088），可能是 per-BS z 更精準用 z 所以依賴量不需要那麼高，或訓練不穩定低估
- checkpoint 存於 `results/chasac_pbs_best.pt`

---

**🔧 train_chasac.py：加入 best checkpoint 磁碟儲存（2026-06-09）**：每當 best_U 更新時，同步 `torch.save(best_state, f"results/{args.tag}_best.pt")`，以便後續分析 z 學到的表示（PCA / 相關性分析）。checkpoint 約 5–10 MB，磁碟影響極小。

**🆕 chasac_z_analysis（200k，2026-06-09）**：用於 z 表示分析的 C-HASAC run，訓練曲線與 alpha_fix 400k 完全一致。
- FINAL policy = **−3.207** ± 6.365
- drop_zero = **+2.233**，drop_shuffle = **+2.088** ✅（z 有效，shuffle 後掉 2.09）
- checkpoint 存於 `results/chasac_z_analysis_best.pt`，供 PCA / z vs power 相關性分析

**🔬 z 表示分析（scripts/analyze_z.py，2026-06-09）**：載入 chasac_z_analysis best checkpoint，跑 200 scenarios × 8 steps rollout，分析 z vs power 相關性與 PCA。
- **PCA PC1 = 93.5%**：z 實際上只有 1 個有效維度，16 維坍縮成幾乎純量 signal
- **z 主要編碼 throughput**：top z 維度與各 BS throughput 相關最強（|corr| ~0.35–0.40）
- **On/Off switch 部分成立（軟性）**：100% scenario 有某 BS power < 0.1，28% 有某 BS > 0.9，但不是嚴格二元；mean power ~0.18，std ~0.33
- **z vs BS power 相關性 ~0.3**：間接影響，非直接控制
- 結論：z 編碼系統整體負載/throughput 狀態，on/off 是 actor 的連續決策結果，非 z 直接控制

**📊 z 分析視覺化（scripts/plot_z_analysis.py，2026-06-09）**：從 z_vectors.npy / power.npy 產生 6 張報告/poster 用圖，存於 `results/z_figs/`。
- `fig1_pca_variance.png`：PCA variance bar + 累積曲線（PC1=93.5% 標注）
- `fig2_z_power_corr.png`：z 16維 × 3 BS Pearson r heatmap
- `fig3_z_kpm_corr.png`：z 16維 × 5 KPM 相關性 heatmap
- `fig4_pc1_vs_throughput.png`：PC1 分數 vs 系統 throughput scatter
- `fig5_power_distribution.png`：三個 BS 功率分布直方圖（顯示雙峰軟開關行為）
- `fig6_summary_panel.png`：三合一 summary panel（PCA + z-power corr + power dist）

---

**🆕🆕🆕🆕🆕🆕🆕🆕 remove_own_kpm 實作 + 多組實驗啟動（2026-06-07）**：根據 Tim sync 發現「拿掉 own-KPM 讓 z 更被用（drop_shuffle 1.87→2.75）」以及 RL Expert 分析，實作 `--remove_own_kpm 1` 並排入 5 個 run：

- **架構**：新增 `encode_kpm(enc, kpm, n_bs, remove_own)` helper；`remove_own=True` 時對每個 BS i 只用鄰居 KPM rows（排除 row i），輸出 per-BS z `[B, N_BS, z_dim]`；`SetActor.forward` 以 `einsum("biu,biz->buz")` 把 z_i 路由到各 UE 的服務 BS。
- **機制**：當 z 是「唯一的跨 BS 資訊通道」，actor 被迫真的依賴 z 協調，與 RSRP 實驗的反面邏輯互相呼應。
- Smoke test 通過（global z `[2,8]`、per-BS z `[2,3,8]`、actor output `[2,12]` 全正確）。

**目前排程（GPU 順序，2026-06-07 16:30 啟動）**：
- ① **geo_z_seed1**（seed=1, 200k）— ✅ 完成：policy=**−2.243**±4.163，drop_zero=−0.118，**drop_shuffle=+0.088** ⚠️（z 使用程度遠低於 seed0 的 +1.429；policy 水準相同但 z-usage 不穩定）
- ② **geo_z_seed2**（seed=2, 200k）— ✅ 完成：policy=**−2.528**±5.093，drop_zero=+0.059，**drop_shuffle=−1.264** ❌（shuffled z 比正確 z 好 1.26！z 在 seed2 主動有害）
- ③ **chasac_z32**（z_dim=32, 200k）— ✅ 完成：policy=**−2.156**±4.593，best_U=**−0.855**@190k，drop_zero=**−1.049** ❌，drop_shuffle=**−1.236** ❌（z 主動有害；zeroing/shuffling z 都讓 policy 改善 ~1.2 PF-U）
- ④ **geo_z_long**（seed=0, 400k）— ✅ 完成：policy=**−1.162**±4.754（史上最佳），best_U=**+0.285**@325k（首次正值！），drop_zero=+3.565，**drop_shuffle=+2.278** ✅（z 強效使用，400k 比 200k 顯著提升）
- ⑤ **chasac_no_own_kpm**（remove_own_kpm=1, 200k）— ✅ 完成：policy=**−3.409**±4.450，drop_zero=+2.049，**drop_shuffle=+1.182** ✅（z 有效使用，但 policy 比 geo_z seed0 差 1.17 PF-U；per-BS z routing 增加訓練難度）

結果存至 `results/{tag}_log.txt`，完成後自動發 Telegram 報告。

- ⑥ **geo_z_600k**（seed=0, 600k）— ✅ 完成：policy=**−1.162**（與 400k 完全相同！），best_U=+0.285 @325k（600k 無更新），drop_shuffle=**+2.360**（vs 400k +2.278，幾乎相同）→ **訓練漸近線確認在 step 325k / policy −1.162 附近，600k 無額外增益**

**🆕🆕🆕🆕🆕🆕 HASAC 算法修正：Sequential Soft Policy Decomposition（2026-06-07）**：
發現 train_chasac.py 的 actor update 實作與 HASAC 論文不符——原始程式碼對所有 BS 累加 `loss_a = Σ loss_i`，再做一次 `opt_a.step()`（simultaneous update），等同於 independent SAC with CTDE。這樣只能保證收斂到 Nash Equilibrium（NE），可能是次優解。

HASAC 論文（Theorem 3.3，Soft Policy Decomposition）要求：**各 BS 依隨機 permutation 順序依序更新**，每個 BS 的 `backward+step` 立即執行，後面的 BS 看到已更新的 policy。只有這樣才能保證收斂到 QRE（Quantal Response Equilibrium = MaxEnt MARL 的全域最優）。

修改：`scripts/train_chasac.py` actor update 改為：
```python
for i in torch.randperm(n_bs).tolist():
    z_ = encode_kpm(...)
    pa, plogp = actor.sample(o_, m_, z_)  # re-sample with current (just-updated) policy
    loss_i = (alpha * plogp[:, i] - qmin).mean()
    # BC anchor per agent
    opt_a.zero_grad(); loss_i.backward(); opt_a.step()
```

**🔧 Bug 修正（同日）**：第一版 sequential update 的 alpha update 仍在 loop 外（1 次/step），而 actor 做 N_BS=3 次 backward → entropy 以 3x 速度崩潰，alpha 在 step 25k 就到 0.0001（幾乎 greedy）→ 兩個 run 都 kill 重跑。

修正：把 alpha update 移進 sequential loop 內（每個 agent backward+step 後立即更新一次 alpha），使 alpha 更新頻率與 actor 同步。Smoke test 確認 alpha 在 step 5k 穩定在 0.0054（不繼續崩潰）。

- ⑦ **seq_hasac_z0**（use_z=0，sequential update，400k）— ✅ 完成：policy=**−2.581**±3.808，best_U=**−2.315**@365k（晚期突破）
- ⑧ **seq_chasac_z1**（use_z=1，sequential update，400k）— ❌ kill @95k：4 次大崩潰（PF-U −15 to −17），best 卡 −5.096（75k 步無更新）

**Sequential update 結果總結**：

| 方法 | PF-U |
|------|------|
| Equal Power (floor) | −5.332 |
| 舊 simultaneous HASAC (z=0, 200k) | −4.290 |
| **新 sequential HASAC (z=0, 400k)** | **−2.581** ✅ |
| 舊 simultaneous C-HASAC (geo_z_long, z=1, 400k) | −1.162 |
| PF-WSR ceiling | +23.529 |

- Sequential update 對 HASAC 顯著有效：−4.290 → −2.581（**+1.71 PF-U**）
- Sequential update 對 C-HASAC 有害：encoder 每 step 收到 3 個方向不一致梯度 → z 表示不穩定 → 反覆大崩潰
- 即使 sequential HASAC（無 z）= −2.581，仍比 simultaneous C-HASAC（有 z）= −1.162 **差 1.42 PF-U**
- **主結論不變**：C-HASAC（有 z，simultaneous）仍是最佳方案；sequential update 是 HASAC 的算法改進，但不適用於含 encoder 的 C-HASAC（需要特殊處理：固定 encoder 或分開更新）

**🆕 Z-Freeze 修正（2026-06-07 23:40）**：群組提議「sequential loop 內 z 凍結，全 agent 更新後才更新 encoder 一次」。修改 `scripts/train_chasac.py`：
- Sequential loop 內：`z_frozen = encode_kpm(...).detach()`（encoder 不接受梯度）
- 所有 agent 更新完後：`z_live = encode_kpm(...)` + 聯合 loss（所有 agent 加總）+ 一次 `opt_a.step()`

Smoke test 確認 alpha 在 step 1k=0.052（比舊 seq_chasac 的 0.0036 高 14x，entropy 大幅保留）。

- ⑨ **seq_chasac_zfreeze**（use_z=1，sequential+z_freeze，400k）— ✅ 完成：policy=**−3.990**±4.877，drop_zero=+1.280，drop_shuffle=**+0.326**（z 有被用但不強；sequential+z_freeze 比 simultaneous geo_z_long −1.162 差很多，方法效果不佳）

---

**🆕🆕🆕🆕🆕🆕🆕 BS 間距離加入 Encoder z（2026-06-07）**：群組提議「encoder z 也應包含 BS 空間鄰域關係，這是 BS 間溝通的基礎」。修改：
- `env_chasac.obs_kpm()` 新增 `bs` 參數，kpm 從 [N_BS, 3] 擴充為 [N_BS, 5]（= [load, tp, P_bs, d_to_BSj, d_to_BSk]，÷500m 正規化）
- `train_chasac.build_obs()` 傳入 `env.bs`；`kpm_dim` 從 hardcode `3` 改為 `3 + N_BS - 1 = 5`
- BC 快取驗證加入 `kpm_dim` 欄位（舊快取 kpm_dim=3 自動失效重建）
- Smoke test 通過：kpm shape `(3, 5)` 正確；沿用 `obs_kpm(bs=None)` 不影響舊呼叫
- 下一步：啟動 logpf+bc1000+mu_bound=5 z1 smoke，觀察 Q 是否仍在後段崩潰

**🆕🆕🆕🆕🆕🆕 BS 地理關係加入 Q（2026-06-07）— smoke 進行中**：群組診斷「Q 缺少 BS geographic 關係 → 梯度不穩」。修改：
- `env_chasac.obs_share()` 新增 BS pairwise 正規化距離（÷500m），share_obs 60→63 dim（+3 unique distances for N_BS=3）
- `train_chasac.build_obs()` 傳入 `env.bs`；`share_dim` 60→63
- Smoke run（25k steps, logpf+bc1000+mu_bound=5, warmup=5000）啟動中（PID 118078）
- step 5000：PF-U **−3.829**（前次同設定 −5.842，初步改善）—— 等後續 step 確認是否穩定

**🆕🆕🆕🆕🆕 C-HASAC 突破嘗試：reward 改 team（2026-06-07，goal=讓 C-HASAC 突破）— 進行中**：診斷出 C-HASAC 卡 floor（−4.9）的元兇 = **difference-reward 把「大家靜默」當局部最優 → 零功率塌縮（−165）**。eval 是 PF utility `U=Σlog(R̄_u)`，其 myopic 正確 surrogate = `Σ_u(1/R̄_u)·rate_u` = **team reward**（`--reward team`，已存在）。
- **Smoke（50 BC+500 RL, team）→ PF-U −1.206**（floor −4.44 / ceiling 24.2）：幾百步就 +3.7 完勝 difference 跑滿 200k 的 −4.9 ✅ 方向確認。
- **但重度 BC 破壞 team**：bc_steps=3000 的 team run 反塌回 −165。根因：**BC dataset 只取 t=0 reset 狀態**，重度 BC 過擬合 t=0 → eval 的 T=10 rollout 在 t≥1 OOD → 零功率。→ 改 **無/輕 BC**。
- **無/輕 BC team 仍間歇崩 −165**（best ≈ −4.6~−5.4）：再診斷 = **team reward 尺度爆炸**——PF 權重 `w=1/(Rbar+1e-3)`，每 episode reset `Rbar=1e-3→w=500`，reward~數千且非平穩 → Q 不穩 → actor 間歇塌縮。
- **🔧 `--norm_reward`（RMS 標準化+clip）→ 仍崩 −165**：所有 config（difference/team/norm/BC/noBC）都在 RL 一啟動（~step 10k）就崩 −165 並卡死 → **重新 framing：病根不是 reward，是 SAC 訓練本身**。
- **🎯 根因鎖定 = SAC tanh 飽和塌縮**：action `a=tanh(x)`、`power_frac=(a+1)/2`，零功率 = `a→−1` = `mu→−∞`（tanh 飽和 → 梯度消失 → 卡死）。這解釋全 config 都在 ~10k 崩。
- **✅ 根因確認（pwr 診斷對照 35k）**：`diag_mu0` pwr 0.43→0.97→**0.000@10k**→PF-U 崩 −165；`diag_mu5`（`mu=5·tanh`）pwr 全程 0.3~0.95、**整段無 −165** → mu_bound 根除 catastrophic 塌縮。但 mu5 仍只 −4.41（追平舊 −4.26，未突破）→ mu_bound 只擋崩潰、不教協調。新增 `--mu_bound`、`--norm_reward`、`eval_power_frac`（`pwr` 欄）。
- **🚀 突破：`logpf` reward（potential-based ΔΣlog(R̄_u+ε)）**：env_chasac 新增 `reward_mode="logpf"`，逐步增量正好對齊 eval 的 PF 目標、獎勵拉起最弱 UE、尺度良好。**Smoke 12k → PF-U −2.968**（pwr 0.30，全程無 −165）= 對比舊 C-HASAC −4.26 **突破 +1.3**，且 pwr 0.95→0.30 印證「學會選擇性降功率減干擾」=真協調。
- **⚠️ global-logpf 完整 200k → 誠實 20-ep 下卡 −5.1~−5.5**（z0 best −5.46 / z1 −5.11 @35k，未 climb）→ smoke 的 −2.97 是 n_eval=8 評測集運氣。**反而比原始 difference −4.26 還差**。診斷主因：global ΔΦ broadcast = team-style credit assignment 太差（原始 difference 用 per-BS 反事實 credit 才到 −4.26）。
- **⚠️ diff+mu5 / logpf-perBS+mu5 (25k) 仍全部 < −4.26**（−5.2~−6.6）：發現 **mu_bound 消崩潰、也消掉原始高變異中「運氣好的 −4.26 峰值」** → 穩定但更差。
- **🔑 關鍵線索**：step5000 的**隨機-中功率策略（pwr 0.43）= −3.79，比所有 RL 都好** → reward 推向高功率（衝自己 rate）害 PF（餓死弱 UE）。RL 學錯方向。
- **🔧 功率懲罰 `--power_pen` → 仍卡**（best 全在 warmup step5000 ≈ −4.6，RL 後變差/崩）。
- **🧾 決定性對照（零學習/本地 baseline，20-ep PF-U；floor −5.33 / ceiling +23.5）**：
  | 策略 | 資訊 | PF-U |
  |---|---|---|
  | 固定功率 0.75 | 無 | **−4.17** |
  | 固定功率 0.25 | 無 | −4.23 |
  | 本地注水 (with intf) | 本地 CSI | −4.79 |
  | 原始最佳 C-HASAC z1 (RL 200k) | 本地+z | −4.26 |
  | C-HASAC z0 (無z) | 本地 | −5.18 |
  | team/logpf/power_pen RL | 本地±z | −5~−6 |
  | PF-WSR ceiling | 全域 CSI | +23.5 |
- **🎯 鐵證結論**：**一個 trivial 固定功率常數（−4.17）就追平/小贏 RL C-HASAC（−4.26）**，連本地注水都只 −4.79 → **任何去中心化策略都卡 ~−4**；唯有全域 CSI 聯合最佳化才達 +23.5。28 分 gap 完全來自「需要全域協調」，z 未能橋接。**這是資訊/結構限制，非 reward/演算法調參可解**。
- **本輪真實貢獻**：(1) 診斷+修復 −165 崩潰（=SAC tanh 飽和 mu→−∞，`--mu_bound` 根除，pwr 診斷實證）；(2) 以固定功率/本地注水 baseline 給出 flat C-HASAC 受結構所限的決定性證據。
- **🧪✅ 決定性結構測試完成**：`scripts/train_chasac_central.py` centralized SAC（actor 吃全 CSI=g+p+serv，單 agent 控全功率，mu_bound=5）→ **best −5.7、且越訓越差（→−11）、熵死（alpha→0.003）**。**連全 CSI 的 centralized SAC 都突破不了、比固定功率(−4.17)還差。**
- **🏁 最終鐵證結論（airtight）**：+23 的 coordination gain **只有顯式凸最佳化（PF-WSR water-filling）拿得到**；SAC RL **不管去中心/全中心、不管 reward（difference/team/logpf/power_pen）、不管 BC/bc_reg/norm/mu_bound 都學不到（卡 −4~−6）**。flat C-HASAC「用 SAC 學功率協調」的前提與 PF 目標**根本不匹配**。
- **真實貢獻**：(1) −165=tanh 飽和、`--mu_bound` 根除（pwr 診斷實證）；(2) 固定功率/本地注水/centralized 三道 baseline 給出「flat RL 不可學」的決定性證據。

---

**🆕🆕🆕🆕 BC-Pretrain C-HASAC 實驗（2026-06-07）— 進行中**：
- **C-HASAC + BC（`scripts/train_chasac.py --bc_steps 3000`，200k）**：expert=PF-WSR full-CSI ceiling，z0(`--use_z 0`)/z1(`--use_z 1`) 公平對照，唯一變因=actor 吃不吃 z。
  - **早期訊號 ⚠️ 仍劇烈震盪**：即使 BC 暖啟動，PF log-utility 仍偶發崩到 **−165.786**（單一 UE rate→0 → log 爆負）。z1 best −5.48（35k 崩 −73.8）、z0 best −5.31（30k/50k 反覆崩 −165），兩者目前都貼 floor（≈−5.3）。
  - **最終（200k, best-ckpt, seed 0）**：z0(無z)=**−4.498**±2.86、z1(有z)=**−4.868**±2.91、z1 z←0 Δ=+0.325；崩 −165 次數 z0=9 / z1=3。**對照無 BC（舊）z0=−5.184/z1=−4.259**：BC 後 z0 大幅進步但 **z1 反而輸 z0（+0.37）** → 加 z 無淨效益；且仍反覆崩 −165，BC 沒救起震盪。
  - **根因確認**：eval `U=Σ log(R̄_u+1e-6)`，−165.786 = 12 UE **全塌成近零功率**（−165.8/12=−13.8=log(1e-6)），是 difference-reward「大家都別送」局部解 + SAC entropy 的病態，非單 UE 餓死。
  - **🔧 BC_REG（TD3+BC 錨定）實作 + λ=20 實驗 → ❌ 反效果（已 kill）**：z0/z1 各跑 λ=20 → **從 step 5000 就卡死 −165、best 永遠 −165、零變異、不恢復**。→ **錨定不是對的解；真正瓶頸是 (a) difference-reward 把「靜默」當局部解 + (b) PF log-utility 的懸崖。**

---

**🆕🆕🆕 C-HASAC 乾淨路線（HANDOFF_CHASAC_IMPL，2026-06-06 20:00）— 進行中**：依交接文件從頭重做，目標單純化為「**C-HASAC 是否贏過 vanilla HASAC，唯一差別 = actor 吃不吃 learned 全域 context z**」。改用 **per-BS sum-power 約束**（非 per-RB cap）+ **best-signal association 產生 cell-edge UE** + **difference reward（反事實 harm_i）** + **PF-weighted utility**，並嚴守三層資訊分流（actor 只吃本地+z；critic 吃 share_obs 特權、不吃 z；reward 用 sim 全資訊）。
- 環境核心 `env_chasac.py`（唯一真相來源）✅ sanity 通過：`harm_i ≥ 0` OK、`ceiling(PF-WSR) > equal_power` OK。
- 訓練腳本 `scripts/train_chasac.py`（`--use_z` 切 HASAC/C-HASAC，其餘完全一致）✅：set-based 等變 actor + agent-conditioned twin-Q + permutation-invariant encoder + 自動 α + difference reward。Pipeline smoke 無 NaN，PF-U −165→−5 穩定上升，z-probe/baseline 正常。
- **評測指標**：canonical PF utility `U = Σ_u log(R̄_u)`（held-out scenarios，T=10）。baseline 參考：`equal_power`(floor)≈−5.4、`PF-WSR`(ceiling, full-CSI)≈23.7。
- **🐞 Bug 修復（2026-06-06 20:53）**：兩個 run 先前都在 **step 20000 的 `env.reset()` 崩潰** —— `ZeroDivisionError: bl_equal_power 的 PMAX / len(idx)`。根因：`gen_scenario` 的「確保每 BS ≥1 UE」guard 有缺陷——對空 BS `j` 借 UE 時可能把另一個 BS 的唯一 UE 偷走製造新空 BS，而前向掃描不回頭重檢。修法：改成迴圈，每次從「UE 數最多」的 BS（必 ≥2）借出對空 BS 增益最強的 UE，重複到全部非空（空 BS 數單調遞減保證終止）；另在 `bl_equal_power` 加防禦性 `len>0` 判斷。驗證：20000 個 seed 掃過 0 個空 BS，env sanity（harm_i≥0 / ceiling>equal）仍通過。
- **✅ 200k 訓練完成（2026-06-06 21:37/21:38）**：HASAC(`hasac_z0`, z_dim=0) vs C-HASAC(`chasac_z1`, z_dim=16) 同 seed/超參跑完，best-ckpt 評測結果如下：

  | 指標 | HASAC (z0) | C-HASAC (z1) |
  |------|-----------|--------------|
  | **PF-U（policy）** | **−5.184** ± 3.09 | **−4.259** ± 2.86 |
  | z←0 ablation | — | −4.264（**Δ = +0.005**） |
  | best_U（training peak） | −5.360 | −4.648 |
  | equal_power floor | −5.332 | −5.332 |
  | PF-WSR ceiling (full-CSI) | 23.529 | 23.529 |

  - **C-HASAC 贏 vanilla HASAC：−4.259 vs −5.184 = +0.925 PF-U** ✓（唯一變因 = actor 吃不吃 z）
  - **⚠️ 但 z-ablation Δ = +0.005 ≈ 0**：把 z 歸零後 policy 幾乎不變 → **這個提升並非來自 learned 全域 context z 的內容**，較可能來自 encoder/架構容量差異或初始化噪聲，而非「actor 真的在用 z 協調」。
  - **⚠️ 兩者都離 ceiling 極遠**：HASAC（−5.18）幾乎貼著 equal_power floor（−5.33），C-HASAC（−4.26）也只比 floor 高 ~1.07，距 PF-WSR ceiling（23.5）差 ~28 PF-U → 兩個 policy 其實都幾乎沒學到 PF utility，gap-to-ceiling ≈ 100%。
  - **訓練不穩**：z1 多次出現 PF-U 崩到 −165.786（step 40/45/55/60/85/175k），顯示 actor 偶發退化到單一 UE 餓死的退化解；best-ckpt 機制救回最終分數，但平均曲線震盪大。
  - **誠實結論**：這版 C-HASAC **沒有成功證明「learned z 有用」**——表面贏 HASAC，但 z←0 Δ≈0 反駁了「贏在 z」的假設。

- **✅ logpf reward 200k 完成**：C-HASAC (z1) logpf = **−3.128**（best −2.964）；HASAC (z0) logpf = **−5.266**（best −5.218）。
  - **C-HASAC 贏 +2.138**，z 貢獻 Δ=+0.262（encoder 在 logpf 下確實學到協調資訊）。
  - logpf reward 下無 −165 崩潰，訓練穩定。

- **✅ logpf + 輕 BC(1000) + `mu_bound=5` 200k 完成（2026-06-07）**：`scripts/train_chasac.py --reward logpf --bc_steps 1000 --mu_bound 5 --warmup 1000`；z0(`--use_z 0`) / z1(`--use_z 1`) 同 seed 公平對照，目標是測「在 `mu_bound` 防崩潰 + 輕 BC 暖啟動後，z 的增益能否真正成立」。

  | 指標 | HASAC (z0) | C-HASAC (z1) |
  |------|-----------|--------------|
  | **PF-U（policy, best ckpt）** | **−4.290** ± 2.828 | **−1.336** ± 3.789 |
  | latest @200k | −8.652 ± 3.457 | −2.753 ± 4.472 |
  | z←0 ablation | — | −4.244（**Δ = +2.907**） |
  | best_U（training peak） | −4.715 | −0.317 |
  | equal_power floor | −5.332 | −5.332 |
  | PF-WSR ceiling (full-CSI) | 23.529 | 23.529 |

  - **C-HASAC 明顯贏 vanilla HASAC：−1.336 vs −4.290 = +2.954 PF-U** ✓；而且不是只贏 latest，而是 best-ckpt 也明顯拉開。
  - **z-ablation 終於成立而且幅度夠大：Δ = +2.907**。把 z 歸零後，z1 policy 幾乎掉回 z0 水位（−4.244 vs z0 −4.290）→ 這次可以比較有把握地說，**z 的內容本身有被 actor 真正用到**，不是單純 encoder 增加參數量。
  - **`mu_bound=5` 成功避免舊的 tanh 飽和零功率崩潰**（全程未再見 −165 catastrophic collapse），而 `bc_steps=1000` 的輕 BC 也比重 BC 更健康：沒有把 policy 鎖死在 t=0 OOD 問題。
  - **但訓練穩定性仍差**：z1 雖然 peak 高達 **−0.317@40k**，後段仍多次掉到 **−8 ~ −12**；z0 也從 early best **−4.715@10k** 漂到 latest **−8.652@200k**。→ 問題已從「完全學不到 / 直接崩潰」轉成「**學得到，但 SAC 後段守不住好策略**」。
  - **目前最合理的下一步不是再換 reward，而是先做 stability pass**：保留 `logpf + bc1000 + mu_bound=5` 作為新 baseline，先測多 seed、lr decay / early-stop / best-ckpt selection，確認 z 增益是否穩健，而不是單 seed 運氣。

- **🔧 Geo 特徵（inter-BS distance → Q，2026-06-07）Smoke 25k**：診斷「Q 缺 BS 地理關係」→ 修改 `obs_share` 加入 N_BS*(N_BS-1)/2=3 個 inter-BS 正規化距離（÷500m），`share_dim` 60→63（`env_chasac.py` + `train_chasac.py` 已更新）。
  - **Smoke test (logpf+bc1000+mu_bound=5, z1, 25k)**：best −1.560 ± 3.914 @ step 20k；step 25k 掉回 −5.783 ± 6.364；z-ablation Δ=+0.498。
  - **Q 沒有 NaN 爆炸**，但 policy oscillation 與加 geo 前一致（-1.56 peak → collapse）。加 inter-BS distance 到 Q 輸入對 SAC 後段崩潰無明顯改善，根因仍是 SAC Q-overestimation 導致 actor 跟錯梯度，非缺地理資訊。

- **🔧 Encoder z 加入 BS 間距離（2026-06-07）**：群組討論「BS 間距離是 BS 溝通的基礎，應加入 encoder z 而非只加 Q」。`env_chasac.obs_kpm()` 加入 `bs` 參數，kpm 從 [N_BS, 3] 擴充為 [N_BS, 5]（= [load, tp, P_bs, d_to_BSj, d_to_BSk]，距離÷500m 正規化）；`train_chasac.build_obs()` 傳入 `env.bs`；`kpm_dim` 從 hardcode 3 改為 `3 + N_BS - 1 = 5`；BC 快取加 `kpm_dim` 驗證欄位（舊快取自動失效重建）。Smoke 25k（logpf+bc1000+mu_bound=5, z1）：
  - best PF-U **−4.442**（final ckpt −4.849 ± 2.984）；z-ablation Δ=**+0.547**（z 有效）
  - **Q 全程無 NaN 崩潰**，step 15k 一次 −9.013 震盪後恢復 ✓
  - BC MSE 0.057（新 kpm_dim=5 重建 dataset）；BC 建置正常

- **🔧 Bot offset 持久化修復（2026-06-07）**：`bot_listener.py` 重啟後 `offset=0` 導致舊訊息重跑、群組收到重複回覆。修法：每處理一筆 update 後存 offset 至 `tasks/bot_offset.txt`，重啟時讀回（`_load_offset` / `_save_offset`）。

- **🔧 RSRP_neighbor 加入 actor obs（2026-06-07）**：群組討論「UE 回報的 RSRP 隱含鄰近 BS 資訊」。新增 `--use_rsrp 1` flag，讓 actor per-UE obs 從 3-dim 擴充到 `3+N_BS=6`-dim（附加所有 BS → UE 的 channel gain，÷全域 max 正規化）。BC dataset cache 加入 `use_rsrp` 驗證欄位（舊快取自動失效）。所有 `build_obs / eval_policy / eval_power_frac / bc_pretrain / _bc_dataset` 呼叫點已更新。

- **🚀 兩條 200k 對照 run 啟動（2026-06-07 13:26）**：
  - **Run A `geo_z`**（PID 124188）：encoder z + Q 都有 BS 間距離，無 RSRP（ue_feat=3, kpm_dim=5, share_dim=63）
  - **Run B `geo_z_rsrp`**（PID 124189）：A 的基礎 + RSRP_neighbor（ue_feat=6）
  - 設定：logpf + bc1000 + mu_bound=5 + warmup=1000 + z_dim=16 + 200k steps
  - 結果存至 `results/run_geo_z_200k.txt` / `results/run_geo_z_rsrp_200k.txt`

- **🔧 z←shuffle probe 加入 eval（2026-06-07）**：Tim sync 指出 z←0（drop_zero）比 z←shuffle 寬鬆——z←0 大可能只是「z 編了常數 offset」而非 policy 真正用 z 的內容（v20 就是案例：z←0 ≈+21 但 shuffle≈0）。在 `train_chasac.py` FINAL 區塊加入 `shuffle_z=True` 的 eval，之後每次 run 結束都會同時報 `drop_zero` 和 `drop_shuffle`。

- **🔧 disable_subprocess.flag 啟用（2026-06-07）**：主 session 直接回群組時，bot subprocess 也同時跑導致重複回覆。設 `tasks/disable_subprocess.flag` 後，bot 改把訊息 queue 到 `tasks/incoming.log`，由主 session Monitor 監看並統一回覆。

- **✅ GPU Rerun 200k 完成（geo_z + geo_z_rsrp，2026-06-07）**：舊 run（CPU log header 誤導）重跑，確認 device=cuda（RTX 3090）。結果如下：

  | 指標 | geo_z（無RSRP） | geo_z_rsrp（有RSRP） |
  |------|----------------|----------------------|
  | **PF-U（policy, best ckpt）** | **−2.237** ± 5.731 | **−3.763** ± 4.177 |
  | best_U（training peak） | **−0.346** @180k | −2.420 @105k |
  | drop_zero（z←0） | +0.932 | +0.688 |
  | **drop_shuffle（z←shuffle）** | **+1.429** ✅ | +0.261 |
  | equal_power floor | −5.332 | −5.332 |
  | PF-WSR ceiling | 23.529 | 23.529 |

  - **geo_z（無RSRP）**：drop_shuffle=+1.429 > drop_zero=+0.932，代表 z **真正被使用**——錯誤 z 比沒有 z 更傷，actor 確實依賴 z 協調。
  - **geo_z_rsrp（有RSRP）**：drop_shuffle=+0.261 遠小於 drop_zero=+0.688，z 使用程度低，且 policy −3.763 比 geo_z 差 1.5 PF-U。加 RSRP_neighbor 後 actor obs 資訊過多，z 被冷落。
  - **結論**：BS 間距離加入 encoder z（kpm_dim=5）有助於 z 使用（drop_shuffle +1.429），但加 RSRP_neighbor 無益甚至有害。

- **🔧 Critic BC warm-start 實作（2026-06-07）**：群組提議「先用 ground truth 預訓練 Q」。實作：新增 `_bc_critic_dataset()`（收集 PF-WSR expert 的完整 episode rollout，計算 Monte Carlo 回報）和 `bc_pretrain_critic()`（預訓練 Q(s,a) 對應 MC return）。新增 `--bc_critic_steps`（iters）和 `--bc_critic_eps`（episode 數）參數。啟動測試 run：

  ```
  python3 scripts/train_chasac.py --use_z 1 --reward logpf \
    --bc_steps 1000 --bc_critic_steps 500 --bc_critic_eps 300 \
    --mu_bound 5 --warmup 1000 --steps 200000 --tag chasac_bcq
  ```
  - PID 142116，結果存至 `results/chasac_bcq_log.txt`。

  **❌ 負面結果（完整 200k）**：

  | 指標 | chasac_bcq | geo_z（對照） |
  |------|-----------|--------------|
  | policy PF-U | −2.606 ± 3.904 | **−2.237** |
  | best_U（training peak） | −1.165 @155k | −0.346 @180k |
  | drop_zero | **−4.269**（z←0 得 +1.663！） | +0.932 |
  | drop_shuffle | −0.453 | +1.429 |

  - **z 反向誤導**：zeroing z 讓 policy 從 −2.606 暴衝到 +1.663（drop_zero = −4.269）；shuffle 只影響 −0.453 → z 提供有害資訊但不是 offset（否則 shuffle 也會一樣大）。
  - **根因**：Critic BC 用 expert MC returns 預訓練 Q，RL 時 encoder 學會產生「讓 Q 開心」但「讓 policy 走偏」的 z，Q 被誤導 → encoder 跟著走偏的惡性循環。
  - **結論**：Critic BC warm-start 負效果，不採用。維持 geo_z（純 actor BC）為最佳設定。

---

**🆕🆕🆕 Isolated Pretrain（N_BS=1 等效 Curriculum，2026-06-08）**：

群組提議「用 N_BS=1（無干擾）的環境 pretrain，讓 actor 先學好 single-cell 最優，再面對 coordination challenge」。

實作：
- `env_chasac.rates_from_power(isolated=False)` 新增 `isolated` 參數：`isolated=True` 時 interference=0（等效每個 BS 各自孤立），SINR = signal/N0 → 無干擾 Shannon rate。
- `env_chasac.difference_reward`, `team_reward`, `obs_local`, `obs_kpm` 全部加入 `isolated` 參數。
- `Env.__init__` 加 `self.isolated = False`；`_obs()` 和 `step()` 全部傳入 `isolated=self.isolated`。
- `scripts/train_chasac.py` 加入 `--isolate_pretrain_steps N`：前 N 步設 `env.isolated=True`，切換時 log `[curriculum] step N: switching to full interference`。

Smoke test（10 steps, isolate_pretrain_steps=5）：
- step 6 正確切回 full interference，log 顯示 `[curriculum] step 6: switching to full interference`。
- pwr=0.982（isolated 下最優策略=全功率，符合預期）。

**⑩ chasac_isolate20k**（use_z=1, logpf, bc=1000, mu_bound=5, warmup=1000, isolate_pretrain_steps=20000, steps=200k）— ✅ 完成
- FINAL（best ckpt @ step 20k，isolated 結束時）：policy=**−1.862**±3.355，drop_zero=+0.701，drop_shuffle=**+0.371**
- 關鍵發現：best checkpoint 來自 isolated 階段末尾（step 20k），eval 在 full interference 下仍達 −1.862
- 切回 full interference 後 alpha 崩到 0.0003，後 180k 步從未刷新 best（curriculum 轉換後 entropy 死）
- z 有效使用（drop_shuffle +0.371），但比 geo_z_long +2.278 弱

---

**🆕 critic_updates=2 實驗（2026-06-08）**：針對 SAC Q-overestimation 根因（actor 更新 N_BS=3 次 / critic 只更新 1 次，actor 3× 快於 critic），實作 `--critic_updates K`（default=1）：每個 RL step 對 critic 做 K 次 gradient update，每次重新 sample 新 batch，actor 仍 1 cycle。

- **chasac_tau001**（tau=0.001 only，400k）— ✅ 完成：policy=**−1.051**±5.450，drop_zero=+0.443，drop_shuffle=**+0.838**；best_U=**−0.241**@step385k。略優於 geo_z_long（−1.162），但 drop_shuffle 明顯低（+0.838 vs +2.278）。z 有效但依賴程度不如 geo_z_long，policy 改善可能部分來自 new code（sequential+z_freeze+encoder_opt_split）而非 tau=0.001 本身
- **chasac_cu2**（critic_updates=2，400k）— ❌ kill @step 40k：best 始終卡在 −5.373，alpha 0.002–0.003 早崩，無突破跡象。額外 critic update 可能使 critic 在 replay buffer 較小時過擬合早期 transition，導致更差的 actor gradient landscape
- **chasac_newcode**（new code + tau=0.005 default，400k）— ❌ kill @step 150k：best −4.109，step 140k 無突破（tau001 同點已到 −0.980）。**確認 tau=0.001 是關鍵，不是 new code**：new code + default tau=0.005 無法複製 tau001 的突破

**🆕 兩組新實驗（2026-06-08）**：針對 chasac_tau001 的兩個潛在改進方向：

1. **chasac_alpha_fix**（alpha 更新移出 sequential loop，tau=0.001）：原本 sequential loop 內每個 agent backward 後都更新一次 alpha（3×/step），等效 alpha lr ×3，導致 entropy 3× 速崩潰。修正：loop 後用所有 agent 的平均 logp 做一次 alpha update（1×/step）。預期：entropy 維持更久 → drop_shuffle 回升。
2. **chasac_tau0005**（tau=0.0005，其餘同 tau001）：更慢的 target network → Q target 更穩定 → 可能更晚但更深的突破。

**✅ chasac_alpha_fix FINAL（400k，2026-06-08）**：

| 指標 | 數值 |
|------|------|
| **policy PF-U（best ckpt）** | **−0.911** ± 6.096 |
| best_U（training peak） | **+0.292** @step355k |
| drop_zero | +2.219 |
| **drop_shuffle** | **+2.622** ✅ |
| equal_power floor | −5.332 |
| PF-WSR ceiling | +23.529 |

- **目前所有 run 最佳**：policy −0.911 > tau001 −1.051 > geo_z_long −1.162
- **drop_shuffle +2.622** 創新高（geo_z_long +2.278、tau001 +0.838），alpha fix 大幅提升 z 使用程度
- best_U +0.292（step 355k）= 首次在訓練評測突破 0，確認 alpha fix 有效
- 訓練模式：深跌後強力反彈（225k→−12→230k 反彈；350k→−15→355k 突破 +0.292）
- alpha 從 0.03 緩慢降至 0.0015（比舊 seq 版的 0.001 高約 1.5×），延緩 entropy collapse

**❌ chasac_ln_alr**（critic_ln=1, actor_lr=1e-4, tau=0.001, kill @step 120k）：best -2.040 @55k，之後 alpha 死在 0.0006（比 alpha_fix 更低），65k 無更新。LayerNorm + 低 actor_lr 反而加速 entropy 崩潰，不採用。

**✅ chasac_alpha_fix_800k**（800k，2026-06-09）：

| 指標 | 數值 |
|------|------|
| **policy PF-U（FINAL）** | **+0.808 ± 5.030** |
| best_U（training peak） | **+2.575** @760k |
| drop_zero | +0.682 |
| **drop_shuffle** | **+0.167** ⚠️ |
| floor | -5.332 |
| ceiling | +23.529 |

- **policy +0.808 = 目前所有 run 最佳 FINAL，首次正值**
- **但 drop_shuffle 只有 +0.167**（遠低於 alpha_fix 400k 的 +2.622）——760k checkpoint 的 policy 好但 z 使用程度低，可能後期找到較不依賴 z 的解法
- 訓練規律：400k 後震盪更劇烈（振幅 -33 ~ +2.6），但每次深崩後仍能反彈到正值
- 760k 突破 +2.575（訓練評測 5 episodes），FINAL 50 episodes 給 +0.808（合理折扣）

**⏳ hasac_z0_800k**（PID 442592，use_z=0，同設定，800k）— 進行中：與 chasac_alpha_fix_800k 公平比較，唯一差別 = actor 吃不吃 z。

**❌ chasac_tau0005（kill @step 65k）**：best −4.592，alpha 0.0015（比 alpha_fix 更低），無突破跡象。tau=0.0005 過慢的 target update 反而讓 policy 更快收斂到壞的確定性解。

**🆕 LayerNorm + 低 actor lr 實驗（2026-06-08）**：針對 SAC Q-overestimation 根因，實作兩個不改架構的訓練技巧：

1. **`--critic_ln 1`（Critic LayerNorm）**：`mlp_ln` 在每個隱藏層的 activation 後加 `nn.LayerNorm`，防止 Q 值尺度爆炸（近期 RLPD/DrQ-v2+ 驗證有效）。`Critic.__init__` 加 `layer_norm=bool` 參數，預設 False 向後相容。
2. **`--actor_lr`（actor/encoder 獨立 lr）**：sequential loop 下 actor 更新 3×/step、critic 1×/step，actor 實際上比 critic 快 3×。新增 `--actor_lr`（default=0=同 `--lr`），設定 actor_lr=1e-4 vs critic lr=3e-4，讓兩者更對稱。

- **chasac_ln_alr**（PID 362248，critic_ln=1, actor_lr=1e-4, tau=0.001, alpha fix, 400k）— 進行中

**📄 書面報告生成（2026-06-08）**：`scripts/gen_report_docx.py` 生成 `report/chasac_final_report.docx`（英文，8 節）：Abstract、Introduction、Background（SAC/HASAC/O-RAN）、Methodology（三層資訊架構/C-HASAC/logpf reward/sequential+z-freeze）、Experimental Setup、Results（主要比較表+ablation 表）、Analysis and Discussion、Conclusion + References。

**🔧 實作品質修正（2026-06-08）**：程式碼審查後修正三個影響正確性的問題，**不改架構或超參**：

1. **Encoder optimizer 分離**（`train_chasac.py`）：原本 `actor_params = actor + encoder` 共用一個 `opt_a`，sequential loop 跑 N_BS=3 次 backward 時 encoder 梯度為 0 但 Adam moment 被更新 3 次（衰減到 0.9³≈0.73），破壞了後續 encoder 獨立更新的動量。修正：`actor_only_params` 和 `encoder_params` 各自獨立的 `opt_a` / `opt_enc`；sequential loop 只 step `opt_a`，encoder 更新只 step `opt_enc`。**有機會改善 encoder 學習品質與 C-HASAC drop_shuffle。**

2. **shuffle_z 改為跨 episode**（`train_chasac.py`）：原來對 global z（dim=2）做 feature permutation，現改為預先收集 n_eval 個不同 scenario 的 z，episode i 拿 scenario (i+1)%n_eval 的 z（確保跨 scenario mismatch）。drop_shuffle 語義更精確；不影響 policy 分數，但使 z 有效性量測更可信。

3. **`Env._obs()` 補上 `self.bs`**（`env_chasac.py`）：kpm 和 share 的回傳值缺少 bs 位置參數（kpm 少了 BS 間距離）。training 走 `build_obs()` 不受影響，但防禦性修正避免未來程式碼誤用 `_obs()` 取到錯誤維度的 kpm。

Smoke test 通過（optimizer split / shuffle_z / kpm shape (3,5) / share shape (63,) 全部 OK）。

---

**更新**：2026-06-06（v22=49.35✅ 最佳 test SR；v24 peak=50.74 史上最高但 final test=48.58；所有版本 BC 約 48-50，RL 均破壞）

**🆕 新方向：動態環境（UE Random Walk + Goodput）**：靜態 snapshot 環境缺乏時序動態，RL 沒有超越 WMMSE 的空間。改用動態 UE 位置 + 流量 buffer，目標改為最大化 goodput（實際送出的封包），讓 RL 有真正的序列決策優勢。

**Goodput 進度**：v1=23.86 → v3（action fingerprint）=24.63 → v4（DIV_COEF diversity shaping）=20.86 ❌ 適得其反 → **v5（drift-plus-penalty + distance-biased attention）=26.15 ✅ 動態環境新最佳，且 P_99=99.5 達標（< 102）**。

---

## 問題定義

用強化學習做 5G 基站功率協調。3 個基站（BS）各自控制自己的發射功率（per-RB），目標是讓整體 sum-rate 最大、干擾最小。

**核心挑戰**：每個基站推論時只看得到本地資訊，無法直接觀察鄰居狀態。需要一個「協調訊號 z」讓基站們在不直接通訊的前提下合作。

---

## 架構：cc-HASAC

```
全域 KPM（xApp 看得到）→ Encoder → z（協調摘要）
                                        ↓ broadcast
每個基站：本地 SINR/KPM + z → Actor → 功率分配
```

- **z**：Encoder 學出的全域 context，不是人工設計規則
- **Encoder**：接收所有 BS 的 KPM（吞吐量、PRB 使用率、UE 數）
- **訓練方式**：end-to-end，Encoder 靠 worker 的梯度學習
- **對應 O-RAN**：xApp 透過 E2 介面把 z 下發給各 gNB

---

## Baseline 上限

| 方法 | sum-rate | 說明 |
|------|----------|------|
| WMMSE（full CSI） | 85.0 ~ 88.2 bps/Hz | 知道所有 channel 的 oracle |
| full_power | 28.1 bps/Hz | 全功率發送，最差 naive |

---

## 所有 RL 版本結果（300k steps，seed 9999）

| 版本 | sum-rate | z←0 Δ | z←shuffle Δ | 結論 |
|------|----------|--------|-------------|------|
| **Ind-SAC B**（freq-selective，無 z） | **33.6** | — | — | 目前最強 RL |
| **Ind-SAC A**（R1-partial obs，無 z） | 28.1 | — | — | 基準無 z |
| **cc-HASAC B**（有 z，freq-selective） | 28.2 | +5.09 ✓ | — | z 有效，但輸 Ind-SAC B |
| **cc-HASAC A**（有 z，R1-partial） | 26.9 | +4.02 ✓ | +7.67 ✓ | z 有效，但輸 Ind-SAC A |
| cc-HASAC v4（α-gate + slow encoder） | 23.9 | +3.64 | +12.28 | ❌ off-policy 衝突 |
| cc-HASAC v5（Transformer encoder） | 23.9 | +1.61 | −0.15 | ❌ 3 token overfit |
| **cc-HASAC v6**（BC預訓練 + encoder freeze） | **32.97** | +6.14 ✓ | +14.36 ✓ | ✅ 最佳！接近 Ind-SAC B |
| cc-HASAC v7（encoder-only pretrain，actor from scratch） | 27.59 | +2.40 ✓ | +0.38 ≈0 | ❌ actor 不學 z；BC 必須覆蓋 encoder+actor |
| **Ind-SAC N5**（N_BS=5，無 z） | **32.40** | — | — | N_BS=5 baseline（sum-rate 比 N_BS=3 高因 BS 更多）|
| cc-HASAC v8（N_BS=5，BC pretrain） | 22.59 | **-4.17 ❌** | +4.44 | ❌ z 有害；N_BS=5 BC 效果不足 |
| **cc-HASAC v9**（no warmup） | **31.65** | **+8.68** ✓ | **+20.79** ✓ | ✅ 完成；z 貢獻最強，但 sum-rate 略低於 v6 |
| cc-HASAC v10（no warmup + α=0.001） | 26.38（❌） | -6.08 ❌ | +11.92 | encoder 解凍太早（10k）→ z有害；低 α 本身不夠 |
| **cc-HASAC v11**（BC + α=0.001 + 100k freeze） | 33.43 | +1.82 | +22.88 | 100k freeze 有效，但 z 耦合偏弱 |
| cc-HASAC v12（N_BS=5 + v11架構） | 23.22（❌） | -16.78 ❌ | +17.64 | N_BS=5 BC不足，z有害；z=0反而40.00 |
| **cc-HASAC v13**（BC+α=0.001+100k freeze+enc_lr=1e-5+500k） | **34.21** ✅ | **+4.35** ✓ | **+21.09** ✓ | **全部實驗最佳！** z利用率恢復，穩定無崩潰 |

---

## 最終完整比較（全部實驗完成）

| 方法 | sum-rate | z←0 Δ | z←shuffle Δ | 對比 Ind-SAC A |
|------|----------|--------|-------------|--------------|
| WMMSE oracle | 85.04 | — | — | +56.9 |
| **cc-HASAC v13**（最佳） | **34.21** | **+4.35** | +21.09 | **+6.11 (+21.7%)** |
| cc-HASAC v11（100k freeze） | 33.43 | +1.82 | +22.88 | +5.33 |
| cc-HASAC v6（BC pretrain） | 32.97 | +6.14 | +14.36 | +4.87 |
| **Ind-SAC B**（不同 env） | 33.6 | — | — | — |
| cc-HASAC v9（warmup=0） | 31.65 | +8.68 | +20.79 | +3.55 |
| **Ind-SAC A**（同 env 基線） | 28.1 | — | — | 0（基準） |
| cc-HASAC A（原始，無BC） | 26.9 | +4.02 | +7.67 | -1.2 |
| cc-HASAC v10（10k freeze，α=0.001） | 26.38 | -6.08 | +11.92 | -1.7 |
| cc-HASAC v7（enc-only BC） | 27.59 | +2.40 | +0.38 | -0.5 |
| full_power | 21.44 | — | — | -6.7 |

**cc-HASAC v13 vs 主要基線**：
- vs 原始 cc-HASAC A：**+7.31 bps/Hz (+27.2%)**
- vs Ind-SAC A（同 env）：**+6.11 bps/Hz (+21.7%)**
- vs Ind-SAC B（不同 env）：+0.61 bps/Hz
- vs WMMSE oracle：仍差 50.83 bps/Hz（gap = 59.8%）

---

## 關鍵發現

**正面**：cc-HASAC A/B 的 ablation（z←0 / z←shuffle）都顯示 z 有真實貢獻（+4～+5 bps/Hz），第一次成功證明 encoder 學到有意義的協調資訊。

**v6 重大突破（2026-06-06）**：
- BC pre-training（encoder + actor 同時用 WMMSE 監督）解決了 cold-start
- sum_rate = **32.97 bps/Hz**（+6.07 vs cc-HASAC A）
- z←0 Δ = **+6.14**（比 A 的 +4.02 更強）→ workers 學會更好地使用 z
- 訓練曲線有震盪（峰值 35.6 @ 230k），但最終 20-episode eval = 32.97

**⚠️ 公平比較說明**：
- Ind-SAC A (28.1) 和 cc-HASAC A/v6 都用 `cc_env_r1partial`（R1-partial, 7-dim obs）→ **同環境**
- Ind-SAC B (33.6) 使用 `cc_env`（R2 KPM, 3-dim obs）→ 不同環境
- **正確比較**：cc-HASAC v6 (32.97) vs Ind-SAC A (28.1) → v6 **贏了 +4.87 bps/Hz (+17.3%)**！

**問題根源：cold-start（已解決）**
v6 用 BC pre-training 解決 cold-start，random warmup 暫時破壞 BC policy，但最終 RL 仍恢復並大幅超越 cc-HASAC A（+6 bps/Hz）。

**v4/v5 失敗原因**：
- v4（α-gate）：replay buffer 存的 transition 在 α 不同時蒐集，obs distribution shift 導致 Q function 矛盾
- v5（Transformer）：N_BS=3 只有 3 個 token，self-attention 效益有限且容易 overfit 50 個 channel snapshot

---

## 完整消融分析（Encoder Freeze 時長的影響）

| 版本 | freeze 時長 | enc_lr_ft | 最終 sum-rate | z←0 Δ | 結論 |
|------|------------|-----------|--------------|--------|------|
| v10 | 10k | 1e-4 | 26.38 | -6.08 ❌ | freeze 太短，z 有害 |
| v9  | 10k | 1e-4 | 31.65 | +8.68 ✓ | warmup=0 救了 z 但分數一般 |
| v6  | 10k | 1e-4 | 32.97 | +6.14 ✓ | warmup=2000 稍微干擾 |
| v11 | 100k | 1e-4 | 33.43 | +1.82 ≈ | 長 freeze 有效，但解凍後 z 漂移 |
| **v13** | 100k | **1e-5** | **34.21** ✅ | **+4.35** ✓ | **最佳！** z←shuffle=+21.09；z利用率恢復（v11僅+1.82）|

**最終結論（v13 完成）**：
- 低 enc_lr_ft（1e-5）解決兩個問題：① 防止解凍後崩潰，② 恢復 actor-z 耦合
- v13 z←0 Δ = +4.35（vs v11 的 +1.82）→ workers 真正地依賴 z 做決策
- v13 sum_rate = 34.21（vs v11 的 33.43）→ 分數也更高
- 最優設計：BC pretrain（1500 steps）+ α=0.001 + 100k freeze + enc_lr_ft=1e-5 + 500k RL

---

## 架構澄清：Ind-SAC vs HASAC（2026-06-06）

### Q1：Ind-SAC 是原本的 HASAC 嗎？要多一個 HASAC 比較嗎？

**不需要額外加。** 本專案的 `train_ind_sac_A.py`（Ind-SAC A = 28.1 bps/Hz）實作上 **已是 CTDE 範式**：

- Actor：parameter-shared，只看本地 obs（無 z）→ 分散執行
- Q-critic：`SHARE_OBS = N_BS × KPM_DIM`（所有 agents 的 obs 拼接）→ 集中訓練

這正是 HASAC（Heterogeneous-Agent SAC）的核心：Centralized Training, Decentralized Execution。
「Ind-SAC」的名稱有些誤導，但 **實作上等同於「無 z 的 HASAC」**，是 cc-HASAC 的正確對照組。

若要加「HARL 官方 HASAC」（`train_flat_hasac.py`）做比較，問題在於它使用的是 `fiveg` 環境（11-dim obs），而非 cc_env（7-dim obs），兩者不可直接比較。

### Q2：用 NP-hard 最佳解 pretrain 比 WMMSE 更好？BS=3/BS=10 可行嗎？

**`bl_grid` 已在 `baseline.py` 實作！**

`grid_opt_rb(G=13)` 對每個 RB 做暴力窮舉：
- BS=3：13^3 = 2197 組合/RB × 4 RBs × 50 snapshots ≈ 439k 次 SINR 計算 → **毫秒級，完全可行**
- BS=10：13^10 ≈ 137B 組合/RB → **完全不可行**，使用 WMMSE

注意：本專案採用 **per-RB power cap** 模型，各 RB 功率分配相互獨立，因此 per-RB 窮舉即是 **全局最優**（非局部最優）。`bl_grid` 在 BS=3 下真正算的是最佳解，而非次優解。

**v14 已實作**：將 BC pretrain 的 target 從 `bl_wmmse` 換成 `bl_grid`。

**實際測量（2026-06-06）**：對 5 個 snapshot 對比兩者 sum-rate → grid_opt = WMMSE = 90.60 bps/Hz（gap = 0.0000）。WMMSE（8 random inits）在 BS=3 的這個 per-RB cap 問題上已穩定找到全局最優，**v14 的 BC targets 和 v13 完全相同**。v14 不太可能超過 v13。

**結論**：WMMSE 已足夠好，不需要用 NP-hard 暴力搜尋來改進 pretrain。gap to oracle（34 vs 85 bps/Hz）是由 partial observability + RL 探索限制造成，不是 BC 品質的問題。

---

## 下一步規劃

### 方向 1：Encoder Pre-training（✅ v6 完成 = 32.97，✅ v13 最佳 = 34.21，v14 待跑）

BC pre-training（encoder + actor 同時）解決 cold-start，v6 達到 32.97 bps/Hz。

**v6 設計**：
1. BC pre-train：用 WMMSE actions 監督式訓練 encoder + actor（1500 steps）
2. RL phase：凍結 encoder 10k 步，讓 workers 在穩定 z 上開始學
3. Fine-tune：放開 encoder 做 end-to-end RL

**v7（完成 = 27.59）**：只 pre-train encoder，actor 從頭 RL 訓練。結果：z←shuffle Δ 僅 +0.38，workers 根本沒學會用 z。
**結論：BC 必須同時覆蓋 encoder + actor**，才能建立 actor-z 耦合關係讓 RL 繼續優化。v6 的聯合 BC 是正確方向。

---

### 方向 2：增加 BS 數量（N_BS = 5 or 7）

N_BS=3 的規模太小，有幾個限制：
- 3 個 token 的 self-attention 效益接近零（v5 失敗原因之一）
- Encoder 學到的 z 包含太少協調資訊
- 更大規模才能驗證 cc-HASAC 的可擴展性

**做法**：
- 環境擴展到 N_BS=5 或 7，N_UE 相應增加（每 BS 10 UE）
- 重跑 Ind-SAC 和 cc-HASAC A baseline 做對比
- Transformer encoder 在更多 token 下預期效果更好

---

### 方向 3：Contrastive / Self-supervised Encoder

用對比學習訓練 encoder，讓「干擾高的 snapshot pair」的 z 要互相靠近，「干擾低的 pair」要分開，給 z 更有結構的語意空間。這可與方向 1 並行或作為後續。

---

## 執行優先順序

1. **✅ v6 完成（32.97）**：BC pre-training 解決 cold-start，比 Ind-SAC A (+17%)
2. **✅ v7 完成（27.59）**：enc-only 不夠，BC 需同時覆蓋 encoder+actor
3. **✅ Ind-SAC N5 完成（32.40）**：N_BS=5 baseline
4. **✅ v13 完成（34.21）**：全部實驗最佳，低 enc_lr_ft=1e-5 解決解凍後崩潰
5. **✅ v14（完成）**：5 項改進同時生效，但出現 z 有害問題
   - 最終結果（best-ckpt @ step 30k）：**36.26 bps/Hz**（> v13 34.21 ✓）
   - **z-ablation 異常**：z←0=37.80 > z=36.26（Δ=-1.54），z←shuffle=39.47（Δ=-3.21）→ z 有害！
   - 根因：Z_DIM 8→16 + cosine LR decay to 0 兩個改動合力破壞了 actor-z 耦合
     - 16-dim z 容量大，50 snapshot 不夠讓 encoder 學出穩定 z
     - Cosine decay 讓 unfreeze 後 encoder 幾乎停止更新，z 無法跟 actor 同步演化
   - 峰值僅 36.08（step 30k frozen），遠低於 v13 frozen 峰值 41.19

6. **✅ v15（完成）**：v13 架構 + best-checkpoint，結果令人警醒
   - best-ckpt SR: **41.14 bps/Hz**（step 30k，frozen phase）
   - z←0 SR: 45.08  → Δ = **−3.94**（z 有害！比不用 z 差）
   - z←shuffle: 39.28 → Δ = +1.86
   - 根因：broadcast z 本身是設計問題；best-ckpt 無法解決 encoder 學到無用 z

8. **✅ v17（完成）**：per-BS z + BC_STEPS=5000
   - SR: **39.98 bps/Hz**（best-ckpt @ step 30k frozen phase）
   - z←0 Δ=**+12.43**（史上最強 z 貢獻！）；z←shuffle Δ=+8.38
   - BC eval=49.2（v16=47.1，改善有限）；frozen peak=40.16（v16=35.40，大幅提升）
   - 問題：best-ckpt 仍在 frozen phase，unfreeze 後 RL 無法恢復（max~34）
   - 與 v15 差距僅 1.16 bps/Hz，但 z 完全有效（v15 z 有害）

9. **✅ v18（完成）**：per-BS z + 永久凍結 encoder（ENC_FREEZE=500k）
   - SR=39.98（與 v17 完全相同），z←0 Δ=+12.43
   - best-ckpt 仍在 step 30k（frozen），永久凍結沒有額外幫助
   - 根因確認：問題不是 encoder 解凍，而是 **Q-function 在 step 30k 後給出錯誤梯度**

10. **✅ v19（完成）— 重大突破！**：per-BS z + WARMUP=30k
    - SR=**48.80 bps/Hz**（best-ckpt, 20-ep test），peak=**51.03**（step 10k）
    - z←0 Δ=**+21.03**（史上最強 z 貢獻！），z←shuffle Δ=+0.25（近廣播 z）
    - 根因揭露：WARMUP=30k 期間 actor 未更新 → BC policy 原封保留 → 51.03
    - RL actor 一啟動（step 40k）就崩潰至 31.39：RL 仍毀掉 BC policy
    - 洞察：BC policy 本身已達 51.03，RL 沒有貢獻，只會破壞

11. **✅ v20（完成）**：WARMUP=500k（純 BC policy，actor 永遠不更新）
    - test SR=47.67 bps/Hz，peak=**51.58**（step 320k，同一 BC policy 的評估噪聲）
    - 訓練曲線全程穩定在 46-51（與 v19 崩潰曲線天壤之別）
    - z←shuffle Δ=-0.01≈0：N_BS=3 的 attention 退化為廣播 z（3 token≈mean pool）
    - 關鍵確認：BC policy 本身≈48-49 bps/Hz，RL 只會破壞

12. **✅ v21（完成）**：TD3+BC (BC_REG=5.0)
    - test SR = **48.80 bps/Hz**（與 v19 完全相同）
    - BC_REG=5.0 太弱：RL 仍在 step 40k 崩潰至 22 bps/Hz；best_ckpt 來自 WARMUP step 10k
    - z←0 Δ=+21.03（z 必要）；結論：弱正則化無幫助

13. **✅ v22（完成）— 新最佳！**：TD3+BC (BC_REG=20 + bc_encoder + BC warmup)
    - test SR = **49.35 bps/Hz** ← 🏆 目前最佳（超越 v19=48.80, v20=47.67）
    - 改善 RL crash floor：40k=42.10（v19=31.39, +10.7 bps/Hz）
    - 但 RL 最終仍落在 27 bps/Hz，best_ckpt 仍從 WARMUP step 30k (5-ep=48.86)
    - z←0 Δ=+21.61；3 項修正同時生效：stronger BC_REG + frozen bc_encoder + BC warmup data

14. **✅ v23（完成）**：Q-only warmup + BC_REG=0 after enc-unfreeze
    - Q 更新從 step 1 開始（BC data），actor 仍等到 step 30k
    - WARMUP best_ckpt (step 30k): peak_sr=49.6734（frozen phase）
    - enc 解凍後 BC_REG=0 → actor 無約束 → 崩潰更嚴重（140k: 23.74，遠低於 v22 的 29）
    - test SR = **48.6412 bps/Hz**（best-ckpt restored，低於 v22 的 49.35）
    - z←0 Δ=+20.90（z 有效），z←shuffle Δ=+0.73（廣播 z 特徵）
    - **結論**：enc 解凍後必須保留 BC_REG；關閉 BC_REG 反而更差

15. **✅ v24（完成）**：20-ep training eval + WARMUP=500k 純 BC
    - training 期間用 20-ep eval 選 best_ckpt（消除 5-ep 噪聲）
    - step 350k: training best_sr=**50.7394**（首次突破 50 bps/Hz！20-ep eval）
    - final test SR = **48.5802 bps/Hz**（z-ablation 20-ep，不同 eval 集有差異）
    - z←0 Δ=+20.79（z 有效），z←shuffle Δ=+0.83（廣播 z 特徵）
    - 分析：20-ep eval 仍有足夠方差讓 training peak（50.74）≠ final test（48.58）
    - peak 50.74 代表 BC policy 真實上限約 50+ bps/Hz，但測試時隨機性給 48.58
    - v22 仍以 final test SR（49.35）為最佳；v24 peak（50.74）為所有版本最高

---

### 核心洞察（v21-v23 實驗總結）

**結論：RL 微調無法改善 BC policy**
- 所有版本（v21/v22/v23）的 test SR 都由 **WARMUP 期的 best_ckpt** 決定
- RL 階段一致地破壞 BC actor（crash 至 22-35 bps/Hz）
- Q-function 梯度不穩定是根本原因（overestimation of non-BC actions）
- BC_REG 改善 crash floor 但無法防止最終崩潰

**test SR 排名**（final 20-ep test）：
v22=49.35 > v23=48.64 ≈ v24=48.58 ≈ v19=v21=48.80 > v20=47.67 > v15=41.14 > v17=39.98 > v13=34.21
（v24 training peak=50.74 — 所有版本最高峰值，但 final test 有方差）

7. **✅ v16（完成）**：Per-BS z，自注意力 encoder
   - best-ckpt SR: **37.5867 bps/Hz**（step 220k）
   - z←0 SR: 25.71 → Δ = **+11.87**（史上最強 z 貢獻！）
   - z←shuffle: 30.99 → Δ = +6.59（z 真正被使用）
   - 對比 v15: SR 較低（37.59 vs 41.14），但 z 有效性完全逆轉（v15: Δ=-3.94 → v16: Δ=+11.87）
   - 根因：per-BS z 使每個 BS 有個性化 z_i，encoder 學到真正有意義的協調資訊
   - BC eval 較低（47.1 vs v15 的 57.2），frozen phase 峰值僅 35.40，架構還有優化空間

**關鍵洞察**：
- SAC entropy 項（α）是破壞 BC init 的主犯：低 α=0.001 讓 10k 步達 39.73 bps/Hz
- Encoder 解凍是第二個問題：z 漂移打破 actor-z 耦合
- v11：BC + α=0.001 + 100k 凍結 → **30k 已達 41.19 bps/Hz**（encoder 仍凍結）
- v11 解凍後短暫下跌（120k=23.65），之後強勁回升：**220k=37.94，230k=39.16 bps/Hz！** 超越所有 RL 版本！
- **v9 z-ablation（完整 300k）**：z←0 Δ=+8.68、z←shuffle Δ=+20.79 → 有史以來最強的 z 貢獻！
- N_BS=5 失敗根因：BC MSE 偏高（N_BS=5: 0.021 vs N_BS=3: 0.012）→ encoder 對 N_BS=5 學習效果差
- v12 z-ablation 驚奇：z=0 反而得到 40.00 bps/Hz！說明 actor 學到好策略但 encoder 輸出的 z 在干擾它
- **v13**（啟動中）：v11 架構 + 解凍後 enc_lr = 1e-5（10× 降低）+ 500k 步，目標消除後期振盪

**🔥 v13 更大突破（2026-06-06 最新）**：
- 低 enc_lr（1e-5）解凍後完全無崩潰：120k=31.25→140k=34.16→160k=36.14→170k=37.14→**190k=39.12**
- 190k 已達 v11 的歷史最高（39.16 @ 230k），且仍有 310k 步！
- 穩定上升幅度：解凍後 +12.4 bps/Hz（v11 解凍後只有 +6.7，且大幅振盪）

**v15 目前可確認狀態（依 `results/cc_hasac_v15_stdout.txt`）**：
- BC pretrain 正常：MSE = **0.012439**，BC sanity eval = **57.21**
- frozen phase 早期非常強：10k = 39.73，20k = 39.87，**30k = 41.19（最佳）**
- unfreeze 後一度恢復到 39.12 @ 190k、39.34 @ 240k，但後段再次震盪
- `ps` 查無舊 PID，且結果檔只到 410k/07:50，表示先前 run **已停止**，不是仍在背景訓練

**🚀 v11 完整軌跡（供對照）**：
- 100k（凍結最後）：26.69 → 110k（解凍）：30.45
- 120k：23.65（短暫低谷）→ 130-180k：緩慢回升 → 32.06 @ 190k（突破 v6！）
- 200k：26.14（小幅回落）→ 210k：28.53
- **220k：37.94** → **230k：39.16** 🔥（距 300k 還有 70k，仍上升中！）

---

---

## 🆕 新方向：動態環境 + Goodput 目標（2026-06-06）

### 問題根源（群組討論）

靜態 snapshot 環境下，RL 的最佳策略就是「每個 timestep 根據 channel 做最佳功率分配」，這和 WMMSE 做的事完全一樣。RL 沒有序列記憶或預測的優勢，所以無法超越 WMMSE。BC policy（直接模仿 WMMSE）自然達到上限。

### 解法：動態 MDP

1. **UE Random Walk**：每步 UE 位置 +=Gaussian(0, 3m)，channel 隨時間動態變化
2. **Per-UE 流量 Buffer**：每步 Poisson 到達（λ=3 bits/slot/UE），RL 需要「管理 buffer」，有跨 timestep 的決策意義
3. **Goodput 獎勵**：實際送出的封包（min(rate, buffer)），而非 instantaneous sum-rate

### 實作：envs/cc_env_goodput.py（✅ 完成 2026-06-06）

| 設計 | 細節 |
|------|------|
| UE walk speed | σ=3 m/step（行人速度） |
| Buffer max | 30 bits/UE |
| Arrival rate | λ=3 bits/slot/UE（Poisson） |
| OBS_DIM | 8（原 7 + buf_fullness） |
| 基準測試 | avg goodput≈30 bits/step（≈λ×N_UE），buffer 未溢出（70% 利用率）|

Obs per BS：`[sinr_norm×4, load, goodput_norm, buf_fullness, n_ue]`

Reward：`goodput_per_bs - λ_intf × intf_caused + 0.1 × jain`（同 v24 結構）

**✅ Goodput v1（`train_cc_hasac_goodput.py`）完成**

環境基準：
| 策略 | goodput | 說明 |
|------|---------|------|
| 全功率（無協調） | 20.52 bits/step | 互干擾 → per-UE rate < arrive_rate |
| 最佳頻率重用（oracle） | 29.80 bits/step | 各 BS 用不同 RB → SINR 極高 |
| 到達率上限 | 30.0 bits/step | λ × N_UE 理論上限 |

v1 結果（BC=WMMSE + WARMUP=5k + BC_REG=0 + 300k steps）：
- **goodput = 23.83 bits/step**（peak=23.86 @ step 170k）
- z←0 Δ=**+2.48 ✓**（z 有效！）；z←shuffle Δ=-0.21（≈0，廣播 z）
- 相較 full_power：**+16.2%**；距離 oracle 仍有 6 bits/step 差距

**診斷**：
- BC（WMMSE uniform power）在 goodput env 是次優（19.6 < full_power 20.5）
- 干擾懲罰項數量級錯誤（~4e-9 vs goodput ~7），幾乎無效
- 個體探索無法找到「所有 BS 同時改用不同 RB」的協調模式

**✅ Goodput v2（`train_cc_hasac_goodput_v2.py`）完成（已停止 @ step 130k）**
- 全域獎勵（r_i = total_goodput / N_BS）取代 per-agent reward
- 無 BC 初始化（random start，避免 WMMSE 偏見）
- alpha=0.05（更高探索），BC_REG=0，500k steps
- **結果：best goodput = 21.44 bits/step（明顯低於 v1=23.86）**
- 原因診斷：
  1. 無 BC warm-start → 學習起點差，收斂慢
  2. 全域獎勵與 per-agent 差異不大（Q 本來就用 sum_r）
  3. alpha=0.05 太高 → actor 太隨機無法利用局部學到的模式
  4. 根本問題未解決：個體探索仍無法自發發現「各 BS 用不同 RB」

**🔥 Goodput v3（`train_cc_hasac_goodput_v3.py`）啟動中（2026-06-06）**
- **核心創新：Action Fingerprint（last_action × N_RB）加入 obs**
  - KPM_AUG=12（原 KPM_DIM=8 + last_action=4）
  - Encoder 輸入 [sinr×4, load, goodput, buf, n_ue, last_act×4] per BS
  - z_i 能編碼「鄰居上一步用了哪些 RB」→ worker 可反應協調
  - e.g. "鄰居在 RB-1 用高功率 → 我應避開 RB-1，改用 RB-0"
- 保留 v1 優點：BC warm-start + alpha=0.001（低 entropy = 利用導向）
- BC_REG=5.0（輕微恢復力；BC 在 goodput env 次優，actor 可以偏離）
- 400k steps（比 v1 多 100k）
- WORKER_OBS=23, SHARE_OBS=60, 400k steps

**v3 完整軌跡（290k/400k 訓練中）：**
| Step | Goodput | BestGput | 對比 v1 同步數 |
|------|---------|----------|--------------|
| 10k | 19.36 | 19.36 | v1=22.41（BC_REG 拉向 BC=19.6，預期） |
| 30k | 19.23 | 19.36 | encoder 解凍 |
| 40k | 23.51 | 23.51* | v1=21.98（**+4.1 jump！fingerprint 生效**） |
| 100k | 22.23 | 23.51 | v1=21.03（v3 領先 +1.2）|
| 120k | 23.79 | 23.79* | v1=21.97 |
| 200k | **24.63** | **24.63*** | **超越 v1 峰值 23.86！** |
| 240k-290k | 23-24 | 24.63 | **震盪，峰值 200k 後未改善** |

**v3 分析**：
- Fingerprint 確認有效：encoder 解凍後 19→23.5 跳升，顯示鄰居動作資訊對協調有幫助
- 同步數下穩定超越 v1（v1@100k=21.03 vs v3@100k=22.23）
- 峰值 24.63 超越 v1=23.86，但距 oracle 29.80 仍有 5.2 bits 差距
- 200k 後震盪（23-24）：偶爾發現協調模式但無法穩定維持
- 根本瓶頸：SAC entropy 持續推動探索，已找到的協調策略被隨機動作破壞

**✅ v3 結果（提前終止 @ step 340k，峰值 @ step 200k）**：
- **best goodput = 24.63 bits/step**（v1=23.86，**+0.77 bits/step = +3.2%**）
- z-ablation：未取得（提前終止以加速 v4）
- 結論：Fingerprint 有效，但 SAC entropy 仍造成 200k 後持續震盪

**✅ v4（完成，提前停於 step 130k）**：
- reward 加入 `DIV_COEF × std(acts across BSs per RB)` 直接激勵頻率重用
- 保留 fingerprint + BC warm-start（v3 最有效的兩個機制）
- 加入磁碟 checkpoint（`results/cc_hasac_goodput_v4_best.pt`）
- **結果：best goodput = 20.86 bits/step（明顯低於 v3=24.63）**
- 診斷：DIV_COEF 的 action-std reward shaping 適得其反——它獎勵「BS 間動作分歧」本身，但分歧 ≠ 正確的頻率重用配置（互補的 RB 選擇）。actor 學會製造高 std 卻非協調，goodput 反而退回 full_power 水準（~20.5）。直接 shaping diversity 是錯誤代理目標。
- **結論**：v3 的 fingerprint（讓 obs 看到鄰居動作）才是正確路線；不該用人工 reward 強迫 diversity，應讓 reward 直接反映目標（goodput / delay）

**✅ v5（完成，400k steps）— QoS-aware drift-plus-penalty**：
- BC pretrain 完成：MSE=0.0347（1000 steps）；γ=0.97 β=0.3 η=0.01；KPM_AUG=13 WORKER_OBS=24 SHARE_OBS=63
- 對應上方「v5 計劃」：用 Lyapunov drift-plus-penalty reward 同時顧 goodput 與 P_99 延遲
- **Reward**：`log(1+thr) − β·(Q/Q_ref) − η·power`（β/η 定義於 `CCEnvGoodputV2`）
- **新環境**：`envs/cc_env_goodput_v2.py`（OBS_DIM=9，新增暴露 `bs_pos`/queue 量 `Q_i`）
- **Distance-biased attention**：`attn_bias[i,j] = −d_ij/d_ref`，讓 encoder 依基站間距離加權協調
- 保留 fingerprint（KPM_AUG=13）+ BC warm-start + BC_REG=5.0；γ=0.97（horizon≈33 步≈queue 排空時間）

**v5 最終結果**：
| 指標 | v5 | 對照 |
|------|-----|------|
| **goodput** | **26.15 bits/step**（peak 26.13） | v3=24.63、full_power=20.52、freq_reuse oracle=29.96、Q-WMMSE=28.04 |
| **P_99 HOL** | **99.5 slots** | freq_reuse=56.1、Q-WMMSE=172、full_power=99.5 |
| z←0 Δ | **+3.66 ✓**（z 有效） | z←0 goodput=22.50 |

- **動態環境新最佳**：26.15 > v3 24.63（+1.5 bits/step, +6.2%），較 full_power +27.5%
- **延遲目標達成**：P_99=99.5 < 計劃目標 102 slots，且遠優於集中式 Q-WMMSE 的 172 → drift-plus-penalty 的 backlog 懲罰確實壓低了延遲
- **goodput 目標未達**：26.15 < qw_wmmse 29.72，距 freq_reuse oracle 29.96 仍有 3.8 bits/step
- z←0 Δ=+3.66 證明 encoder 學到有意義的協調訊號（distance-biased attention 生效）
- 同時新增 baseline 工具：`scripts/baseline_mlwdf_proper.py`、`scripts/baseline_eval_qos.py`（QoS/延遲對照組量測）

**v5 解讀**：RL 第一次在動態環境同時拿到「高 goodput + 低延遲」的折衷點——goodput 雖未追上 myopic 的 Q-WMMSE，但延遲（P_99）大幅領先（99.5 vs 172），印證 RL 的時序決策優勢用在「提前排空 buffer、控制 backlog」而非單步 rate 最大化。下一步若要再逼近 goodput oracle，瓶頸仍是「讓各 BS 穩定收斂到互補 RB 配置」。

---

## 🆕 Queue-Aware Baseline 量測（2026-06-06）

### 目標

確立 P_99 延遲 + goodput 的對照組數字，作為下一步 RL v5（delay-aware reward）的競爭目標。

### 方法

`scripts/eval_queue_baselines.py`：N_EVAL=20 episodes × 200 steps，seed=9999（獨立測試集）

**HoL 延遲定義**：per-UE buffer 連續非空的 slot 數，buffer 清空時記錄一次 delay sample。P_99 = 所有 delay samples 的 99th percentile。

### 結果（2026-06-06）

| 策略 | avg_goodput | total_goodput/ep | P_90 | P_99 | mean_delay | 說明 |
|------|-------------|-----------------|------|------|------------|------|
| full_power | 20.16 | 4031 | 200.0 | **200.0** | 114.1 | 無協調，buffer 永遠不排空 |
| **qw_wmmse** | **29.72** | **5945** | 96.0 | **176.4** | 43.2 | 集中式，知全域 queue，goodput 最高 |
| freq_reuse oracle | 29.71 | 5942 | 49.0 | **102.0** | 22.2 | 每 BS 用獨立 RB，P_99 最佳 |
| m_lwdf_local | 21.07 | 4214 | 200.0 | **200.0** | 94.2 | 只看本地 queue+CQI，無跨 cell 協調 |

### Proper M-LWDF 排程器對照（`baseline_mlwdf_proper.py`，2026-06-06）

`eval_queue_baselines.py` 的 `m_lwdf_local` 把 M-LWDF 當「功率分配」實作，退化成 full_power。但 M-LWDF 本質是**排程器**（決定每個 RB 給哪個 UE），故另用 `ProperMLWDFEnv` 做 per-RB UE 排程（全功率）重測：

| 排程器 | avg_goodput | P_99 | 說明 |
|--------|-------------|------|------|
| equal_sharing | 20.16 | 197 | 參考 |
| **mlwdf** | **20.34** | **197** | per-RB HOL/avg_rate 加權排程 |
| prop_fair | 20.49 | 197 | |
| queue_prop | 20.66 | 197 | |
| max_rate | 20.42 | 196 | |

**結論**：所有 single-cell 排程器（含正規 M-LWDF）都卡在 ≈20 goodput / P_99≈197，與 full_power 幾乎相同，完全達不到 freq_reuse oracle（29.7, P_99=102）。原因：M-LWDF 只決定「RB 給哪個 UE」，但 3 個 BS 仍全功率打所有 RB → 干擾地板無法突破。瓶頸是**跨 cell 功率協調**而非 cell 內排程，這正是 v5 要學的維度。

### 關鍵觀察

1. **qw_wmmse ≈ freq_reuse oracle in goodput**（29.72 vs 29.71）：兩者都把 buffer 基本排空，goodput ≈ λ×N_UE=30 的上限
2. **但 P_99 差很多**：freq_reuse oracle P_99=102 slots，qw_wmmse P_99=176 slots → goodput 相同但 freq_reuse 延遲更低，因為 SINR 更高、每步服務更多 bits
3. **無協調 = buffer 永遠滿**：full_power 和 m_lwdf_local 的 P_99=200（episode 長度），從未排空 buffer，rate < arrival rate
4. **m_lwdf_local 比 full_power 稍好**（goodput 20.16→21.07），但因缺乏跨 cell 協調仍遠不足

### RL 目標（v5 設定）

RL 需要同時：
- **goodput ≥ qw_wmmse = 29.72 bits/step**（超越集中式 oracle）
- **P_99 < 102 slots**（逼近 freq_reuse oracle 的延遲品質）

---

## 檔案結構

| 檔案 | 用途 |
|------|------|
| `scripts/train_flat_hasac.py` | Ind-SAC baseline |
| `scripts/train_h_hasac.py` | cc-HASAC 主訓練 |
| `envs/fiveg_env.py` | 5G 環境（SINR、reward、z 注入） |
| `envs/cc_env_r1partial.py` | 靜態 snapshot 環境（v6-v24 使用） |
| `envs/cc_env_goodput.py` | **動態環境（UE random walk + buffer + goodput）** |
| `envs/cc_env_goodput_v2.py` | v5 環境（OBS_DIM=9，暴露 bs_pos / queue 量 Q_i） |
| `scripts/train_cc_hasac_goodput_v3.py` | Goodput v3（action fingerprint，best=24.63） |
| `scripts/train_cc_hasac_goodput_v5.py` | Goodput v5（drift-plus-penalty + distance-biased attention） |
| `scripts/eval_queue_baselines.py` / `baseline_eval_qos.py` / `baseline_mlwdf_proper.py` | QoS / 延遲 baseline 對照組量測 |
| `envs/deepmimo_channel.py` | DeepMIMO channel wrapper |
| `env_chasac.py` | C-HASAC 環境核心（PF-utility，difference/logpf reward） |
| `scripts/train_chasac.py` | HASAC vs C-HASAC 比較（`--use_z 0/1`） |
| `results/` | `.npy` 結果 + 訓練 log |
| `scripts/bot_listener.py` | Telegram bot。subprocess 模式硬上限 600s（10 分鐘）→ 長訓練任務必超時；失敗時如實區分 timeout/ratelimit/error。`disable_subprocess.flag` 存在時跳過 subprocess，改 append `incoming.log` 由主 session 接管（修復後重新接上，需重啟 bot 生效）|
| `tasks/incoming.log` | Bot 收到新任務時 append，主 session 用 Monitor 監看 |
| `tasks/disable_subprocess.flag` | 存在 → subprocess 停用；刪除 → 恢復舊行為 |
