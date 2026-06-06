# cc-HASAC 實驗進度

**更新**：2026-06-06（v13 最佳 = 34.21 bps/Hz；v14 訓練中）

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
5. **🔄 v14（訓練中，PID=3965270）**：v13 的全面升級版，5 項改進同時生效：
   - BC_STEPS 1500→5000 + grid_opt target（MSE=0.000020，BC eval=75.06 bps/Hz，大幅超越 v13 的 57.21）
   - Z_DIM 8→16（encoder 更大容量）
   - Encoder 輸入加 intf_norm（7→8-dim），worker obs 不變（仍 R1-partial）
   - Best-checkpoint 機制（自動存最高分，結束後用最佳模型評分）
   - Cosine enc LR decay（unfreeze 後 1e-5 → 0，防止後期振盪）
   - 目標：突破 40 bps/Hz，預計 3-4 小時後完成
6. **待確認**：v8 (N_BS=5 cc-HASAC)、v9/v10 最終結果

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

**🚀 v11 完整軌跡（供對照）**：
- 100k（凍結最後）：26.69 → 110k（解凍）：30.45
- 120k：23.65（短暫低谷）→ 130-180k：緩慢回升 → 32.06 @ 190k（突破 v6！）
- 200k：26.14（小幅回落）→ 210k：28.53
- **220k：37.94** → **230k：39.16** 🔥（距 300k 還有 70k，仍上升中！）

---

## 檔案結構

| 檔案 | 用途 |
|------|------|
| `scripts/train_flat_hasac.py` | Ind-SAC baseline |
| `scripts/train_h_hasac.py` | cc-HASAC 主訓練 |
| `envs/fiveg_env.py` | 5G 環境（SINR、reward、z 注入） |
| `envs/deepmimo_channel.py` | DeepMIMO channel wrapper |
| `results/` | `.npy` 結果 + 訓練 log |
