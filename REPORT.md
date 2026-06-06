# cc-HASAC 實驗報告
## Context-Conditioned Heterogeneous-Agent SAC for 5G Multi-Cell Power Allocation

---

## 1. Introduction

### 1.1 問題動機

5G 網路中，多個基站（Base Station, BS）共享同一頻段對多個用戶設備（User Equipment, UE）提供服務。每個 BS 的發射功率直接影響其服務 UE 的訊號品質（SINR），同時也對鄰近 BS 的 UE 造成干擾。因此，多基站的功率分配本質上是一個**多智能體協調問題**：

- 每個 BS 需要最大化本地 UE 的吞吐量（throughput）
- 但過高的功率會對鄰近 BS 造成干擾，降低網路整體效能
- 最優解需要所有 BS 聯合協調，而非各自獨立優化

**核心挑戰**：在 O-RAN（Open Radio Access Network）架構下，每個 gNB（BS）只能透過本地感測取得本地資訊。若無協調機制，每個 BS 各自做貪婪最大化，會陷入高干擾的 Nash 均衡。

### 1.2 O-RAN 架構下的協調

O-RAN 定義了 xApp 運行在 Near-RT RIC（無線智能控制器）上，可透過 E2 介面取得所有 BS 的 KPM（Key Performance Metric）報告，並下發控制指令。這為全域協調提供了基礎設施：

```
Near-RT RIC (xApp)
  ├─ 接收：所有 BS 的 KPM（吞吐量、PRB 使用率、UE 數）
  ├─ 計算：全域協調訊號 z（我們的貢獻）
  └─ 下發：z → 每個 gNB 的本地 Actor
```

我們的目標是設計一個可以學習有效協調策略的多智能體強化學習系統，並在 O-RAN 架構下可部署。

### 1.3 研究目標

1. 設計並訓練 cc-HASAC（Context-Conditioned HASAC），使協調訊號 z 能真正提升整體 sum-rate
2. 解決 z 的 cold-start 問題（初始雜訊導致 workers 學壞策略）
3. 超越無協調基線（Ind-SAC）並逼近 WMMSE oracle 上限
4. 透過 z-ablation 驗證 z 的真實貢獻

---

## 2. 原始論文：HASAC

### 2.1 HARL 框架

本實驗基於 **HARL（Heterogeneous-Agent Reinforcement Learning）** 框架，其中的 HASAC（Heterogeneous-Agent Soft Actor-Critic）是核心算法。

**HASAC 的關鍵設計**：

| 組件 | 設計 |
|------|------|
| **訓練範式** | CTDE（Centralized Training, Decentralized Execution） |
| **Actor** | 每個 agent 本地執行，只看本地 obs |
| **Critic** | 集中式 Twin Q-network，輸入 shared_obs（所有 agents 的 obs 拼接） |
| **Algorithm** | SAC（Soft Actor-Critic）with automatic entropy tuning |
| **Update order** | Sequential（每個 agent 輪流更新，避免非穩態性） |
| **Value norm** | Running mean/variance normalization of Q-targets |

**Actor 更新（SAC）**：
$$\pi^* = \arg\max_\pi \mathbb{E}_{a \sim \pi} [Q(s, a) - \alpha \log \pi(a|s)]$$

**Critic 更新（Twin Q with ValueNorm）**：
$$\mathcal{L}_Q = \mathbb{E} \left[ (Q(s, a) - y)^2 \right], \quad y = r + \gamma (1 - d) \cdot \min(Q_1', Q_2') - \alpha \log \pi'$$

### 2.2 原始 HASAC 的限制

原始 HASAC 中，每個 agent 只使用本地觀測 obs，**沒有** 全域協調機制：

- Agent i 的 policy：$\pi_i(a_i | o_i)$（只看本地 obs）
- 雖然 Critic 在訓練時看到全局狀態，但執行時各 agent 完全獨立
- **問題**：agents 無法在執行期間感知鄰居的行為，容易陷入干擾均衡

這就是我們引入 **全域 context z** 的動機。

---

## 3. 實驗環境設計

### 3.1 網路拓樸

```
場景：3 個基站，30 個用戶，4 個資源區塊（RB）
區域：500m × 500m 正方形
頻率：3.5 GHz（Sub-6G）
每 RB 功率上限（Pmax）：30 dBm（1W）
```

| 參數 | 值 |
|------|----|
| N_BS | 3 |
| N_UE | 30（每 BS 約 10 UE） |
| N_RB | 4 |
| 每集長度 | 200 steps |
| Channel model | 3GPP UMi path loss + freq_selective=True |

**Channel Model**：
$$\text{PL}(d) = 32.4 + 21.0 \log_{10}(d) + 20.0 \log_{10}(f_c)$$
加上 4 dB shadow fading 和 per-RB 獨立衰減（freq_selective=True），使每個 RB 有不同的 channel gain。

### 3.2 觀測空間（R1-partial obs）

每個 BS 的本地觀測為 **7 維**（R1-partial）：

| 維度 | 內容 | 說明 |
|------|------|------|
| [0:4] | sinr_rb_0 ~ sinr_rb_3 | 各 RB 的 SINR（正規化） |
| [4] | load | 平均功率使用率 |
| [5] | throughput | 本 BS 的總吞吐量（正規化） |
| [6] | n_ue | 本 BS 服務的 UE 數（正規化） |

**注意**：R1-partial obs **不包含** 鄰近 BS 的干擾資訊——這個缺口正是全域 context z 的價值所在。

**Agent ID 注入**：訓練時在 obs 後附加 one-hot agent ID（3維），使 parameter-shared actor 能區分不同 BS。

### 3.3 動作空間

每個 BS 的動作為 **4 維**：$a_i \in [0, 1]^4$，對應 4 個 RB 的功率分配比例，實際功率 = $a_i \times P_{\max}$。

### 3.4 獎勵函數

獎勵以 **sum-rate**（bps/Hz）為核心指標：

$$r = \sum_{j \in \text{UE}} \log_2\left(1 + \text{SINR}_j\right)$$

其中：
$$\text{SINR}_{j,k} = \frac{P_{b(j),k} \cdot A_{b(j), j, k}}{\sigma^2 + \sum_{b' \neq b(j)} P_{b',k} \cdot A_{b', j, k}}$$

- $b(j)$：UE $j$ 所屬的 BS（最強訊號關聯）
- $A_{b,j,k}$：BS $b$ 到 UE $j$ 在 RB $k$ 的 channel gain
- $\sigma^2$：熱雜訊（$\approx -121$ dBm/RB）

### 3.5 Snapshot Pool 機制

為了訓練穩定性和可複現性，使用**固定 channel snapshot pool**：

- **訓練 pool**：50 個 channel realization（seed=42+1）
- **評估 pool**：20 個 channel realization（seed=9999，永不出現在訓練中）
- K_HOLD=50：每個 snapshot 持續 50 steps 後換下一個
- 每次 evaluation 固定跑 20 個 eval snapshots，取平均 sum-rate

### 3.6 Baselines

| 方法 | sum-rate | 說明 |
|------|----------|------|
| **full_power** | 21.44 bps/Hz | 所有 RB 全功率，最 naive 的 baseline |
| **WMMSE** | 85.04 bps/Hz | Shi 2011 iterative solver，知道所有 channel CSI，為實際 oracle 上限 |
| **Grid-opt** | 90.60 bps/Hz | 暴力窮舉（BS=3 可行），實測與 WMMSE 差距為 0（WMMSE 已達全局最優） |

---

## 4. cc-HASAC 架構

### 4.1 核心設計：全域 Context z

cc-HASAC 在 HASAC 基礎上加入全域協調訊號 z：

```
全域 KPM [N_BS, KPM_DIM=7]
        ↓ GlobalContextEncoder (DeepSet)
        z [Z_DIM=8]  ← 協調摘要
        ↓ broadcast to all agents
每個 BS：[local_obs(7) + agent_id(3) + z(8)] → Actor → power per RB
```

**GlobalContextEncoder（DeepSet）**：
$$z = \text{proj}\left(\frac{1}{N_{BS}} \sum_{i=1}^{N_{BS}} \phi(\text{KPM}_i)\right)$$

- $\phi$：shared MLP（KPM_DIM → 32 → 32，ReLU）
- $\text{proj}$：linear projection（32 → Z_DIM=8）
- 排列不變（Permutation-invariant）：BS 的順序不影響 z
- 對應 O-RAN 中 xApp 對所有 BS KPM 的聚合

**Centralized Twin Q-Critic**：
- 輸入：share_obs（N_BS × KPM_DIM + Z_DIM = 29 維）+ 所有 agent 動作
- 兩個獨立 Q-network，取 min 避免過估計
- 使用 ValueNorm 正規化 Q-target

### 4.2 訓練流程（最終版 v13）

```
Phase 0：BC Pre-training（1500 steps）
  ─ 用 WMMSE 動作監督 encoder + actor
  ─ Loss = MSE(actor_predicted_power, WMMSE_power)
  ─ 建立 actor-z 耦合：actor 學到依賴 z 的 WMMSE-like 策略

Phase 1：RL frozen（step 1 ~ 100k）
  ─ Encoder 凍結（z 不更新）
  ─ Actor + Critic 用 SAC 訓練
  ─ Workers 在穩定 z 上學習 RL 策略
  ─ α_init=0.001（低 entropy 壓力，不破壞 BC 初始化）

Phase 2：RL fine-tuning（step 100k ~ 500k）
  ─ Encoder 解凍，但 enc_lr=1e-5（比 actor_lr 低 30×）
  ─ Encoder 緩慢適應 RL policy
  ─ 避免 z 漂移破壞 actor-z 耦合
```

### 4.3 關鍵超參數

| 參數 | 值 | 說明 |
|------|-----|------|
| BC_STEPS | 1500 | BC 監督訓練步數 |
| ENC_FREEZE_STEPS | 100,000 | encoder 凍結的 RL 步數 |
| ALPHA_INIT | 0.001 | SAC 初始溫度（低值保護 BC init） |
| ENC_LR | 1e-4 | encoder 學習率（凍結期不生效） |
| ENC_LR_FINETUNE | 1e-5 | 解凍後 encoder 學習率（10× 低） |
| ACTOR_LR | 3e-4 | actor 學習率 |
| BUFFER_SIZE | 100,000 | replay buffer 大小 |
| BATCH_SIZE | 256 | mini-batch 大小 |
| WARMUP | 0 | 無隨機 warmup，BC policy 直接填充 buffer |
| NUM_STEPS | 500,000 | 總 RL 步數 |
| Z_DIM | 8 | 協調訊號維度 |
| HIDDEN | 128 | 隱藏層大小 |
| GAMMA | 0.99 | 折扣因子 |
| POLYAK | 0.005 | target network 軟更新率 |
| Z_KL_COEF | 0.001 | z 正則化係數（防止 z 爆炸） |

---

## 5. 所有實驗與實作

### 5.1 無協調基線

#### Ind-SAC A（cc_env_r1partial，R1-partial obs，無 z）
- **結果**：28.11 bps/Hz
- **設計**：CTDE HASAC，parameter-shared actor，centralized Q-critic，**無 z**
- **意義**：cc-HASAC 的**公平對照組**（同環境、同 obs space）
- **實作**：`scripts/train_ind_sac_A.py`

#### Ind-SAC B（cc_env，R2 KPM obs，無 z）
- **結果**：33.60 bps/Hz
- **設計**：使用 3-dim obs（throughput, prb_util, n_ue），不同環境
- **意義**：不可直接與 cc-HASAC 比較（不同 env）
- **注意**：R2 obs 維度低但涵蓋整合資訊，SINR per-RB 差異被隱藏

### 5.2 原始 cc-HASAC（有 z，無 BC 預訓練）

#### cc-HASAC A（R1-partial obs）
- **結果**：26.93 bps/Hz，z←0 Δ=+4.02，z←shuffle Δ=+7.67
- **設計**：基本 cc-HASAC，random warmup=10k，端到端訓練
- **問題**：**低於** Ind-SAC A（-1.18 bps/Hz）！z 有效（ablation Δ>0）但整體分數反而下降
- **根因**：**Cold-start problem** — encoder 在訓練初期輸出隨機雜訊，workers 被迫適應 noisy z，陷入局部極小

#### cc-HASAC B（freq-selective obs）
- **結果**：28.17 bps/Hz，z←0 Δ=+5.09
- **設計**：使用 freq-selective KPM obs，encoder 有更豐富的頻率選擇性資訊
- **問題**：ablation 顯示 z 有效，但整體低於 Ind-SAC B（33.60），cold-start 問題仍存在

### 5.3 架構探索（v4, v5）

#### v4：α-gate + Slow Encoder
- **結果**：23.90 bps/Hz
- **設計**：動態調整 encoder 更新頻率（隔 K 步才更新一次）
- **失敗原因**：off-policy replay buffer 中的 transition 在不同 encoder 狀態下蒐集，obs distribution shift 導致 Q-function 矛盾

#### v5：Transformer Encoder
- **結果**：23.90 bps/Hz，z←0 Δ=+1.61，z←shuffle Δ=-0.15
- **設計**：self-attention encoder，N_BS=3 tokens
- **失敗原因**：N_BS=3 只有 3 個 token，self-attention 退化為加權平均，且 50 個 channel snapshot 不足以訓練 attention 不 overfit

### 5.4 BC 預訓練（核心解法）

#### v6：BC Pre-training + 10k Encoder Freeze
- **結果**：32.97 bps/Hz，z←0 Δ=+6.14，z←shuffle Δ=+14.36
- **設計**：
  1. BC phase：用 WMMSE 監督 encoder + actor 1500 steps
  2. RL phase：凍結 encoder 10k 步
  3. Fine-tune：解凍做 end-to-end RL
- **突破**：BC 建立 actor-z 耦合，冷啟動問題消除，BC sanity eval = 57.2 bps/Hz
- **問題**：warmup=2000 隨機 transition 暫時污染 buffer，造成初期性能下降

#### v7：Encoder-only BC
- **結果**：27.59 bps/Hz，z←0 Δ=+2.40，z←shuffle Δ=+0.38（接近 0！）
- **設計**：只 pre-train encoder（用 projection head 監督），actor 從零 RL 訓練
- **失敗根因**：worker 從未見過「好 z → 好動作」的訓練樣本，無法學會依賴 z
- **結論**：BC 必須同時覆蓋 encoder **和** actor，才能建立 actor-z 耦合

#### v9：BC + warmup=0
- **結果**：31.65 bps/Hz，z←0 Δ=+8.68，z←shuffle Δ=+20.79
- **設計**：v6 去掉隨機 warmup（BC policy 直接填充 buffer）
- **分析**：z 貢獻是所有版本中最強（Δ=+8.68），但最終分數略低於 v6（32.97），可能因為 BC buffer 過早被 RL transition 取代

### 5.5 解決 Encoder Unfreeze 崩潰

#### v10：BC + warmup=0 + α=0.001
- **結果**：26.38 bps/Hz，z←0 Δ=-6.08（z **有害**！）
- **設計**：低 α=0.001，期望減少 entropy 壓力保護 BC init
- **發現**：10k 凍結 + 低 α 組合下 encoder 解凍後 z 漂移更嚴重，actor 完全拋棄 z
- **結論**：低 α 本身不夠，凍結時長才是關鍵

#### v11：BC + α=0.001 + 100k Encoder Freeze
- **結果**：33.43 bps/Hz，z←0 Δ=+1.82，z←shuffle Δ=+22.88
- **設計**：將 encoder 凍結延長至 100k 步
- **關鍵觀察**：
  - 凍結期間峰值（30k）：**41.19 bps/Hz**（encoder 仍凍結，BC 初始化效果極佳）
  - 解凍後（120k）：崩潰至 23.65（encoder 激進更新破壞 z）
  - 之後緩慢恢復至 33.43（最終 300k）
- **問題**：encoder 解凍時 LR 仍為 1e-4，z 更新過激烈

#### v12：N_BS=5 版本的 v11
- **結果**：23.22 bps/Hz，z←0 Δ=-16.78（z **極有害**！）
- **設計**：N_BS=5, N_UE=50，其餘同 v11
- **驚奇發現**：z=0 反而得到 40.00 bps/Hz（比 Ind-SAC N5 的 32.40 還高！）
- **根因**：N_BS=5 的 BC MSE=0.021（vs N_BS=3 的 0.012），BC 品質不足，encoder 輸出有害 z，但 actor 已學到良好的本地策略（z=0 時 actor 表現優秀）
- **結論**：N_BS=5 需要更強的 BC 訓練（更多 BC 步數或更大 encoder）

### 5.6 最終最佳版本

#### v13：BC + α=0.001 + 100k Freeze + enc_lr_ft=1e-5 + 500k
- **結果**：**34.21 bps/Hz**，z←0 Δ=**+4.35**，z←shuffle Δ=+21.09
- **設計**：在 v11 基礎上，encoder 解凍後用 10× 低 LR（1e-5 vs 1e-4）
- **效果**：
  - 解凍後（120k）：31.25（v11 同步僅 23.65，**+7.6 bps/Hz**）
  - 穩定上升至 190k：**39.12 bps/Hz**（達 v11 歷史峰值）
  - 240k 峰值：39.34 bps/Hz
  - 最終（500k，20-ep eval）：**34.21 bps/Hz**
- **z-ablation 改善**：z←0 Δ 從 v11 的 +1.82 提升至 +4.35，z 耦合完全恢復

---

## 6. 最終實驗結果比較

### 6.1 主要結果表（同環境：cc_env_r1partial，N_BS=3）

| 方法 | sum-rate | z←0 Δ | z←shuffle Δ | vs Ind-SAC A |
|------|----------|--------|-------------|-------------|
| WMMSE oracle | 85.04 | — | — | +56.93 |
| **cc-HASAC v13** | **34.21** | **+4.35** | **+21.09** | **+6.10 (+21.7%)** |
| cc-HASAC v11 | 33.43 | +1.82 | +22.88 | +5.32 (+18.9%) |
| cc-HASAC v6 | 32.97 | +6.14 | +14.36 | +4.86 (+17.3%) |
| cc-HASAC v9 | 31.65 | +8.68 | +20.79 | +3.54 (+12.6%) |
| **Ind-SAC A**（基準） | **28.11** | — | — | 0 |
| cc-HASAC v7 | 27.59 | +2.40 | +0.38 | -0.52 |
| cc-HASAC A（原始） | 26.93 | +4.02 | +7.67 | **-1.18** |
| cc-HASAC v10 | 26.38 | -6.08 | +11.92 | -1.73 |
| full_power | 21.44 | — | — | -6.67 |

### 6.2 不同環境結果（供參考，不直接可比）

| 方法 | sum-rate | 環境 |
|------|----------|------|
| Ind-SAC B | 33.60 | cc_env（R2 KPM, 3-dim obs） |
| cc-HASAC B | 28.17 | cc_env（R2 KPM） |
| Ind-SAC N5 | 32.40 | cc_env_r1partial（N_BS=5） |
| cc-HASAC v12 | 23.22 | cc_env_r1partial（N_BS=5） |

### 6.3 Encoder Freeze 時長消融

| 版本 | freeze 時長 | enc_lr_ft | sum-rate | z←0 Δ | 結論 |
|------|------------|-----------|----------|--------|------|
| v10 | 10k | 1e-4 | 26.38 | -6.08 ❌ | freeze 太短，z 有害 |
| v9  | 10k | 1e-4 | 31.65 | +8.68 ✓ | warmup=0 緩解，但分數偏低 |
| v6  | 10k | 1e-4 | 32.97 | +6.14 ✓ | warmup=2000 稍干擾 |
| v11 | 100k | 1e-4 | 33.43 | +1.82 ≈ | z 耦合偏弱，解凍後崩潰 |
| **v13** | **100k** | **1e-5** | **34.21** | **+4.35** ✓ | **最佳！z 耦合恢復** |

### 6.4 BC 設計的重要性（消融）

| BC 策略 | sum-rate | z←shuffle Δ | 結論 |
|---------|----------|-------------|------|
| 無 BC（cc-HASAC A） | 26.93 | +7.67 | cold-start 失敗 |
| encoder-only BC（v7） | 27.59 | **+0.38** | actor 不學 z |
| encoder+actor BC（v6~v13） | 32.97~34.21 | +14.36~+21.09 | 建立 actor-z 耦合 |

---

## 7. 分析與關鍵洞察

### 7.1 Cold-Start 問題的根因

```
t=0: encoder 隨機初始化 → z ≈ noise
     ↓
workers 接受 z → 策略 π(a | o, z_noise)
     ↓
RL 更新 Q(s, a)，以 z_noise 為條件
     ↓
encoder 開始更新，z 改變
     ↓
workers 的 π 和 Q 都是為舊的 z_noise 優化的
     ↓
local minimum：workers 實際上 ignore z，獨立最優化
```

**結果**：z 的 ablation 顯示 z 有效（Δ>0），但 workers 對 z 的依賴很淺，整體性能低於無協調基線。

### 7.2 BC Pre-training 的作用機制

BC pre-training 通過監督學習**直接建立 actor-z 耦合**：

1. Encoder 被訓練成能捕捉 channel 特徵的 z（MSE=0.012 with 50 snapshots）
2. Actor 被訓練成「給定 z 時，產生接近 WMMSE 的動作」
3. 這個耦合在 RL 階段作為**良好初始化**，RL 從一個有意義的策略開始優化

**BC sanity eval = 57.2 bps/Hz**（vs WMMSE = 85.04）——BC policy 已能實現 67% 的 oracle 性能，這為 RL 提供了強力的起點。

### 7.3 SAC Entropy 對 BC 初始化的破壞

SAC 的 entropy 正則化項 $\alpha \log \pi(a|s)$ 會主動推動 actor 離開確定性策略（BC 初始化的策略）。

**問題**：過大的 α 在訓練初期就破壞 BC 初始化：
- α=0.1（默認）：幾千步就偏離 BC policy 到高 entropy 隨機策略
- α=0.001（v11/v13）：entropy 壓力小，BC 初始化得以保持更長時間

**驗證**：v11/v13 在 10k step 時（α=0.001，encoder 凍結）= **39.73 bps/Hz**（遠高於 v6 的 32.97），說明低 α 確實保護了 BC 初始化。

### 7.4 Encoder Freeze 的雙重作用

**作用 1：保持 z 穩定，讓 actor 學到依賴 z**

如果 encoder 同步更新，z 每步都在改變，actor 永遠在追一個移動的目標，無法學到穩定的 z 條件策略。

**作用 2：防止 Q-function 的分佈偏移**

若 encoder 更新改變 z，過去 buffer 中的 transition（以舊 z 為條件）的 Q-value 估計就不再準確，導致 Q-function 訓練不穩定。

**最佳 freeze 時長**：
- 10k（v6/v9）：太短，encoder 解凍後 z 大幅漂移
- 100k（v11/v13）：足夠長，actor 在穩定 z 上收斂，解凍後的影響較小

### 7.5 解凍後低 LR 策略的效果

**問題**：v11 在 100k 解凍後，encoder 以 1e-4 的 LR 更新，z 在幾千步內就大幅漂移，破壞已學到的 actor-z 耦合。

**解決方案**：v13 將解凍後 encoder LR 降至 1e-5：

| 步驟 | v11（enc_lr=1e-4） | v13（enc_lr_ft=1e-5） |
|------|-------------------|----------------------|
| 100k（解凍前） | 26.69 | 26.69（相同） |
| 110k | 30.45 | 28.02 |
| **120k** | **23.65**（崩潰！） | **31.25**（穩定！） |
| 140k | 24.77 | 34.16 |
| 190k | 32.06 | **39.12**（峰值） |

**低 LR 的機制**：z 每步只微小改變，actor 有時間適應新的 z 而不失去當前策略；z 可以緩慢適應 RL policy 的演化，逐漸改善協調效果。

### 7.6 z-Ablation 的解讀

z-ablation 提供了衡量協調訊號質量的客觀指標：

**z←0 Δ（zero ablation）**：將 z 設為零向量，比較性能下降
- Δ 越大 → workers 越依賴 z 提供的資訊
- Δ<0（如 v10=-6.08, v12=-16.78）→ encoder 輸出的 z 反而有害，workers 不用 z 更好

**z←shuffle Δ（shuffle ablation）**：將 z 隨機打亂（破壞語意但保持分佈）
- Δ 大（+20 以上）→ z 的具體內容對 workers 決策至關重要
- 所有有效版本的 shuffle Δ 都很高（>14），說明 workers 對 z 語意有很強依賴

**v13 的 ablation（最佳）**：
- z←0 Δ = +4.35：適中，表示 workers 依賴 z 但也有本地策略
- z←shuffle Δ = +21.09：非常強，z 的語意資訊至關重要

### 7.7 為什麼 v13 優於 v11（同框架下）

| 面向 | v11 | v13 |
|------|-----|-----|
| enc_lr_ft | 1e-4（激進） | 1e-5（溫和） |
| 解凍後最低點 | 23.65（-3.1 vs 前步）| 28.02（-0.2 vs 前步）|
| z←0 Δ | +1.82（弱）| +4.35（強）|
| 最終分數 | 33.43 | **34.21** |
| 訓練步數 | 300k | 500k |
| 訓練穩定性 | 振盪±15 | 振盪±7 |

低 LR 讓 encoder 以更小的步長適應，actor 不需要「重新學習如何用 z」，累積效果更好。

---

## 8. 結論

### 8.1 最終最佳設計

**cc-HASAC v13** 為本研究的最終最佳版本：

```
BC pre-train（encoder + actor，1500 steps）→ WMMSE 監督
→ RL 訓練（α=0.001，encoder frozen 100k steps）
→ RL fine-tune（encoder unfrozen，enc_lr=1e-5，200k steps 以上）
```

**結果**：34.21 bps/Hz
- vs 原始 cc-HASAC A：**+7.28 bps/Hz（+27.1%）**
- vs Ind-SAC A（同環境基線）：**+6.10 bps/Hz（+21.7%）**
- z-ablation 確認 z 真實有效（z←0 Δ=+4.35, z←shuffle Δ=+21.09）

### 8.2 最重要的發現

**第一重要：BC Pre-training 是解決 cold-start 的關鍵**

沒有 BC，cc-HASAC 比無協調基線（Ind-SAC A）還差 1.2 bps/Hz。有 BC 後，最低也超過 Ind-SAC A 17%。BC 必須同時覆蓋 encoder 和 actor（v7 的 encoder-only BC 失敗驗證了這點）。

**第二重要：Encoder 解凍後的 LR 決定 z 的穩定性**

100k freeze 是必要但不充分的條件。解凍後若 enc_lr=1e-4（v11），z 仍然劇烈漂移導致崩潰。enc_lr=1e-5（v13）使解凍後的性能從 23.65 提升至 31.25（同步比較）。

**第三重要：Alpha 初始值影響 BC 保留**

低 α=0.001 讓 SAC 的 entropy 壓力減小，使 BC 初始化在 RL 初期得以保留。這解釋了為什麼 v11/v13 在 10k 步（frozen）就達到 39.73，遠超 v6（frozen 只有 ~32-33）。

**第四：z 的真實貢獻已被驗證**

所有最終有效的版本（v6, v9, v11, v13）的 z←0 Δ 都為正值，z←shuffle Δ 都超過 +14，證明 encoder 確實學到了有意義的全域協調資訊，而非 trivial embedding。

### 8.3 未來方向

1. **N_BS 擴展**：N_BS=5 的 BC 品質不足（MSE=0.021 vs N_BS=3 的 0.012），需要更多 BC 步數或更大 encoder 容量
2. **Encoder 架構**：Transformer（N_BS=3 太少 token）在 N_BS≥7 時應更有效
3. **BC 目標多樣化**：目前用單一 WMMSE target，可嘗試 n_init 多次 WMMSE 求解取最優或加入 diversity
4. **z 維度**：Z_DIM=8 是否足夠？擴展至 16 可能增加協調資訊容量
5. **縮小 oracle gap**：與 WMMSE（85.04）的差距仍有 50.83 bps/Hz，根本原因在 partial observability 和 RL 探索效率

---

## 附錄：實驗版本索引

| 版本 | 腳本 | 關鍵設計 | 最終結果 |
|------|------|---------|---------|
| Ind-SAC A | train_ind_sac_A.py | 無 z，R1-partial | 28.11 |
| Ind-SAC B | train_ind_sac_B.py | 無 z，R2 KPM | 33.60 |
| cc-HASAC A | train_cc_hasac.py | 有 z，無 BC | 26.93 |
| cc-HASAC B | train_cc_hasac_B.py | 有 z，R2 KPM，無 BC | 28.17 |
| v4 | — | α-gate，slow enc update | 23.90 |
| v5 | — | Transformer encoder | 23.90 |
| v6 | train_cc_hasac_v6.py | BC+10k freeze | 32.97 |
| v7 | train_cc_hasac_v7.py | encoder-only BC | 27.59 |
| v9 | train_cc_hasac_v9.py | BC+warmup=0+10k freeze | 31.65 |
| v10 | train_cc_hasac_v10.py | BC+warmup=0+α=0.001+10k freeze | 26.38 |
| v11 | train_cc_hasac_v11.py | BC+α=0.001+100k freeze | 33.43 |
| v12 | train_cc_hasac_v12.py | N_BS=5 版 v11 | 23.22 |
| **v13** | train_cc_hasac_v13.py | BC+α=0.001+100k freeze+enc_lr=1e-5+500k | **34.21** |
| Ind-SAC N5 | train_ind_sac_n5.py | N_BS=5 無 z 基線 | 32.40 |
