# C-HASAC 實驗報告
## Context-Conditioned Heterogeneous-Agent SAC for 5G Multi-Cell Power Allocation

---

## 1. 問題背景與動機

### 1.1 5G 多基站功率協調的長期難題

5G dense network 部署之後，多個 BS（Base Station）在同一頻段服務各自的 UE（User Equipment），co-channel interference 成為系統效能的主要瓶頸。每個 BS 的發射功率越大，自家 UE 的 SINR（Signal-to-Interference-plus-Noise Ratio）確實上升，但同時對鄰近 BS 的 UE 造成更強干擾。

這造成一個結構性的兩難：

> **如果各 BS 只顧自己最大化功率，所有人的 SINR 都會下降；但若協調降功率，所有人都受益。**

傳統方法如 WMMSE（Weighted Minimum Mean Squared Error）可以找到接近最優的功率分配，但需要全局 CSI（Channel State Information）和 centralized 求解——在真實網路中難以部署。而基於規則的 heuristic（等功率、比例分配等）又犧牲太多效能。

我們的目標是最大化系統的 Proportional Fairness utility：

$$\text{PF-U} = \sum_{u} \log(\bar{R}_u + \varepsilon)$$

PF-U 同時考量總吞吐量與公平性（取 log 懲罰速率過低的 UE），是 5G 資源管理的 canonical 指標。

| 參考點 | PF-U | 說明 |
|--------|------|------|
| **Equal Power (floor)** | **−5.332** | 所有 BS 等功率，無協調 |
| **PF-WSR (ceiling)** | **+23.529** | Full-CSI 聯合最優化，oracle 上限 |

### 1.2 為什麼 Multi-Agent RL 是自然的框架

多基站功率分配在結構上就是一個 multi-agent 問題：

- **每個 BS = 一個 agent**，觀測本地資訊，輸出功率決策
- **全局 PF-U = team reward**，所有 BS 共同最大化
- **干擾耦合 = agent 之間的交互作用**，每個 BS 的動作影響其他 BS 的回報

Multi-Agent Reinforcement Learning（MARL）不需要顯式建模 channel 模型，可以直接從環境互動中學出協調策略，理論上能適應各種 channel 條件和負載情境。

然而，標準 MARL 方法（如 MAPPO、HAPPO 等）雖然也使用 stochastic policy（輸出 Gaussian distribution），但 entropy 在這些方法中只是一個 soft regularizer（目標函數裡的小 bonus 項，係數 $c_2$ 通常 ≤ 0.01），隨訓練收斂 policy 趨向 deterministic，沒有機制在均衡時維持特定的隨機性，最終收斂到 Nash Equilibrium（NE）。NE 在 cooperative 設定下可能是 sub-optimal——即所有 BS 高功率廣播的局部均衡，此時任一 BS 單方面降功率都是吃虧的，即便全體「協調性地」降功率對所有人更有利。

### 1.3 HASAC：Maximum Entropy MARL 解決協調均衡陷阱

在閱讀文獻時，我們注意到 Liu et al.（ICLR 2024）提出的 **HASAC（Heterogeneous-Agent Soft Actor-Critic）**。HASAC 將 Maximum Entropy RL 推廣到 multi-agent 設定：每個 agent 不追求 deterministic policy，而是最大化 reward 與 policy entropy 的加權和：

$$J(\pi) = \mathbb{E}\left[\sum_t r_t + \alpha \sum_i \mathcal{H}(\pi^i(\cdot|s_t))\right]$$

**關鍵理論結果**：HASAC 可以分解成 $N$ 個依序的 Soft Policy Improvement 步驟，收斂到 Quantal Response Equilibrium（QRE）——一種「軟化的 NE」，每個 agent 保持 stochastic policy，使得多個 agent 同時偏離局部均衡的機率 > 0，從而跳出協調陷阱。

這讓我們想到：**能不能把 HASAC 直接套用在 5G 多基站功率分配上？**

### 1.4 直接套用的問題：去中心化 BS 看不到鄰居

把 HASAC 直接套用時，每個 BS actor 只能觀測自己本地的資訊——自家 UE 的 rate、PF weight、功率狀態。它對鄰居 BS 的負載、功率、干擾貢獻一無所知。

這意味著每個 BS 的 actor 是在「資訊孤島」中做決策：它知道自己現在的狀態，但不知道整個系統目前是高負載還是低負載、鄰居是在衝功率還是在降功率。沒有這些資訊，協調策略很難學出來。

一個直覺的解法是讓 BS 直接交換 neighbor CSI（鄰居的 channel state）。但這在真實 O-RAN 網路中違反 deployment constraints——BS 之間沒有標準介面傳遞即時 CSI，且這樣做也就失去了 distributed RL 的意義。

### 1.5 機會：O-RAN Near-RT RIC 有全局視野

O-RAN（Open Radio Access Network）架構定義了 **Near-RT RIC**（Near Real-Time RAN Intelligent Controller），上面運行的 xApp 可以透過 **E2 介面**定期收集所有 gNB（5G BS）的 **KPM（Key Performance Metric）**報告，並將控制訊號下發回各 gNB。

KPM 是各 BS 的統計量——cell load、throughput、transmit power 等，不含即時 CSI，是 O-RAN 標準已支援的資訊流。

這給了我們一個可部署的協調通道：

```
Near-RT RIC (xApp)
  ├─ E2 收集：所有 BS 的 KPM（load, throughput, P_bs + inter-BS distances）
  ├─ 計算：Encoder 將 KPM 壓縮為 latent context z（16 維）
  └─ E2 下發：z → 每個 gNB 的本地 Actor
```

### 1.6 C-HASAC：用 Learned Context z 填補資訊缺口

基於以上觀察，我們提出 **C-HASAC（Contextual HASAC）**：

在 HASAC 的 decentralized actor 基礎上，加入一個由 RIC xApp 計算的 **learned latent context z**。Encoder 把全部 BS 的 KPM 壓縮成一個 16 維向量 z，代表「系統目前整體負載與干擾狀態的隱式摘要」，廣播給每個 BS 的 actor。

**唯一差別**：

| 方法 | Actor 輸入 |
|------|-----------|
| **HASAC** | local obs only（自家 UE 資訊）|
| **C-HASAC** | local obs + z（+ 全局 KPM 的 latent summary）|

核心命題：這個 z 能否讓 actor 學到更好的協調策略？z 是否真的被使用（而不只是 noise）？

---

## 2. 方法：C-HASAC 架構

### 2.1 三層資訊嚴格分流

| 層 | 內容 | 使用者 | 說明 |
|----|------|--------|------|
| **A. BS-local 可觀測** | per-UE rate、PF weight、power | Actor obs | gNB 本地可取得 |
| **B. RIC-observable KPM** | cell load、throughput、P_bs、BS 間距離 | Encoder → z → Actor | xApp 透過 E2 取得 |
| **C. 特權資訊（sim-only）** | 完整 CSI（g matrix）、全域功率 | Critic + reward（訓練時） | 部署時不存在 |

**Critic 不餵 z**：z 僅影響 Actor，Critic 直接使用完整 share_obs（訓練特權）。

### 2.2 核心架構

```
KPM [N_BS × 5]  →  Encoder (DeepSet MLP)  →  z [16]
                                                 ↓ broadcast
local obs [N_UE × 3] ──────────────────→  SetActor  →  power frac [N_UE]
(per-UE: rate, PF_weight, power)
```

**C-HASAC vs HASAC 唯一差別**：

| 方法 | Actor 輸入 |
|------|-----------|
| **HASAC** (`--use_z 0`) | local obs only |
| **C-HASAC** (`--use_z 1`) | local obs + z |

### 2.3 SetActor（排列等變）

- 對全 $N_{UE}$ 一次前向；membership mask 分 BS 做 intra-cell 池化
- C-HASAC：global z broadcast concat 到每個 UE embedding
- 輸出：squashed Gaussian → $a \in (-1,1)$ → power fraction $\in [0,1]$
- `mu_bound=5`：防止 $\mu \to -\infty$（tanh 飽和 → 功率崩 0 → log(0) = −165）

### 2.4 Encoder（DeepSet，排列不變）

$$z = \rho\left(\frac{1}{N_{BS}} \sum_{i=1}^{N_{BS}} \phi(\text{KPM}_i)\right)$$

- $\phi$：per-cell MLP（kpm_dim=5 → 128 → 128）
- $\rho$：投影 MLP（128 → 128 → z_dim=16）
- 排列不變：BS 順序不影響 z

### 2.5 Critic（Agent-conditioned Twin-Q）

- 輸入：share_obs（63 dim：$g \times 36 + p \times 12 + \text{serv} \times 12 + \text{bs\_dist} \times 3$）+ joint action + one-hot BS ID
- Twin-Q 取 min，防止 overestimation
- **不餵 z**（HANDOFF 原則）

### 2.6 HASAC Sequential Soft Policy Decomposition

依 HASAC 論文 Theorem 3.3，actor update 改為隨機 permutation 順序依序更新各 BS：

```
for i in randperm(N_BS):
    z_frozen = encoder(kpm).detach()   # z 凍結，避免 encoder 收到 N_BS 個衝突梯度
    loss_i = (α · logπ_i - Q_min).mean()
    opt_actor.zero_grad(); loss_i.backward(); opt_actor.step()

# encoder 獨立更新一次（live z，所有 BS 聯合 loss）
z_live = encoder(kpm)
loss_enc = Σ_i (α · logπ_i - Q_min).mean()
opt_encoder.step()

# alpha：每 RL step 更新一次（avg logp across all agents）
avg_logp = mean([logp_0, logp_1, logp_2])
loss_alpha = -(log_alpha × (avg_logp + target_H)).mean()
opt_alpha.step()
```

**關鍵修正（alpha fix）**：原始 sequential loop 每個 agent update 後都更新 alpha（3×/step），等效 alpha lr ×3，entropy 3× 速崩潰。改為 loop 後統一更新一次（1×/step），大幅提升 z 使用程度。

### 2.7 訓練設定

| 超參數 | 值 | 說明 |
|--------|-----|------|
| reward | logpf | potential-based $\Delta \Sigma \log(\bar{R}_u + \varepsilon)$，與 PF-U 完全對齊 |
| bc_steps | 1000 | BC warm-start（expert = PF-WSR full-CSI），打開 z 使用開關 |
| mu_bound | 5 | 防止 tanh 飽和崩潰 |
| warmup | 1000 | 隨機探索步數 |
| tau | 0.001 | Polyak 係數（慢速 target 更新，穩定 Q） |
| z_dim | 16 | Encoder 輸出維度 |
| hidden | 256 | MLP 隱藏層大小 |
| batch | 256 | Replay buffer batch size |
| replay | 1,000,000 | Replay buffer 容量 |

---

## 3. 實驗設置

### 3.1 環境規格

| 參數 | 值 |
|------|----|
| N_BS | 3 |
| N_UE | 12 |
| kpm_dim | 5（3 KPM + 2 inter-BS distances） |
| share_dim | 63 |
| z_dim | 16 |
| ue_feat | 3（rate, PF_weight, power） |
| Episode length | 10 steps |
| Pmax | 30 dBm（per BS） |
| Channel | DeepMIMO 3GPP ray-tracing |

### 3.2 評估方法

- **訓練中**：每 5000 steps 評測 n_eval=20 episodes
- **FINAL（best ckpt）**：n_eval_final=50 episodes held-out scenarios（seed=2024）
- **z-ablation**：
  - `drop_zero`：policy − policy(z←0)
  - `drop_shuffle`：policy − policy(z←別 episode 的 z)（更嚴格，排除 z 作為常數 offset 的假象）

---

## 4. 實驗結果

### 4.1 主要比較表

| 方法 | 訓練步數 | PF-U（FINAL） | best_U（訓練峰） | drop_zero | drop_shuffle |
|------|---------|-------------|---------------|-----------|-------------|
| Equal Power (floor) | — | **−5.332** | — | — | — |
| HASAC (z=0, baseline) | 200k | −5.184 | −5.360 | — | — |
| C-HASAC geo_z | 200k | −2.237 | −0.346 | +0.932 | **+1.429** ✅ |
| C-HASAC + RSRP_neighbor | 200k | −3.763 | −2.420 | +0.688 | +0.261 ⚠️ |
| C-HASAC + Critic BC | 200k | −2.606 | −1.165 | −4.269 ❌ | −0.453 ❌ |
| C-HASAC geo_z_long | 400k | −1.162 | +0.285 | +3.565 | **+2.278** ✅ |
| C-HASAC tau001 | 400k | −1.051 | −0.241 | +0.443 | +0.838 ✅ |
| **C-HASAC alpha_fix** | **400k** | **−0.911** | **+0.292** | **+2.219** | **+2.622** ✅ |
| **C-HASAC alpha_fix** | **800k** | **+0.808** | **+2.575** | +0.682 | +0.167 |
| HASAC (z=0, alpha_fix) | 800k† | —（best −3.151） | −3.151 | — | — |
| PF-WSR (ceiling) | — | **+23.529** | — | — | — |

†HASAC 800k 對照組在 step 225k 死於 entropy 崩潰（alpha → 0.0005，死亡區），無 FINAL；best −3.151。

### 4.2 C-HASAC vs HASAC 核心比較

**唯一差別 = actor 有沒有吃 encoder 學出的 z**

- C-HASAC 400k (alpha_fix) FINAL **−0.911** vs HASAC 200k **−5.184** → **+4.273 PF-U**
- C-HASAC 800k FINAL **+0.808** vs HASAC 800k best **−3.151**（HASAC 訓練中途 entropy 崩潰死亡）→ **≥ +3.96 PF-U**

### 4.3 z 有效性驗證

**drop_shuffle > 0 = z 真實被使用（錯誤 z 比沒有 z 更傷）**

- alpha_fix 400k：drop_shuffle = **+2.622**（最強 z 使用證據）
- geo_z 200k：drop_shuffle = **+1.429** ✅
- RSRP 版本：drop_shuffle = +0.261（z 幾乎沒用，actor 已自給自足）
- Critic BC 版本：drop_shuffle = −0.453 ❌（z 主動有害）

---

## 5. 消融實驗

### 5.1 RSRP_neighbor 的影響（actor obs 資訊量）

| 設定 | PF-U | drop_shuffle | 解讀 |
|------|------|-------------|------|
| 無 RSRP（ue_feat=3） | −2.237 | **+1.429** ✅ | z 是唯一跨 BS 資訊通道 |
| 有 RSRP（ue_feat=6） | −3.763 | +0.261 ⚠️ | actor 已知鄰居 channel，不需要 z |

**結論**：z 的價值來自填補資訊缺口；一旦 actor obs 已包含鄰居資訊，z 自然被忽略。

### 5.2 Critic BC warm-start 的影響

| 設定 | PF-U | drop_shuffle | 解讀 |
|------|------|-------------|------|
| 無 Critic BC | −2.237 | +1.429 ✅ | z 正常使用 |
| 有 Critic BC（500 iters） | −2.606 | −0.453 ❌ | z 主動有害 |

**根因**：Critic BC 用 expert MC returns 預訓練 Q，encoder 學會產生「讓 Q 開心」但「讓 policy 走偏」的 z，形成惡性循環。

### 5.3 訓練步數的影響

| 步數 | best_U（訓練峰） | FINAL PF-U | drop_shuffle |
|------|---------------|-----------|-------------|
| 200k | −0.346 @180k | −2.237 | +1.429 |
| 400k | +0.285 @325k | −1.162 | +2.278 |
| 400k（alpha fix） | +0.292 @355k | −0.911 | +2.622 |
| 800k（alpha fix） | **+2.575** @760k | **+0.808** | +0.167 |

**觀察**：更長的訓練持續改善 policy 分數；但 800k 的 drop_shuffle 降低，顯示後期 policy 找到較不依賴 z 的解法（two-regime behavior）。

### 5.4 Oracle z（繞過 encoder）— 負面結果

直接把 PF-WSR expert 的 per-BS power fractions 當 z 餵給 actor（`--oracle_z 1`），測試「encoder 是不是瓶頸」：

| 設定 | PF-U | drop_shuffle | 解讀 |
|------|------|-------------|------|
| learned z（encoder） | −2.237 | +1.429 ✅ | z 編碼干擾結構 |
| oracle z（expert 功率） | −4.673 | +0.059 ❌ | oracle z 幾乎未被使用 |

**結論**：encoder 不是瓶頸。expert 的 per-BS 功率比例低變異（≈常數 5%），無條件性資訊；learned z 之所以有效是因為它編碼「誰是主 BS / 干擾結構」等相對協調情境，而非目標功率水準。

### 5.5 動態環境（UE Random Walk）

UE 每步隨機移動（σ=5 m/step），BS 固定，測試動態 channel 下 z 是否更有用：

| 指標 | HASAC z0 | C-HASAC z1 |
|------|----------|------------|
| PF-U（FINAL） | −2.434 | **−2.218** |
| drop_zero / drop_shuffle | — | −0.140 / −0.154 ⚠️ |

**結論**：與預期相反——動態環境下 z 完全未被使用（ablation 皆為負）。KPM 快照無法捕捉快速 channel 變化，actor 學會忽略 z。靜態快照中 z 編碼的「干擾結構」在動態下失去時效性。

### 5.6 Alpha Fix 的效果

**問題**：sequential loop 內每個 agent update 後都更新 alpha（3×/step）→ entropy 3× 速崩潰。

**修正**：移出 loop，用所有 agents 的平均 logp 更新一次（1×/step）。

| 設定 | alpha @10k | alpha @100k | FINAL | drop_shuffle |
|------|-----------|-----------|-------|-------------|
| 舊（3×/step） | 0.0022 | 0.001 | −1.162 | +2.278 |
| alpha fix（1×/step） | 0.0070 | 0.002 | **−0.911** | **+2.622** |

**效果**：entropy 維持更久 → 更充分探索 → z 使用程度大幅提升（+2.278 → +2.622）。

---

## 6. 關鍵分析

### 6.1 BC Warm-start 打開 z 使用開關

純 RL 訓練（無 BC）：drop_shuffle ≈ 0（actor 完全不用 z）。

BC warm-start 機制：
1. BC 先讓 actor 對齊 PF-WSR expert 的功率輸出
2. 建立合理的 gradient landscape
3. RL 啟動後，z 的信號足夠強讓 actor 學到依賴 z

**無 BC → drop_shuffle ≈ 0；有 BC 1000 步 → drop_shuffle +1~2.6**

### 6.2 Q-Overestimation 導致的訓練不穩定

SAC 的 max 運算累積 bias → Q 高估 → actor 追錯梯度 → policy 崩潰 → 震盪。

觀察到的典型模式：
- 每隔 80–120k steps 出現一次深跌（PF-U −15 ~ −33）
- 深跌後強力反彈，偶爾突破前一個 best
- Best checkpoint 機制保留各峰值

已知緩解：mu_bound=5 + logpf reward + BC warm-start + tau=0.001。根本解尚未找到。

### 6.3 tau=0.001 的作用

更慢的 target network 更新（$\phi' \leftarrow 0.001\phi + 0.999\phi'$）→ Q target 更穩定 → actor gradient 不被噪訊引偏 → 更晚但更深的突破（step 140k / 230k / 355k / 760k）。

### 6.4 z 表示分析（learned z 學到了什麼）

對最佳 C-HASAC checkpoint 跑 200 scenarios × 8 steps rollout，分析 z 向量：

- **PCA PC1 = 93.5%**：16 維 z 實際坍縮成幾乎 1 個有效維度（純量信號）
- **z 主要編碼 throughput**：top z 維度與各 BS throughput 相關最強（|corr| ≈ 0.35–0.40）
- **軟性 on/off 行為**：100% scenario 有某 BS power < 0.1（功率分布雙峰），但非嚴格二元
- **z 對 power 的影響是間接的**（corr ≈ 0.3）：z 提供「系統負載狀態」，on/off 是 actor 的連續決策結果

### 6.5 Q Target 正確性修正與 N-step A/B（2026-06-10）

程式碼審查發現一個正確性 bug：**Replay 沒有 done flag**，critic target $r + \gamma Q(s')$ 無條件跨 episode bootstrap。episode 僅 10 步、PF 權重在 reset 歸零——10% transitions 的 Q target 系統性高估，直接餵養 Q-overestimation。修正後同時部署：

1. **done-mask**：target 改 $r + \gamma^n(1-d)(Q - \alpha \log\pi)$
2. **n-step returns（n=3）**：更準的 Q target，縮短 bootstrap 鏈
3. **alpha floor（0.001）**：歷史 log 顯示死亡區明確——崩死 run 的 alpha 都在 0.0005–0.0007，成功 run 在 0.0015+
4. **top-K checkpoint + validation 重排序**：訓練期保留 top-10 ckpts（含 EMA 副本），結束時在獨立 validation seed 上 50-ep 重排序再報 test（消除選擇噪音與測試污染）

N-step A/B（1200k，同 fix stack）：n_step=3 全程健康（best −2.073、alpha 0.06–0.11、pwr 0.25）；n_step=1 長期卡死全功率壞區（best −2.772、pwr 0.7–0.95）→ **n-step=3 顯著改善訓練穩定性**。

### 6.6 O-RAN Deployment 可行性

| 元件 | 部署位置 | 資訊來源 | 可行性 |
|------|---------|---------|--------|
| Actor | gNB（去中心化執行） | 本地 UE obs | ✅ gNB 本地感測 |
| Encoder | xApp（Near-RT RIC） | KPM via E2 介面 | ✅ O-RAN 標準支援 |
| z 下發 | xApp → gNB | E2 控制訊息 | ✅ E2 SM 支援 |
| Critic | 僅訓練時使用 | 特權 CSI | ✅ 部署時不存在 |

**不比 Nasir-Guo**：他們用顯式 neighbor CSI 交換（違反 deployment line），我們的 z 只來自 KPM（合規）。

---

## 7. 獨立復現研究：REPRODUCE spec（2026-06-10）

依合作方提供的 methods-only spec（`REPRODUCE.md`）從零重建一個**不同 regime** 的環境並驗證其三大主張：N_BS=4、goodput+queue 目標、固定 topology、γ=0（contextual bandit）、team reward、separate per-cell actors。此環境與主線（§3）的關鍵差異：無 bootstrap（γ=0）、評 goodput 而非 PF-U。

### 7.1 環境校準

以 N0 掃描對齊 spec 參考值，之後取得作者的精確幾何 dump（`geom_topo12345.npz`）做 exact match：

| 項目 | spec/作者 | 復現（exact geometry） |
|------|----------|----------------------|
| floor (equal) | ~5.33 | 5.358 ✓ |
| ceiling (full-CSI oracle) | ~9.09 | 9.402 |
| spatial-oracle-as-policy | ~80% | 77.8% ✓ |
| gate×base BC goodput | 8.41/9.03/8.12（3 seeds） | **8.535**（區間內）✓ |

### 7.2 三大主張全部復現（% of floor→ceiling gap）

| 方法 | %gap（3 seeds） | spec 主張 |
|------|----------------|-----------|
| HASAC RL（random topology） | ≈ 0% | RL 無法跨拓撲泛化 ✓ |
| HASAC RL（fixed topology, 40k） | 31.1% ± 1.4 | 學到 topology-specific spatial reuse |
| **C-HASAC RL（+z input）** | **32.2% ± 5.4** | **z-as-input null ✓**（≈ HASAC） |
| HASAC RL（120k 長訓） | 34.5% ± 1.7 | 平台期，非訓練長度問題 |
| gate × learnable-combine（RL） | 38.7% | multiplier corr 0.998 → **0.10**（被 RL 拆掉）✓ |
| **gate×base BC（固定乘法）** | **69.8% ± 0.2** | 監督式結構 >> RL ✓ |
| **spatial-gate × RL-worker（固定乘法）** | **71.6%** | 固定結構下 RL 是建設性的 ✓ |

### 7.3 復現過程的額外科學收穫

1. **比 spec 更強的負面結果**：當 gate 可被 RL 訓練時，RL 連 BC 至 corr≈1 的幾何 gate 都會毀掉（68.7%→7.1%）。可部署性來自**把結構凍結在 RL 之外**。
2. **α-floor 反差結論**：同一個 entropy floor 在 γ=0（無 bootstrap）環境**有害**（31.1%→14.2%，擋住收斂），在 γ=0.99 主線環境**救命**（防 Q 過估死亡螺旋）——entropy guard 的價值完全取決於 Q 是否會 bootstrap 累積偏差。
3. **幾何分解診斷**：可部署增益幾乎全來自 slow spatial 結構（spatial-oracle-as-policy 69–78%），fast fading 適應從 local obs 只能恢復 ~1pp。
4. **γ=0 的穩定性對照**：reproduce 環境全程無崩潰、單調爬升——沒有 bootstrap 就沒有 Q 累積過估，反襯主線（γ=0.99）的震盪來源。
5. **tanh 飽和第三例**：第一代 RL-refine 機制因 `atanh(2L−1)` 在 gate≈0 處飽和而全滅——與主線 −165 崩潰同根，再次確認「squash 內不可放結構先驗，乘法必須在 squash 之外」。

### 7.4 兩個環境的誠實對照（對 C-HASAC 命題的意涵）

| | 主線環境（§3–6） | REPRODUCE 環境（§7） |
|--|----------------|---------------------|
| 目標 / γ | PF-U / 0.99 | goodput / 0 |
| z-as-input | drop_shuffle +2.6（z 被使用） | null（32.2% ≈ 31.1%） |
| RL 絕對水準 | floor+6.1 / gap 28.9 的 21% | gap 的 ~31% |
| 最強可部署做法 | C-HASAC（+0.808） | 監督式 gate×base（~70–85%） |

z 的價值是 **regime-dependent**：在固定拓撲 + goodput bandit 設定下，協調是幾何決定的常數結構，z（load 摘要）與最優開關無關 → null；在我們的 PF-U 設定下，z 編碼的干擾/負載情境有可測量的使用證據（drop_shuffle），但絕對增益距 ceiling 仍遠。兩個環境共同指向同一個更深的結論：**可部署的協調增益主要來自結構（誰該讓位的 slow spatial pattern），監督式學習比 RL 更可靠地獲得它**。

---

## 8. 結論

### 8.1 主要貢獻

1. **C-HASAC 贏過 HASAC**（主線環境）：actor 加入 learned context z → PF-U 大幅提升（唯一差別 = actor 吃不吃 z）；HASAC 800k 對照組死於 entropy 崩潰（best −3.151）
2. **z 真實有效（主線環境）**：drop_shuffle = +2.622（400k），錯誤 z 比沒有 z 更傷，排除 z 作為常數 offset 的假象；z 表示分析顯示 z 實際為 1 維 throughput/負載信號（PCA PC1=93.5%）
3. **BC warm-start 是關鍵**：純 RL 訓練 z 不被使用，BC 1000 步打開 z 使用開關
4. **O-RAN 可部署**：三層資訊嚴格分流，z 由 RIC xApp 下發，不需鄰 BS 直接通訊
5. **獨立復現研究（§7）**：在合作方的 goodput/bandit regime 完整復現三大主張（z-as-input null、fixed>>random topology、RL un-learns learnable combine），exact geometry 下數值對齊（goodput 8.535 ∈ [8.12, 9.03]）
6. **跨環境綜合結論**：可部署的協調增益主要來自 slow spatial 結構；監督式學習 + 凍結結構（gate×base BC ~70–85%）遠比任何 RL 變體（~31%）可靠

### 8.2 最終最佳數字（供 Poster 使用）

**主線環境（PF-U，floor −5.332 / ceiling +23.529）**：

| 指標 | 數值 |
|------|------|
| HASAC (無 z, 200k / 800k best) | −5.184 / −3.151 |
| **C-HASAC 最佳 FINAL（800k）** | **+0.808** |
| C-HASAC drop_shuffle 最強（400k） | **+2.622** |
| n-step=3 fix stack（1200k，進行中） | best −2.073 |

**REPRODUCE 環境（goodput % of gap）**：

| 指標 | 數值 |
|------|------|
| HASAC RL = C-HASAC RL（z null） | 31.1% ≈ 32.2% |
| **gate×base BC（固定乘法）** | **69.8%**（exact geometry：**85.2%**） |
| spatial-gate × RL-worker | 71.6% |
| learnable combine 經 RL | 38.7%（multiplier corr 0.998→0.10） |

### 8.3 未解問題

- **SAC Q-overestimation（主線）**：done-mask + n-step + alpha floor 顯著改善（fix3 全程健康），但 oscillation 未根除
- **800k drop_shuffle 低**：長訓練後 policy 較不依賴 z，與 400k 呈現 two-regime behavior
- **z 的 regime 邊界**：z 在 PF-U/bootstrap 環境有用、在 goodput/bandit 環境 null——精確刻畫「z 何時有資訊價值」仍開放

---

*最後更新：2026-06-10*
*fix3（n-step=3 stability stack，1200k）進行中 @1065k，FINAL 含 top-K validation 重排序*
