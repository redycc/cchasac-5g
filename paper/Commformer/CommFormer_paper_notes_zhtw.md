# CommFormer 深度拆解與評估報告

> **論文**: Learning Multi-Agent Communication from a Graph Modeling Perspective
> **作者**: Shengchao Hu, Li Shen, Ya Zhang, Dacheng Tao
> **單位**: SJTU + Shanghai AI Lab + JD Explore Academy + NTU
> **會議**: ICLR 2024
> **Code**: https://github.com/charleshsc/CommFormer (亦同 jd-opensource/CommFormer)
> **報告語言**: 繁體中文 (zh-TW) — 技術術語保留英文

---

## 1. Overview（論文總覽）

### 論文資訊
- **Title**: Learning Multi-Agent Communication from a Graph Modeling Perspective
- **Authors / Affiliation**: Shengchao Hu (SJTU), Li Shen (corresponding, JD Explore), Ya Zhang (SJTU), Dacheng Tao (NTU)
- **Venue**: ICLR 2024 (poster)

### 一句話摘要
**CommFormer 把「多 agent 之間誰應該跟誰通訊」直接視為一張可學習的有向圖 (learnable directed graph)，透過 Gumbel k-hot 連續鬆弛 + bi-level optimization 與 Transformer encoder/decoder 端到端聯合學習通訊拓樸與 policy，在 bandwidth-constrained 條件下逼近 fully-connected 的上界效能。**

### 研究動機（Motivation）
作者觀察到三個 communication learning in MARL 長期未解的痛點：

1. **Pre-defined topology 變異極大**：Figure 1 顯示在 SMAC `1c3s5z`、`8m_vs_9m` 等任務中，用不同隨機 seed 手動指定通訊圖，勝率方差巨大 — 證明「拓樸結構本身」就是顯著影響因子，不應交給人工經驗。
2. **Full communication 不切實際**：CommNet / TarMAC / MAT 等方法允許所有 agent 互通，但在 wireless 場景中受 bandwidth 與 contention 限制；當 N 變大時 noise 主導 signal，反而傷害合作 (Jiang & Lu, 2018)。
3. **Dynamic scheduling 太重**：ATOC / IC3Net / ToM2C 在 inference 期動態決定誰跟誰講話，需要額外的 gating / scheduling 模組，bandwidth 浪費且難部署到真實無線通訊上。

#### 三大痛點詳解（What Each Pain Point Means）

##### (a) Pre-defined Topology 變異極大 — 手工拓樸的「靠運氣」問題

**直白定義**：在 communication MARL 中，「誰連誰」就是一張**通訊圖（communication topology）**。傳統做法是工程師根據先驗知識（ring、star、fully-connected、k-nearest 等）手動指定這張圖，再把它當常數塞進演算法裡。

**為什麼是問題？**
- Figure 1 的關鍵實驗：**同一個 SMAC 任務、同一個演算法**，只是換不同隨機 seed 來「隨機抽一張 fixed topology」，最終勝率落差可達 30–60%。這意味著「拓樸選擇」本身就是顯著影響因子。
- 但 MARL paper 普遍只寫「我們採用 fully-connected」或「我們用 k-nearest」，**從不分析為什麼這個拓樸最佳** — 結果是 paper 之間的 baseline 比較其實混進了「拓樸運氣」這個 confounder。
- 真實無線網路下根本不存在「自然」的拓樸：5G/6G base station 之間 X2/Xn interface 的可用度受地理位置、回傳容量、政策路由動態影響，**沒有一張「對的」靜態圖可抄**。

**具體後果**：
- 同一演算法搬到新任務需要重新「猜拓樸」，工程迭代成本高。
- Reproducibility 差 — 論文中 work 的 topology 在你的 setup 可能完全失效。
- 「拓樸是 hyperparameter」這件事被整個 communication MARL 圈集體忽視。

##### (b) Full Communication 不切實際 — 「全連通」假設的兩大破口

**直白定義**：「Full / dense communication」指允許**所有 N 個 agent 在每一個時間步互相傳遞訊息**。CommNet、TarMAC、MAT 都採此設計，看似最 powerful（資訊上界）。

**為什麼是問題？**
1. **Bandwidth 物理限制**：5G/6G NR 基站之間的 X2/Xn interface 有實體頻寬上限；100 個 BS 全互通則控制平面流量是 $O(N^2)$，會直接壓垮 transport network。同樣道理也適用於多無人機編隊（air-to-air link 受功率限制）與車聯網（V2V 頻譜稀缺）。
2. **Signal-to-noise 反向 scaling**：當 N 變大，每個 agent 收到的 message vector 經 attention 或 mean-pool 聚合後，**實質是「全體的平均」**，個別重要訊號被稀釋。Jiang & Lu (2018) 的實驗顯示，N > 10 時 dense communication 反而比 partial communication 差 — 這是反直覺但已被多次重現的現象。
3. **Contention 與延遲**：wireless MAC 層上多 agent 同時送訊息會碰撞，real-time 部署不可能保證 every agent every step 都能收齊所有人的 message。

**具體後果**：
- MAT 在 SMAC 小 N（5–10 agent）跑得好，但在 25m, 27m_vs_30m 這種高 N 任務勝率明顯下滑（Table 1 數字反映此點）。
- 任何「全互通」的 paper 在工業部署評估時都會被打回票 — 因為頻寬假設不成立。
- 學術上的「資訊上界」與工程上的「不可實作」之間有巨大鴻溝。

##### (c) Dynamic Scheduling 太重 — 「即時排程」的工程災難

**直白定義**：ATOC、IC3Net、ToM2C 等方法在 inference 期**動態決定**「這個時間步誰跟誰講話」 — 它們不固定圖，而是訓練一個 scheduler / gating 模組來即時開關通訊邊。

**為什麼是問題？**
- **Scheduler 自身就要通訊**：要決定「i 是否該跟 j 講話」，scheduler 通常需要先看到 i 與 j 的部分狀態 — 這變成 chicken-and-egg：要知道誰該講話，自己已經先講了一輪。
- **Bandwidth 浪費在 control plane**：scheduler 的決策訊號（gating bits）跟實際資料訊息混搶有限頻寬，相當於要先付一筆「協商費」才能講話。
- **Latency 不可預測**：每個時間步重新計算圖，hardware-level 的 setup / teardown 成本很高 — 5G NR slot 只有 0.5ms，沒空在 slot 內重新建一條 X2 連線。
- **訓練不穩定**：scheduler 是**離散決策**（接通 / 不接通），需要 Gumbel-Softmax 或 REINFORCE 之類的高方差 estimator，常常要靠 reward shaping 才訓得起來。

**具體後果**：
- ATOC / IC3Net 在 paper 上看似 elegant，但實作複雜、reproduce 困難，**且幾乎都沒在大規模任務（>20 agent）驗證過**。
- 對 5G/6G 部署來說，動態 scheduler 違反「control plane 與 data plane 分離」的設計原則 — 工程師看到就頭痛。
- 即使能 train 起來，inference 期的不確定延遲讓 system-level QoS 保證變得困難。

##### 5G / 6G 場景對應（SCH-MARL final project 的現實意義）

> 你正在做的 SCH-MARL（5G/6G 多基站異質網路資源分配）**剛好同時被這三個痛點打中**：
> - **痛點 (a)**：BS 之間「該不該交換 channel state / load info」沒有顯然的最佳拓樸 — 不同地理部署、不同 traffic pattern 下都不一樣。
> - **痛點 (b)**：所有 BS 全互通會壓垮 backhaul，且 macro / micro / IoT relay 的異質性讓 mean-pool aggregation 喪失語義。
> - **痛點 (c)**：3GPP 規範下 X2/Xn interface 是「長連接」，不可能每個 slot 重新排程；scheduler 本身的訊息也要走同一條 backhaul。
>
> 因此 CommFormer「**離線學一張稀疏靜態圖、上線只用這張圖**」的策略剛好命中你的需求 — 它跟你 final project 的工程約束（bandwidth-constrained、deployable、低 control overhead）天然契合，這也是為什麼 CommFormer 是值得拿來做 SCH-MARL backbone 的核心理由。

---

作者主張：**在 inference 前先「離線」搜出一張稀疏靜態圖，比運行時動態調度更實用、更省頻寬**。

### 解決方案概述
- 將通訊架構建模為 **directed graph G = (V, E)**，N 個 agent 為 nodes，adjacency matrix $\alpha \in \mathbb{R}^{N \times N}$ 是**可學參數**。
- 用 **k-hot Gumbel-Softmax** 把離散邊選擇鬆弛為可微，允許梯度流回 $\alpha$。
- 訓練採 **bi-level optimization**：上層更新 $\alpha$（用 validation rollout），下層更新 encoder/decoder 參數 $\theta, \phi$（用 training rollout）。
- 用 **Multi-Agent Transformer (MAT, Wen et al. 2022)** 為 backbone — encoder 處理 obs sequence、decoder auto-regressive 產出 action sequence，並將 $\alpha$ 同時作為 **attention mask**（硬約束）與 **edge embedding**（注入結構資訊）。

---

## 2. Key Features（核心特色與方法拆解）

### 2.1 核心方法 Pipeline

| 階段 | 元件 | 作用 |
|------|------|------|
| Graph 取樣 | $\alpha \in \mathbb{R}^{N\times N}$ → k-hot Gumbel | 為每個 agent 取出 sparsity = $S$ 的 adjacency row $e_i$ |
| Encoder | Relation-enhanced Transformer | 觀察序列 $(o_1,\dots,o_n)$ → 表徵 $(\hat o_1,\dots,\hat o_n)$ |
| Critic head | Encoder 後接 MLP → $V_\phi(\hat o)$ | 估計 value，用 GAE 算 advantage |
| Decoder | Auto-regressive Transformer | 由 $\hat o_{1:n}$ 與已產出 action $a_{0:m-1}$ 預測 $\pi_m(a_m\|\hat o_{1:n}, a_{1:m-1})$ |
| 雙重 mask | Adjacency mask + causal (j<i) mask | 確保只接收連通 agent 的訊息 + 序列更新 |
| Loss | $\mathcal{L} = \mathcal{L}_{\text{Encoder}} + \mathcal{L}_{\text{Decoder}}$ | TD value loss + PPO clipped objective |
| Bi-level update | Eq.(9): $\theta, \phi \leftarrow$ train loss; Eq.(10): $\alpha \leftarrow$ val loss | 交替更新內外層 |

### 2.2 學習通訊拓樸的數學細節

**Adjacency 連續鬆弛 (Eq. 11)**：
```
e_i = k_hot ( k-arg max [Softmax(α_ij + g_j), j=1..n] )
```
其中 $g_j \sim \text{Gumbel}(0,1)$。直觀理解：
- 一般 Gumbel-Softmax 是 one-hot 的可微近似；CommFormer 把它推廣成 **k-hot**（每個 agent 必恰好連 $k = S \cdot N$ 條邊）。
- 訓練期 forward 用 hard k-hot，但 backward 用 softmax 的連續梯度（straight-through estimator 風格）。
- Inference 期 (Eq. 12) 直接 deterministic top-k：$e_i = \text{k\_hot}(\text{k-argmax}(\alpha_{ij}))$，不再注入 Gumbel noise。

**Bi-level Optimization (Eq. 6–8)**：
$$
\min_\alpha \mathcal{L}_{\text{val}}(\theta^*(\alpha), \phi^*(\alpha), \alpha) \quad \text{s.t.} \quad \theta^*, \phi^* = \arg\min_{\theta,\phi} \mathcal{L}_{\text{train}}(\theta,\phi,\alpha), \quad ||\alpha|| \leq S\cdot N^2
$$
作者沿用 **DARTS (Liu et al. 2019)** 的 first-order approximation：不解 inner optimization 到收斂，只走一步 SGD 後就用當前 $\theta, \phi$ 估計外層梯度。這是 NAS 圈最常見的近似手段，計算上才可行。

### 2.3 Relation-Enhanced Attention（Eq. 1–3）

把 vanilla self-attention $s_{ij} = o_i W_q^T W_k o_j$ 改寫為：
$$
s_{ij} = (o_i + r_{ij}) W_q^T W_k (o_j + r_{ji})
$$
其中 $r_{ij}$ 是從 $\alpha$ 取得的 **edge embedding**（不只是 mask，而是把邊的「身份」直接當 query/key 的偏移）。再施加 hard mask：
$$
\bar s_{ij} = s_{ij} \cdot [e_{ji}=1] + (-\infty) \cdot [e_{ji}=0]
$$
這個設計的**精妙之處**在於：mask 只決定「能不能通訊」，edge embedding 進一步表達「邊的語義」（誰指向誰、距離、優先級等）— 這是借自 Graph Transformer (Cai & Lam 2020) 與作者自己的 Graph Decision Transformer (Hu et al. 2023)。

### 2.4 技術新穎性（Novelty）

| 元件 | 是否新穎 | 借自 |
|------|----------|------|
| 把 communication topology 視為 learnable graph | **新** (在 MARL communication 領域) | NAS / DARTS 的概念遷移 |
| k-hot Gumbel-Softmax 處理 cardinality 約束 | 中等新穎 | one-hot Gumbel (Jang 2016) 的延伸 |
| Bi-level optimization for $\alpha$ vs $\theta,\phi$ | 借用 | DARTS (Liu et al. 2019)、NAS 通用框架 |
| Transformer encoder/decoder backbone | **借用** | MAT (Wen et al. 2022) — 幾乎沿用其架構 |
| Relation-enhanced attention | 借用 | Graph Decision Transformer (Hu et al. 2023, 同作者) |
| Auto-regressive sequential update + monotonic improvement | 借用 | MAT / HAPPO |

**結論**：CommFormer 的真正貢獻是把 MAT 的 fully-connected attention 替換成「**bandwidth-constrained, learned-sparse** attention」。架構創新較少，但問題形式化（formulation）與訓練策略（bi-level + k-hot Gumbel）是首次在 MARL communication 上系統化提出。

### 2.5 直觀理解（Intuition）

可以把 CommFormer 想像成**「自動排無線電群組」**：
- 你有 N 個前線小隊，每隊只能聽 k 個其他隊伍的無線電（頻寬限制）。
- 一開始你不知道誰該聽誰，於是給每對隊伍一個「重要性分數」 $\alpha_{ij}$。
- 在每次任務中（rollout），系統根據分數排出 top-k 收聽清單，執行任務拿到分數（reward）。
- 任務結束後，反向傳播更新「重要性分數」 — 表現好的連線就強化，表現差的就削弱。
- 訓練久了，重要性矩陣穩定，就得到一張「最佳無線電網」 — 部署時就用這張靜態圖，不用即時排程，省頻寬又可靠。

### 2.6 訓練流程偽代碼（Algorithm 1 簡述）

```
for each training iteration:
    # rollout (用 deterministic top-k graph，與 inference 一致)
    sample e_t = k_hot(k-argmax(α))
    生成 ô_t via Encoder with mask & edge_emb
    auto-regressively 生成 a_t via Decoder
    收集 (o_t, a_t, r_t) 入 replay buffer

    # learning (用 stochastic Gumbel k-hot，引入梯度)
    sample minibatch
    重新 sample e = k_hot k-argmax(softmax(α + g))
    forward Encoder + Decoder
    compute L_Encoder (TD) + L_Decoder (PPO clipped)
    update θ, φ with L_train (Eq. 9)
    update α with L_val (Eq. 10)
```

---

## 3. Contributions（貢獻分析）

### 3.1 作者宣稱的貢獻
論文 §1 末尾列出三點：
1. 提出將通訊架構視為 graph 並透過 bi-level optimization 同時學習 $\alpha$ 與 $\theta, \phi$。
2. 在 graph modeling framework 下用 attention 動態分配 message credit，並聲稱享有 **monotonic performance improvement guarantee**。
3. 大規模實驗（4 環境、>20 任務）驗證一致超越強 baseline。

### 3.2 我的獨立評估

| 貢獻 | 評價 | 理由 |
|------|------|------|
| Bi-level + k-hot Gumbel for graph learning | **Substantial** | 在 MARL communication 上是 first-of-its-kind formulation；以 NAS 角度看雖不算大新穎，但跨領域遷移本身有價值 |
| Monotonic improvement guarantee 聲稱 | **Over-claimed** ⚠️ | 此 guarantee 完全繼承自 MAT (Wen et al. 2022) 的 sequential update — 跟學習通訊圖**沒有直接關係**。論文沒有針對「動態變化的 $\alpha$ 如何不破壞 monotonicity」給出新證明。這是個 reviewer 紅旗。 |
| Static graph 哲學 | **Substantial** | 故意對立於 ATOC/ToM2C 的 dynamic scheduling，提出「pre-inference 學固定拓樸」是 deployable 的視角，對工程落地意義大 |
| 大規模實驗 | **Substantial 但不完美** | 涵蓋面廣，但見 §4 詳細評論 |

### 3.3 對社群的影響
- **學術**：把 NAS 的 bi-level 框架引入 MARL communication 是一個範式轉移；後續 ICLR/NeurIPS/AAMAS 應會出現多篇沿著這條路徑的延伸工作（如 dynamic re-search、heterogeneous edge type、multi-modal message）。
- **工業**：對 5G/6G 邊緣計算、多無人機協作、自動駕駛車隊有實際參考價值 — Appendix F 已點出 logistics warehouse 的應用場景。
- **限制**：因 backbone 完全綁 MAT，沿用其優缺點；對 value-decomposition 派（QMIX, QPLEX）不友善。

---

## 4. 實驗設計（Experimental Design）

### 4.1 Datasets / Benchmarks

| 環境 | 類型 | 為何選 | 是否合理 |
|------|------|--------|----------|
| **SMAC** (StarCraftII Multi-Agent) | 離散、hard exploration | cooperative MARL 標配 | 合理且必要 |
| **Predator-Prey (PP)** | 連續/離散混合、homogeneous | 經典 MPE 衍生 | 合理 |
| **Predator-Capture-Prey (PCP)** | 異質 agent（predator + capturer）| 測試 heterogeneous | 合理且重要 |
| **Google Research Football (GRF)** 3v2 | 稀疏 reward、隨機性高 | 測試 robustness | 合理 |

**缺漏的常用 benchmark**：
- ❌ **MAMuJoCo**（連續控制）— HARL/HASAC 標配，CommFormer 完全沒測，導致無法跟 off-policy continuous baseline (HATD3, HASAC) 直接比較。
- ❌ **Bi-DexHands** — 同上。
- ❌ **大規模 agent 環境**（如 MAgent, Neural MMO）— 既然主打 scalability，N>20 的測試卻缺席，論文最大的論據之一未驗證。

### 4.2 Baselines

**SMAC 部分**（Table 1 上半）：
- Communication-free: MAPPO, HAPPO, QMIX, UPDeT, MAT, FC（fully-connected CommFormer，作為 upper bound）
- 涵蓋面 OK，但**缺 HASAC**（同年 ICLR 2024）— 雖然時間上可能來不及加，但這是同會議直接競爭者，應在 camera-ready 補上。

**SMAC hard+ 任務**（Table 1 中段）：
- QGNN, SMS, TarMAC, NDQ, MAGIC, QMIX
- ⚠️ **這部分的 baseline 數字直接從 QGNN 論文 (Kortvelesy & Prorok 2022) 抄來**，非作者實跑 — 紅旗。雖然引用前人實驗在 MARL 圈算常見（避免重複跑大型實驗），但會削弱 fairness。

**PP/PCP/GRF**（Table 1 下半）：
- MAGIC, HetNet, CommNet, IC3Net, TarMAC, GA-Comm
- 涵蓋 communication 派 baseline 還算完整。

**Cherry-picking 嫌疑**：
- HAPPO 在多個 SMAC 任務（25m, 27m_vs_30m, MMM2, 6h_vs_8z, 3s5z_vs_3s6z）勝率 0% — 與 PKU-MARL 官方數字落差大。可能 baseline 跑得不夠認真，有過度貶低嫌疑。
- ⚠️ 這是讀 Table 1 時最該警覺的點。

### 4.3 Metrics
- **Win rate**（SMAC）— 標準
- **Success rate / Steps Taken**（PP, PCP, GRF）— 強調「最短時間達成」是合理動機
- **Average cumulative reward**（GRF）— 標準
- ❌ **缺通訊量度量** — 既然主打「省 bandwidth」，應該有「actual bytes / messages per step」的對比，但全文沒有；只用 sparsity $S$ 間接證明，說服力打折。

### 4.4 Ablations

**Sparsity $S$ ablation**（Figure 3）：
- 在多個 SMAC 任務測試 $S \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$。
- 結論：簡單任務（1c3s5z）$S=0.2$ 即可滿勝；複雜任務（10m_vs_11m, 27m_vs_30m）需要 $S \geq 0.4$。
- 這是論文最有價值的 ablation，給了實際部署選擇 $S$ 的指引。

**Architecture searching ablation**（Figure 4）：
- 用不同 random seed 生成手動 fixed graph，與 learned graph 對比。
- 證明 learned > random，但**沒對比 learned vs heuristic graph**（如 ring, fully-connected, k-nearest）— 這是個明顯 gap。

**消融的不足**：
- ❌ **沒拆掉 edge embedding $r_{ij}$ 看純 mask 的效果** — 不能確定 edge embedding 到底貢獻多少。
- ❌ **沒拆掉 bi-level**（用單層 unified loss 同時更新所有參數）— 不能確定 bi-level 是否真的必要。
- ❌ **沒對比 first-order vs second-order DARTS 近似** — 雖然 NAS 圈有共識用 first-order 即可，但對 reviewer 仍是一個合理問號。

### 4.5 實驗結果解讀
- **Headline 結果**：CommFormer($S=0.4$) 在 14 個 SMAC 任務有 11 個達 100% 勝率，多數匹敵或超越 FC（$S=1$）— 這非常關鍵，表示**40% 的邊已能達 100% 邊的效能**，是論文最 solid 的實驗論據。
- **Marginal gain**：5m_vs_6m (89.6% vs FC 93.8%)、3s5z_vs_3s6z (87.5% vs FC 100%) — 這兩個是 hard+ 任務，CommFormer 仍有 4–13% gap，誠實反映稀疏約束的代價。
- **令人困惑**：MAT 在 25m 上 0.0% — 與 MAT 原論文數字嚴重不符（原 MAT 在 25m 應有 100%），高度懷疑 baseline reproduction 出問題。

### 4.6 可重現性
- ✅ Code 已開源（github.com/charleshsc/CommFormer）
- ✅ Hyperparameters 詳列於 Tables 2–4（PPO epochs、clip、num blocks=1、num heads=1、stacked frames、γ=0.99）
- ⚠️ **單層單頭 Transformer**（num_blocks=1, num_heads=1）— 作者刻意保持極簡 backbone 以證明 graph learning 的貢獻，但這也意味著效能有相當部分依賴 backbone 設計簡潔，scaling up 是否仍 work 未驗證。
- ✅ 訓練成本：未明確報告 wall-clock，但 1 個 head + 1 個 block 的 Transformer 計算成本不大；推測單張 V100/3090 可訓 SMAC 任務。

---

## 5. Critical Thinking（深度思辨）

### 5.1 方法的根本假設
CommFormer 的**三個隱藏假設**：
1. **「最佳通訊圖是靜態的」**：論文明確選擇學一張固定圖，但這在許多場景不成立 — agent 角色會切換、任務 phase 會變、地圖是動態的。SMAC 雖然 work，是因為 agent 數量固定且角色分明（stalkers, zealots）。一旦 agent population 動態（如 Neural MMO、星海全戰），靜態圖可能崩壞。
2. **「Bandwidth 是 hard constraint，不是 cost」**：作者把 $S$ 當成 budget，但實際無線通訊中 bandwidth 是**可變 cost**（與能源、延遲耦合）。把它建模為 soft penalty 應該更貼近現實，但這會破壞 k-hot 的乾淨形式。
3. **「Sequential update + auto-regressive decoder 是 cooperative MARL 的正解」**：這完全繼承 MAT，但 MAT 的 sequential update 在 agent 數量大（>30）時 inference 延遲是 $O(N)$ — 對即時系統不友善，CommFormer 沒解決這個問題。

### 5.2 潛在弱點與 Failure Modes

| 弱點 | 描述 | 影響 |
|------|------|------|
| **Static graph 對 dynamic team 失效** | 訓練時 N=8，部署時 agent 增減則整個 $\alpha$ 失效 | 限制工業應用 |
| **Bi-level 不穩定** | First-order DARTS 在 NAS 圈以 collapse / mode degeneracy 聞名 — CommFormer 沒討論這個風險 | 訓練可能難複現 |
| **k-hot 約束過硬** | 每個 agent 必恰好連 k 個（不能有 0 個、也不能 k+1）— 真實系統中通訊需求極不均勻（hub agent vs leaf agent） | 限制表達力 |
| **Auto-regressive 推理延遲** | $O(N)$ sequential decoder forward — 100 個 agent 的時間步無法 real-time | 大規模部署瓶頸 |
| **Encoder 既是 critic 又是 representation generator** | 兩個目標耦合可能讓 critic loss 干擾 representation learning | 訓練不穩 |

### 5.3 隱藏的工程細節（performance 真正來源？）

讀完論文加 Appendix C/D，我懷疑 CommFormer 的效能不只來自 graph learning：
1. **Stacked frames=4 用於 3s_vs_5z** — 作者悄悄在某個 task 用了 frame stacking，這是強 boosting，但只此一個任務有，其他沒有。
2. **Hyperparameters per-task tuning**（Table 4 顯示 PPO epochs ∈ {5, 10, 15}, clip ∈ {0.05, 0.2}）— 每任務調參，這在 cooperative MARL 圈算常規但仍要警覺。
3. **MAT 本身就是強 backbone** — 把 MAT 當 baseline 看 Table 1，CommFormer 對 MAT 的提升只有少數任務顯著（如 3s5z, 25m, 27m_vs_30m, MMM2, 6h_vs_8z），其他任務兩者打平。也就是說，**CommFormer 真正的 gain 集中在 hard+ 異質單位任務**，這也是它最該主打的賣點。

### 5.4 與既有工作的關係

| 既有工作 | CommFormer 的差異 |
|---------|--------------------|
| **MAT (Wen 2022)** | 同 backbone；CommFormer = MAT + learnable sparse mask |
| **TarMAC (Das 2019)** | TarMAC 是 dense soft attention；CommFormer 是 hard k-hot mask |
| **ATOC (Jiang 2018)** / **IC3Net (Singh 2018)** | ATOC/IC3Net 用 gating 動態決定通訊；CommFormer 是 pre-inference static graph |
| **MAGIC (Niu 2021)** | MAGIC 用 hard attention + GAT；CommFormer 直接學 binary adjacency 而非 attention coefficients |
| **CDC (Pesce & Montana 2023)** | CDC 用 diffusion 動態調圖；CommFormer 強調 static |
| **TWG-Q (Liu 2022)** | TWG-Q 學 temporal weight + GCN；CommFormer 學 binary graph |
| **DARTS (Liu 2019)** | NAS 始祖；CommFormer 把 DARTS 的搜尋空間從 op 換成 edge |

**真正超越了它們嗎？** 在 SMAC 上是；在 dynamic team / partial observability 嚴重 / agent 數量動態的場景，未必。

### 5.5 Open Questions

1. **學到的 graph 是否真的 interpretable？** Figure 5 只展示了 1c3s5z 的搜尋過程，沒有對照「stalkers 應該偏好聽誰」的領域直覺。圖學到的結構是否有物理意義？還是純統計巧合？
2. **如果加 noise 在 communication channel 上呢？** 真實無線通訊有 packet loss，CommFormer 沒測 robustness。
3. **能否 zero-shot transfer 到不同 N？** 論文沒給跨 N 遷移實驗。
4. **與 emergent language 結合？** message 目前是 raw obs/action，沒有 emergent symbolic communication。

### 5.6 未來研究方向（資深 reviewer 視角）

1. **Dynamic CommFormer**：把 $\alpha$ 條件化在 task phase 或 agent state 上，搜出一族圖而非一張靜態圖，bridge static vs dynamic 的對立。
2. **Heterogeneous edge types**：把 binary adjacency 換成 multi-relational（誰指揮誰、誰共享狀態、誰僅傳警報），對應現實中多種通訊類型。
3. **CommFormer × off-policy（如 HASAC）**：本文 backbone 限定 on-policy PPO，限制 sample efficiency。把 graph learning 接入 SAC 系列是明顯機會 — **這正是使用者 final project 的切入點**。
4. **Bandwidth-as-cost**：把 hard k-hot 約束改為 soft entropy penalty $\lambda \cdot H(\alpha)$，允許 cardinality 自適應 — 這需要重新設計 Gumbel-Softmax 變體。
5. **與 Graph Neural Network 後 message passing 結合**：目前 CommFormer 只用 Transformer attention 聚合，可比較 GNN-based aggregation 的差異。

---

## 6. 與 HASAC 結合的可行性分析（SCH-MARL Final Project 專屬）

> **背景**：使用者欲將 CommFormer 的 communication graph 學習機制與 HASAC（HARL ICLR 2024）結合，應用於 5G/6G 多基站異質網路資源分配。本節分析整合的具體方案、難點與切入點。

### 6.1 兩者的天然契合與張力

| 面向 | HASAC | CommFormer | 契合度 |
|------|-------|------------|--------|
| 範式 | CTDE off-policy actor-critic | CTDE on-policy（PPO 為核心）| ⚠️ 衝突需橋接 |
| Critic | Centralized Q 函數 $Q_\psi(s, a^{1:n})$ | Encoder 輸出 $V_\phi(\hat o)$ | 部分相容 |
| Actor 結構 | Per-agent independent π，配 reparam | Auto-regressive Transformer decoder | 衝突需重構 |
| 理論保證 | MEHAML 模板 → monotonic + QRE 收斂 | MAT 的 sequential monotonic improvement | 兩套不同框架 |
| Heterogeneity | 原生支援 (HARL = Heterogeneous-Agent RL) | Transformer 共享參數，異質性靠 obs encoding | HASAC 較強 |
| Communication | 無原生機制，假設 fully observable joint state at critic | 內建 sparse graph 學習 | **CommFormer 補 HASAC 缺口** |

### 6.2 整合的三種具體方案

#### 方案 A: CommFormer-as-Communication-Module + HASAC-as-Policy（建議優先）

**架構**：
```
[每個 agent local obs o_i]
       ↓
[CommFormer Encoder + learned α] → [aggregated message m_i]
       ↓
[m_i ⊕ o_i] → HASAC Actor π_i (per-agent independent, with reparam) → a_i
       ↓
[s, a^{1:n}] → HASAC Centralized Critic Q_ψ
```
- **好處**：保留 HASAC 完整 MEHAML 理論（per-agent independent actor 是其 cornerstone），CommFormer 只當「通訊預處理層」。
- **訓練**：bi-level 仍保留 — 內層更新 (HASAC 的 actor + critic + α 控制)、外層更新通訊參數 $\alpha$（用 critic loss 或 HASAC objective 的 validation 變體）。
- **理論問題**：MEHAML 的 sequential update 假設每個 agent 看到「之前 agent 已更新後的 policy」 — 加入 communication 後，message $m_i$ 也是 policy 的函數，需要重新驗證 drift functional $D^i$ 的 monotonicity 條件是否仍成立。
- **HADF 是切入點**：依你的 HARL notes，可改寫 $D^i(\hat\pi^i \| s, \bar\pi^{j_{1:m}})$ 為 $D^i(\hat\pi^i \| s, \bar\pi, G_{\text{comm}})$，其中 $G_{\text{comm}}$ 是 CommFormer 學到的 sparse graph — 這是合法的擴展，因為 graph 在每個 update step 為固定（k-hot 取樣後 detach），不破壞 drift functional 的可求和性。

#### 方案 B: HASAC + Graph-Conditioned Critic

**架構**：保留 HASAC 原生 actor，但 critic $Q_\psi(s, a, G)$ 額外條件化在通訊圖上，actor 之間訊息透過 GNN message passing。
- **好處**：CommFormer 的 graph 直接成為 critic 的輸入，不改 actor 結構。
- **缺點**：actor 之間缺乏 explicit communication，浪費 CommFormer 的 graph 資訊。

#### 方案 C: 完整融合 MEHAML + Communication Bi-level

**架構**：把 CommFormer 的 bi-level 嵌入 MEHAML 模板，提出一個新理論框架（暫稱 **C-MEHAML**）：
- 內層：MEHAML drift functional 最小化（更新 actor + critic）
- 外層：communication graph $\alpha$ 更新（用 validation MaxEnt objective）
- **這是 final project 的最強 selling point**，但需要新的 monotonic improvement proof，難度高。

### 6.3 哪些 component 可以直接用、哪些要改寫

| 元件 | 來源 | 處理方式 |
|------|------|---------|
| **k-hot Gumbel-Softmax** | CommFormer Eq. 11 | ✅ 直接用 |
| **Bi-level update schedule** | CommFormer Eq. 9–10 | ✅ 直接用，但外層 loss 改為 HASAC objective 的 val 版 |
| **Relation-enhanced attention** | CommFormer Eq. 2 | ✅ 可選用，作為 message aggregator |
| **MAT-style auto-regressive decoder** | CommFormer/MAT | ❌ **應丟棄** — 與 HASAC 的 per-agent independent actor 衝突，且 sequential decoder 在 SAC 的 reparam trick 下無法保證 monotonic improvement |
| **HASAC Actor (per-agent π_i)** | HARL/HASAC | ✅ 保留，但 input 加上 message |
| **Centralized Q critic** | HASAC | ✅ 保留 |
| **Auto-tuned α (entropy temperature)** | HASAC | ✅ 必開 |
| **Sequential update + random permutation** | HARL | ✅ 保留（這是 MEHAML 的 cornerstone） |
| **CommFormer Encoder 作為 critic** | CommFormer Eq. 4 | ❌ **應丟棄** — HASAC 已有自己的 Q critic，雙 critic 會互相干擾 |
| **PPO clipped objective** | CommFormer Eq. 5 | ❌ **應丟棄** — HASAC 是 off-policy SAC-style |

### 6.4 HAML / HASPI 是否容許這樣的擴展？

依你的 HARL notes，HAML 的 drift functional 抽象為 $D^i(\hat\pi^i \| s, \bar\pi^{j_{1:m}})$，要求滿足三個性質（non-negative、smooth、bounded difference）。注入 communication graph $G$ 後：
- 若 $G$ 在每個 update iteration 內為**固定 random variable**（k-hot 取樣後 detach 梯度），則 drift functional 視 $G$ 為條件，本質上是 $D^i(\hat\pi^i \| s, \bar\pi^{j_{1:m}}, G)$ — 三個性質**仍然成立**（因為條件化不改變函數的解析性質）。
- 若 $G$ 隨 $\hat\pi^i$ 變化（因為 actor update 會影響下次 graph 取樣），則需要更小心的論證 — 但 CommFormer 的 bi-level 設計剛好把 $\alpha$ 跟 $\theta, \phi$ 的更新**解耦在不同 step**，避免這個問題。

**結論**：CommFormer × HASAC 在理論上是可行的擴展，**HADF (Heterogeneous-Agent Drift Function) 是合法的切入點**，但需要在 final project paper 裡明確寫出條件化版本的 drift functional 並重新驗證 monotonic improvement 條件。

### 6.5 訓練流程整合策略（建議）

```
Initialize: HASAC actor π_i, critic Q_ψ, target Q_ψ̄, comm matrix α, replay buffer B

for each iteration:
    # ROLLOUT (deterministic graph)
    G = k_hot(k-argmax(α))  # detach
    for each step t:
        for each agent i:
            m_i = CommFormerEncoder(o_{1..n}, G)[i]
            a_i = π_i(o_i ⊕ m_i)  # SAC reparam sample
        execute a, observe r, s'
        push to B

    # LEARNING (stochastic graph for inner, val rollout for outer)
    sample minibatch from B
    G_train = k_hot(k-argmax(softmax(α + g)))  # Gumbel sample, gradient flows

    # Inner: HASAC update (with random permutation, sequential)
    for i in random_permutation(N):
        update Q_ψ via TD on (s, a, r, s', G_train)
        update π_i via reparameterized policy gradient (HASAC objective)
        update α_i (auto-tuned entropy)

    # Outer: Communication graph update (every K_outer iterations)
    if iter % K_outer == 0:
        sample val rollout
        compute L_val (HASAC objective on val rollout)
        update α via ∇_α L_val (Eq. 10 of CommFormer)
```

**關鍵實作建議**：
- **Replay buffer 要儲存 $G$**：因為 off-policy buffer 中存的 $(s, a, r, s')$ 是過去 $\alpha$ 對應的圖生成的，re-use 時要小心 importance weighting；最簡單做法是每次 graph 變動就清空 buffer 或用 prioritized replay。
- **外層更新頻率 K_outer** 不要太頻繁（建議 K_outer = 100~500 iterations），否則 graph 抖動會破壞 actor 的學習穩定性。這在 DARTS 圈也是常見做法。
- **驗證集**：可以用 hold-out env seeds 或 latest rollout 的後半段作為 val。

### 6.6 SCH-MARL 應用情境的特殊考量

5G/6G 多基站異質網路資源分配的特性：
- **異質 agent**：macro BS、micro BS、IoT relay — HASAC 原生支援，CommFormer 的 edge embedding 也能編碼 agent 類型差異。
- **動態用戶分布**：load 隨時間變化，**這對 CommFormer 的 static graph 假設構成挑戰** — 建議方案 C 或在 inference 期定期重新搜 graph（小規模 fine-tuning）。
- **嚴格 bandwidth constraint**：CommFormer 的 k-hot 直接對應 X2/Xn interface 的連接限制，應用契合。
- **Stochastic policy 必要性**：用戶分布不確定 → HASAC 的 MaxEnt 是合適選擇；不要退回 deterministic policy（HATD3）。

**Final project 可主打三個賣點**：
1. **第一個** MaxEnt MARL × Learnable Communication Topology 的整合工作。
2. **理論貢獻**：條件化版 MEHAML drift functional 的 monotonic improvement 條件。
3. **應用驗證**：5G/6G 資源分配 simulator（如 Sionna RT 或 ns-3 + Open5GS）下的實證。

---

## 7. 給研究者/學習者的 Takeaways

### 7.1 如果要復現 CommFormer 需注意什麼
- **Backbone 用單層單頭 Transformer**（Table 4 顯示 num_blocks=1, num_heads=1）— 不要無腦堆深。
- **PPO clip 在 SMAC hard+ 任務調到 0.05**（從預設 0.2 下調），這是穩定關鍵。
- **Stacked frames=4 只用於 3s_vs_5z**，其他任務 stacked=1。
- **訓練步數差異大**：Easy task 5e5 steps，Hard+ 1e7~2e7 steps，對應到 SMAC 慣例。
- **Rollout threads=16, episode length=32, batch_size=3200** — 這些數字直接決定你能多快收斂。
- **bi-level 用 first-order approximation 即可**，不需做 second-order Hessian 估計。

### 7.2 如果要在此基礎上做研究（low-hanging fruit）
1. **加 Heterogeneous Edge Types**（multi-relational graph）— 改寫 $\alpha$ 為 $\alpha \in \mathbb{R}^{N \times N \times R}$，R 是邊類型數。
2. **替換 PPO 為 HASAC**（如 §6 所述）— **這正是 SCH-MARL final project**。
3. **Dynamic Re-search at Inference**：在 deployment 期定期觸發 small-step graph update，bridge static vs dynamic 二分。
4. **Soft cardinality penalty** 取代 hard k-hot，允許 cardinality 自適應 + 加 entropy regularization。
5. **Cross-task graph transfer**：學到的 $\alpha$ 能否 zero-shot 應用到不同 N 的任務？
6. **Robustness ablation**：加 message dropout / Gaussian noise 測試 communication channel 不可靠下的表現。

### 7.3 學習價值（對 DL 學生）
| 概念 | 學什麼 |
|------|--------|
| **k-hot Gumbel-Softmax** | 如何把 cardinality-constrained 離散選擇變可微 |
| **Bi-level optimization** | NAS 圈最重要的 trick，在 RL/Meta-Learning 處處可用 |
| **Relation-enhanced attention** | 如何把結構資訊（邊）注入 Transformer，超出純 mask 的層次 |
| **DARTS first-order approximation** | 為什麼可以省略 Hessian 仍 work — 這是 ML 系統設計的工程哲學 |
| **CTDE 範式** | MARL 的標準訓練/部署策略 |

### 7.4 作為 DL final project 的可行性評估

| 維度 | 評估 |
|------|------|
| **實作難度** | 中高 — 需要熟悉 PyTorch + Transformer + MARL training loop |
| **計算資源** | 中 — 1 張 RTX 3090/V100 可訓 SMAC 任務（一個任務 ~12-48 hr） |
| **創新空間** | **大** — CommFormer × HASAC 的整合本身就是 publishable 級別的延伸 |
| **與使用者背景契合** | 高 — 涵蓋 GNN（graph learning）、RL（HASAC）、系統設計（5G/6G 應用） |
| **風險點** | bi-level + off-policy 的穩定性需要謹慎調校；可能需要 2-3 週的 sanity check 才能 reproduce CommFormer 原始效能 |
| **建議起點** | (1) 先在 HARL codebase 跑通 HASAC baseline；(2) 再 fork CommFormer 並只保留 graph learning 模組；(3) 把 HASAC actor/critic 接上 |

---

## 8. 綜合評分（Overall Assessment）

| 維度 | 分數 (1-10) | 評論 |
|------|-------------|------|
| **Novelty（新穎性）** | **7/10** | Bi-level 學 communication graph 是首次系統化提出，但架構大量借用 MAT/DARTS。Formulation 創新 > 技術創新。 |
| **Technical Soundness（技術紮實度）** | **7/10** | Bi-level + k-hot Gumbel 設計嚴謹，但 monotonic improvement 聲稱有 over-claim 嫌疑（不是 graph learning 帶來的，是 MAT 帶來的）。 |
| **Experimental Rigor（實驗嚴謹度）** | **6.5/10** | SMAC 涵蓋廣，但缺 MAMuJoCo / Bi-DexHands；HAPPO baseline 數字異常低；缺 edge embedding 與 bi-level 的關鍵 ablation；缺實際 bandwidth metric。 |
| **Clarity & Presentation（清晰度）** | **7.5/10** | Method 章節數學形式清楚，Algorithm 1 完整。但 Figure 5 的 visualization 不夠深入解讀；experimental setup 不少細節塞 Appendix。 |
| **Potential Impact（潛在影響力）** | **8/10** | 把 NAS 思維引入 MARL communication 是範式級貢獻；對工業 5G/edge computing 有直接應用價值；後續會孕育多篇延伸工作。 |

### Overall Recommendation: **Accept**

**評論摘要**：
CommFormer 在 ICLR 2024 是合格的 accept paper。它解決了一個重要實務問題（bandwidth-constrained communication learning），formulation 清楚，實驗有說服力，code 開源。

**Accept 的關鍵理由**：
- Static learnable graph 是個有 deployment 價值的視角，對立於 dynamic scheduling 路線並提出可行替代方案。
- SMAC 上 $S=0.4$ 即可逼近 FC 上界，這是個 strong empirical evidence。
- 對 cooperative MARL communication 圈是顯著推進。

**Reject 的可能理由**（如果嚴格的 reviewer 給）：
- Monotonic improvement guarantee 過度聲稱，與本文核心貢獻無關。
- 實驗缺 MAMuJoCo / 大 N 場景；HAPPO baseline 數字可疑。
- 關鍵 ablation 缺失（edge embedding、bi-level 的必要性）。

我作為 area chair 會給 **6 / 8 / 6 / 7** 的 reviewer 評分混合，最終 accept as poster — 接近 borderline accept 但 leaning positive。對社群有貢獻但不到 spotlight 等級。

---

## 附錄：與 HARL/HASAC 比較表

| 面向 | HARL/HASAC (ICLR 2024) | CommFormer (ICLR 2024) |
|------|-----------------------|------------------------|
| 主貢獻 | MaxEnt 框架 + MEHAML 模板 + monotonic improvement 證明 | Learnable communication graph + bi-level optimization |
| 範式 | Off-policy actor-critic (SAC-style) | On-policy actor-critic (PPO-style) |
| Communication | 無 (CTDE assumes joint state at critic) | Sparse learned graph |
| 理論貢獻 | 嚴謹 (Joint Soft Policy Decomposition + 4 properties) | 較弱（沿用 MAT） |
| Heterogeneity | 原生支援 | 透過 obs encoding |
| Benchmark 廣度 | 6 個 (SMAC, MAMuJoCo, Bi-DexHands, GRF, MPE, LAG) | 4 個 (SMAC, PP, PCP, GRF) |
| 主打優勢 | 理論完備 + 多 benchmark SOTA | 部署友善 + bandwidth-aware |
| 對 final project 角色 | **Policy backbone** | **Communication module** |

兩篇結合的核心價值：**HARL 提供理論完備的 stochastic policy 框架，CommFormer 補上原生缺失的 communication topology 學習** — 這正是 SCH-MARL 在 5G/6G 場景需要的兩塊積木。

---

*報告完成於 2026-04-23，基於 CommFormer ICLR 2024 全文（含 Appendix A–F）細讀與 HARL/HASAC 記憶筆記交叉分析。*
