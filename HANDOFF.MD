# 交接文件:H-HASAC → **Context-Conditioned HASAC**(架構修正 / 新 Claim / Baseline / 實驗)

> 對象:隊友 + 隊友的 AI coding assistant。
> 目的:取代舊的 H-HASAC 報告框架。舊報告的 motivation 與 manager 設計有結構性問題,本文件給出修正後的 claim、架構、baseline 與可直接實作的規格。
> 核心不變量(全文件最重要的一句):**`metrics()`(channel → SINR → rate/reward)是唯一真相來源,baseline / training / eval 三邊全部呼叫同一個函數,否則比較無效。**

---

## 0. TL;DR(先看這段)

1. **舊報告把 motivation 掛在「3GPP eICIC / X2 RAN Controller 發 power budget」是錯的。** eICIC 是 LTE、是 distributed peer-to-peer、且 3GPP 只標準化 signaling 不標準化決策;5G NR 沒有對應的標準 ICIC。真正對應你架構的是 **O-RAN Near-RT RIC (xApp) ↔ DU via E2**。
2. **舊的 manager「發 hard 約束 / power budget」沒有乾淨的規範旋鈕。** E2SM-RC 沒有 per-PRB 功率控制。改成 **manager 產生一個 learned latent `z`,當作各 BS 的額外 input(soft conditioning)**,並 **end-to-end 訓練、不需要 manager reward**。
3. **Baseline 不能只用 round-robin。** 強 baseline 是 **WMMSE**(實測距 near-global 僅 6%);round-robin/full-power 距最優 70%,贏它證明不了任何事。
4. **發現一個硬限制:max sum-rate 會把 Jain fairness 壓到 ~0.33。** 所以若 reward 含 fairness,baseline 必須用 **同目標**(PF-weighted WMMSE),否則目標不一致、比較無意義。
5. **RL 的貢獻只在「partial-obs / 即時 / 時序」場景成立。** full-CSI 單次場景 WMMSE 通常贏 RL;那種場景 RL 的角色只是天花板對照。

---

## 1. 背景修正(為什麼要改)

### 1.1 3GPP vs O-RAN(motivation 層級)
- **ICIC / eICIC = LTE**(Rel-8 / Rel-10),透過 **X2AP `LOAD INFORMATION`** 交換,IE 為:
  - `RNTP`:DL,per-PRB **1-bit**(功率是否超 threshold),預測性
  - `HII`:UL,per-PRB 1-bit(高干擾敏感),預測性
  - `OI`:UL,per-PRB **3 階**(已發生干擾),反應性
  - `ABS`(eICIC):時域 almost-blank subframe pattern
- **3GPP 只標準化「交換什麼資訊」,不標準化「收到後怎麼決策」**(implementation-specific)。
- **協調是 distributed peer-to-peer,沒有中央 controller**(2G/3G 才有 BSC)。所以舊報告「RAN Controller 集中發指令」的圖是錯的。
- **NR 沒有標準化的 ICIC signaling**(沒有 NR 版 RNTP/HII/OI over Xn)。集中式 + 慢時間尺度的協調,真正的載體是 **O-RAN Near-RT RIC + E2(10 ms–1 s)**,不是 3GPP X2/Xn。
- 參考規範:TS 36.300(LTE overall)、TS 36.423(X2AP IE)、TS 36.213(RNTP threshold)。**引用前請到 3GPP portal 對章節**。

### 1.2 結論:正確的系統對應
| 角色 | 對應到 |
|---|---|
| Manager | **O-RAN Near-RT RIC 上的 xApp** |
| Worker | **gNB-DU 的 MAC scheduler** |
| Manager → Worker 通道 | **E2 介面(E2SM-RC control / A1 policy)** |
| Manager 觀測來源 | **E2SM-KPM Indication report** |

---

## 2. 規範可得的 State / Action(實作要照這個,不能憑空編)

### 2.1 Manager(xApp)的 input — 只有 E2SM-KPM 拿得到的量測(基於 3GPP TS 28.552)
| 類別 | metric | 備註 |
|---|---|---|
| 負載 | `RRC.ConnMean` / `RRC.ConnMax` | 各 cell RRC connected UE 數 |
| 吞吐量 | `DRB.UEThpDl` / `DRB.UEThpUl` (kbps) | cell / UE / node level |
| PRB 使用 | `RRU.PrbTotDl/Ul`、`RRU.PrbAvailDl/Ul` (%) | cell scope |
| 資料量 | `DRB.PdcpSduVolumeDL/UL` | |
| 延遲 | `DRB.RlcSduDelayDl` | |

**拿不到的東西(重要)**:per-RB SINR、raw 通道增益、干擾矩陣。CQI / RSRP / RSRQ 不在 baseline KPM,需走 E2SM-RC UE-info report 或 vendor extension,**不保證有**。
→ Manager state 用 throughput / PRB / load 當干擾的 **間接 proxy**,**不要**直接餵 per-RB SINR(舊報告的 `share_obs` 用 SINR,真機訂不到)。

### 2.2 Worker(DU scheduler)的 state / action
- **state**:per-UE/per-RB CQI/CSI、buffer/QoS(5QI)、HARQ、PRB 佔用、近期 throughput + **Manager 注入的 `z`**。
- **action(規範可控範圍)= E2SM-RC Control Service Styles**:
  | Style | 名稱 | 控制內容 |
  |---|---|---|
  | 1 | Radio Bearer Control | DRB QoS、admission、split bearer |
  | **2** | **Radio Resource Allocation** | **PRB / DRX / SPS,per node/cell/slice/UE/QoS** ← 主力 |
  | 3 | Connected Mode Mobility | handover |
  | 4 | Radio Access Control | admission |
  | 5/6 | DC / CA | |
  | 7 | Idle Mode Mobility | reselection |
  | 8 | UE info & assignment | |
  | 9 | Measurement Reporting Config | |
  | 10 | Beamforming Config | per-UE beamforming |
  | 255 | Multiple Actions | 組合 |
- **關鍵**:**baseline E2SM-RC 沒有 per-PRB Tx power 控制**。功率/RF 屬 E2SM-CCC / O1 / O-RU domain。
  → 若把 action 定成「per-RB power fraction」,**那是 sim-only 抽象**,要在論文明說;規範可落地的 action 是 **PRB allocation / scheduling weight(Style 2)** 或 **beamforming(Style 10)**。

### 2.3 整個 system 的目標(不是 max bandwidth)
O-RAN WG3 標準化排最前的 use case 是 **QoS/QoE optimization + Traffic Steering**。真正目標 = **在每個 5QI/slice 的 QoS 約束下,兼顧 spectral efficiency 與 fairness 的 utility 最大化**。常見目標函數:
- Proportional Fair:`max Σ log(throughput)`(業界 default)
- max-min rate(公平優先)
- QoS-aware weighted utility(eMBB 追 rate、URLLC 要 PER 10⁻³ / 低延遲)

**裸 sum-rate 是錯的目標**(見 §5 數據:會把 Jain 壓到 0.33)。

---

## 3. 為什麼用 RL(誠實版)

- WSRM(weighted sum-rate max,含干擾耦合)是 **non-convex 且 NP-hard**;離散 RB 分配是 NP-complete。**global optimum 確實算不出來。**
- **但**有 WMMSE / FP 等強近似解(收斂到 stationary point),所以「算不出來」**不是**用 RL 的真正理由。
- **RL 真正解決的是 WMMSE 做不到的事**:
  1. **拿不到 full instantaneous CSI**(RIC 只有 KPM)→ RL model-free,能從 partial obs 學。**(最強理由)**
  2. **即時 / amortization**:WMMSE 迭代、複雜度高;RL 推論一次 forward pass。
  3. **時序**:queue/mobility/handover → RL 優化長期 return。
  4. **分散式 + 多智能體賽局**:CTDE 集中訓練、本地執行。
- **推論**:RL 的優勢只在 partial-obs / online / sequential 場景測得出來。**強 baseline 必須是 WMMSE,不是 round-robin。**

---

## 4. 新架構:**Context-Conditioned HASAC**(取代 feudal-budget)

### 4.1 核心想法
vanilla CTDE 假設「inference 時 actor 去中心化、無通訊」,所以全域資訊只活在 training 的 critic。但 **O-RAN runtime RIC→DU 通道(E2)真的存在**,所以這個假設過嚴。
→ 讓 Manager 把全域 KPM **壓成一個 learned latent `z`**,**broadcast 給每個 worker 當額外 input**。BS 彼此看不到,但 `z` 提供它們需要的跨 cell 協調訊號。

```
            global KPM
                │
            f_θ (permutation-invariant encoder)   ← Manager
                │  z ∈ R^d  (每 K 步更新一次, 中間 hold)
       ┌────────┼────────┐
       ▼        ▼        ▼
  π_φ(o₁,z) π_φ(o₂,z) π_φ(o₃,z)   ← Workers (parameter-shared)
       │        │        │
       a₁       a₂       a₃
                │
        Q_ψ(share_obs, a₁..a₃)     ← centralized critic (training only)
```

### 4.2 與舊設計的差異
| | 舊(feudal-budget) | 新(context-conditioned) |
|---|---|---|
| Manager 輸出 | `(p_max, i_thresh, rb_share)` 硬約束 | learned latent `z`(soft input) |
| Manager 訓練 | 獨立 SAC + 自己的 reward/buffer | **可微 encoder,end-to-end backprop,無 manager reward** |
| worker 動作約束 | hard clip | 無,z 是 conditioning |
| 全域資訊用在 | 只在 critic(training) | critic + actor(inference 也用) |
| 規範對應 | 無對應旋鈕 | z 經 E2SM-RC/A1 下發(低維、overhead 低) |
| scalability | subgoal 維度 = 3×N_BS,scale 不了 | z 固定維度 + 共享 worker → 任意 N_BS |

### 4.3 訓練流程(off-policy SAC)
```
forward:  z = f_θ(global_KPM)            # hold K steps
          a_i = π_φ(o_i, z)
          Q = Q_ψ(share_obs, a_1..a_N)

update (從 replay 取 transition; 必須存 global_KPM 與 o_i):
          z = f_θ(global_KPM)            # 重算, 不存 stale z, 讓梯度可回傳
          actor loss = Σ_i SAC_actor_loss(π_φ(o_i,z), Q, entropy)
          ∂loss/∂θ 經由 z 回傳進 encoder  # encoder 學「對集體最有用的 z」
```
- encoder 收到所有 worker actor loss 的總梯度。**完全不需要設計 manager reward。**
- 兩時間尺度:`z` 每 K 步重算(模擬 RIC 慢、DU 快),window 內 z 穩定 → 降低 worker 的 non-stationarity。

### 4.4 encoder 設計(顧 scalability)
`f_θ` 用 **DeepSet / attention pooling**:per-cell KPM → MLP 編碼 → mean/attention 聚合 → `z`。
→ z 維度固定、對 cell 排列不變、N_BS 任意擴展;配 parameter-shared workers → 整套 zero-shot scale。

### 4.5 必做的 ablation:**z-probe**
訓練後測「`z ← 0`(或 shuffle)後 worker 掉多少」。**掉得多才證明 z 真的在協調**;若幾乎不掉,代表 z 退化、設計沒成立。
→ 這也是為什麼必須用 **R2(partial-obs)** 場景:full-CSI 下 z 沒人理,設計沒意義。

### 4.6 命名 / framing
這 **不是** feudal RL(manager 不發 goal、worker 無 goal-following loss)。屬於 **learned-coordination / context-conditioned decentralized policies**(近 CommNet 的 learned message、CTDE 的延伸)。論文這樣定位,別叫 feudal/budget。建議名:**Context-Conditioned HASAC (C-HASAC)**。

---

## 5. Baseline 計畫(已部分實作,見 `baselines.py`)

### 5.1 已驗證數據(200 snapshots,formula channel,3-BS/10-UE/4-RB)
```
method        sum-rate(bps/Hz)   Jain    gap vs optimal   ms/snap
full_power        28.1 ±10.8     0.818        70.5%        0.01
wmmse             89.3 ±13.3     0.420         6.1%       67.8
grid_opt          95.2 ± 7.4     0.333         0.0%       93.7
```
**兩個結論:**
1. **WMMSE 是強 baseline**(距 near-global 僅 6%);full_power/round-robin 是 strawman。
2. **max sum-rate ⟹ Jain 崩到 0.33–0.42。** sum-rate 與 fairness 直接打架。

### 5.2 Baseline 清單(要全做)
| baseline | 角色 | 狀態 |
|---|---|---|
| `full_power` / round-robin | 弱 floor(interference-limited 對照) | ✅ 已實作 |
| `wmmse`(純 sum-rate) | 強 baseline / full-CSI 天花板 | ✅ 已實作 |
| `grid_opt`(3-BS 暴力) | near-global 參考,量 optimality gap | ✅ 已實作 |
| **`pf_wmmse`(PF-weighted)** | **同目標 baseline(reward 含 fairness 時必用)** | ⬜ **TODO** |
| `Independent SAC` | 無協調學習對照 | ⬜ TODO |
| `MAPPO` | on-policy MARL 對照 | ⬜ TODO(HARL 內建) |

> **硬規則**:RL 優化哪個 objective,就要有「優化同一個 objective 的 baseline」。reward 含 fairness → 必須跑 `pf_wmmse`(weight `w_i = 1/throughput_i`),不能拿純 sum-rate WMMSE 比。

---

## 6. 實驗協議

### 6.1 三個資訊 régime(決定 RL 有沒有優勢可證)
| régime | worker obs | 用途 | WMMSE 可否跑 |
|---|---|---|---|
| **R1 full-CSI** | 等效增益 + 自身 rate | 天花板對照(RL 多半輸,誠實) | ✅ |
| **R2 KPM-only** | 只給上輪 `DRB.UEThpDl`/`RRU.PrbTot`/`RRC.ConnMean`,**無 CSI** | **RL 的主戰場(WMMSE 跑不動)** | ❌ |
| **R3 sequential** | R2 + UE 移動 + buffer queue,delay-aware reward | 證明 hierarchy / 長期 return | ❌ |

**你的 claim 只在 R2 / R3 成立。**

### 6.2 必須先鎖定的兩個設計決策
1. **功率約束**:目前 per-RB cap → 因 channel frequency-flat,**4 個 RB 退化成同一子問題,RB 維度無作用**。二選一:
   - (a)改 **per-BS sum-power 約束**(Σ_rb P ≤ P_total)→ RB 耦合,變真正的 spectrum-balancing(建議)
   - (b)加 **per-RB frequency-selective fading**(`baselines.py` 已有 `freq_selective` flag)
2. **資訊 régime**:預設跑 R2 作為主結果,R1 當 ceiling,R3 當延伸。

### 6.3 評測流程
- 固定一批 **held-out seed** 的 snapshot(與 training seed 分開)。
- 把 `trained_policy` / `wmmse` / `pf_wmmse` / `full_power` 全丟進同一個 eval harness,**同一批 channel** 上比。
- 指標:sum-rate(bps/Hz)、Jain、**optimality gap vs grid**、runtime,**5 seeds mean±std**。
- 額外:**z-probe ablation**(§4.5)。

### 6.4 提案結果表(填 TBD)
```
régime  | full_power | pf_wmmse | Ind-SAC | Flat-HASAC | C-HASAC(ours)
--------+------------+----------+---------+------------+--------------
R1      |    ...     |   ...    |   ...   |    ...     |   ...(≈ ceiling)
R2      |    ...     |   N/A*   |   ...   |    ...     |   ...(should win)
* WMMSE 在 R2 無 CSI 不能跑 → 這格的「不能跑」本身就是 RL 的賣點
```

---

## 7. 程式碼現況與檔案規格

### 7.1 已存在
- **`baselines.py`**(numpy-only,可獨立跑):
  - `metrics(P, As, assoc, cfg, nw)` ← **唯一真相來源**(sum_rate / per_bs_rate / jain / shaped reward)
  - channel:`make_snapshot()`(formula UMi,freq bug 已修;`freq_selective` flag)
  - baselines:`bl_full_power` / `bl_wmmse`(Sun 2018 SISO WMMSE)/ `bl_grid`
  - `run_eval()`:對照表 harness

### 7.2 給 AI 的實作指令(TODO,按順序)
> **不變量(每一步都要守)**:任何新的 actor / baseline 都**必須**呼叫 `baselines.py` 的 `metrics()` 算 reward 與指標,不可自己另寫 SINR/rate。channel snapshot 由 `make_snapshot()` 產生,training 與 eval 用不同 seed。

1. **`pf_wmmse`**:在 `baselines.py` 加 weighted WMMSE,weight `w_i = 1/(throughput_i + ε)`,跑數輪 outer loop 更新 weight。加入 `run_eval` 的 methods。
2. **sum-power 版**:加 `Cfg.constraint ∈ {"per_rb_cap","per_bs_sum"}`;sum 模式下 WMMSE 的 v-update 後對每個 BS 做 `v[i,:] *= min(1, sqrt(P_total/Σ_rb v[i,rb]²))` 投影。
3. **RL skeleton(PyTorch,先在 3-BS 驗證,不進 HARL)**:
   - `Encoder(f_θ)`:per-cell KPM `[N_BS, k]` → MLP → attention/mean pool → `z ∈ R^d`(d 設 8–16)。permutation-invariant。
   - `Actor(π_φ)`:`(o_i, z)` → SAC actor(tanh-squashed Gaussian),parameter-shared,obs 內含 agent-id one-hot。
   - `Critic(Q_ψ)`:`(share_obs, a_1..a_N)` → 雙 Q。
   - training loop:off-policy SAC,replay 存 `(global_KPM, {o_i}, {a_i}, r, next…)`;update 時**重算** `z=f_θ(global_KPM)` 讓梯度回傳 encoder;`z` 每 K=10 步更新一次。
   - reward 來自 `metrics()`(R2 régime:obs 只放 KPM proxy,不放 SINR)。
   - **z-probe**:eval 時提供 `z←0` 與 `z←shuffle` 兩個對照。
4. **obs builder**:依 `régime ∈ {R1,R2,R3}` 切換 worker obs 的內容(R1 給增益、R2 只給 KPM、R3 加 buffer 狀態)。
5. **(可選)HARL adapter**:寫薄 `FiveGEnv` wrapper,`step()` 內部呼叫 `metrics()`;確認與 standalone 結果一致再搬。

### 7.3 驗收條件
- `python baselines.py` 重現 §5.1 數量級(wmmse gap ~6%,full_power gap ~70%)。
- C-HASAC 在 **R2** 下 sum-rate 高於 Flat-HASAC,且 **z-probe 顯示拿掉 z 後明顯退化**(證明 z 有用)。
- 所有方法在同一批 held-out snapshot 上比較,指標含 optimality gap。

---

## 8. 一句話總結給隊友
我們把 motivation 從錯誤的「3GPP eICIC budget」改成正確的「O-RAN RIC↔DU via E2」;manager 從「發硬約束的獨立 SAC」改成「產生 learned latent z、end-to-end 訓的 permutation-invariant encoder」;baseline 從「只比 round-robin」升級成「WMMSE / PF-weighted WMMSE + grid 上界」;並且把賣點明確收斂到 **partial-obs(R2)場景下,用 z 做跨 cell 協調**——這是 WMMSE 做不到、而且 z-probe 可量化證明的東西。
