# 交接文件:C-HASAC 實作(給有 GPU 的執行端 Claude)

> 目標讀者:**有 GPU、要把整個專案從頭跑完的 coding agent**。
> 任務:建環境 → 訓練 **vanilla HASAC baseline** → 加 latent context **z 成 C-HASAC** → 跑 baselines 與 ablations → 產出結果表。
> 專案性質:DL 課程 final project。**主要目標只有一個:C-HASAC 要贏 vanilla HASAC**,而且兩者唯一差別是 actor 有沒有吃 z(隔離貢獻)。

---

## 0. TL;DR(先讀這段)

1. 場景:**multi-cell, reuse-1(同頻), 有 cell-edge UE** 的下行功率分配。不這樣就沒有 inter-cell 協調問題,整個方法沒意義。
2. **三層資訊(最重要的設計原則)**:
   - **(A) BS-local 可觀測** → actor 的 `o_i`(自身 UE 的 CQI/rate、自身 load、PF 權重)。
   - **(B) RIC 可觀測的 KPM** → 餵 encoder 產生 `z`(各 cell 的 load/throughput/PRB,**不是 CSI**)。
   - **(C) sim-only 特權資訊** → 只給 **critic 與 reward**(完整 CSI、反事實),**訓練時用、執行時丟掉**。
   這就是 CTDE 的本質:**observation 受限,critic/reward 可特權**。
3. **Reward = difference reward**(偷自 Nasir-Guo,但只當訓練訊號):`自己賺的 − 害鄰居的`,反事實在 sim 算。PF-weighted。
4. **演算法:HASAC(off-policy SAC,HARL 家)**,三個 loss:critic / actor / temperature α。centralized critic + parameter sharing。
5. **動作:one-shot per-slot,permutation-equivariant head**(對自己 UE 集合一次輸出分配)。**不要做 autoregressive 解碼**(那是論文階段的延伸,final project 不需要,會拖慢)。
6. **régime 主打 partial-obs**:actor 只看 (A)+z;**full-CSI 只用在 WMMSE 天花板對照**。
7. baseline:**vanilla HASAC(無 z)= 主要對手**;**PF-weighted WMMSE = full-CSI 天花板(逼近,不期待贏)**;full-power = floor。
8. **絕對不要比 Nasir-Guo**:它的 state 需要顯式 neighbor CSI 交換,違反我們的 deployment-observable line。我們偷它的 **reward**,不抄它的 **state**。

---

## 1. 問題與核心論點

多個 cell 共用會互相干擾的頻譜(reuse-1),要分配各 cell 對其 UE 的發射功率,最大化 **PF-weighted 效用**。RIC 只看得到粗 KPM(無 cross-channel CSI),所以不能跑需要完整瞬時 CSI 的 WMMSE。

**核心論點(要證的)**:用 **deployment-observable 的資訊 + 一個 learned 的跨 cell context z**,在 partial-obs 下達到接近 full-CSI WMMSE 的協調效果,且**贏過沒有 z 的 HASAC**。

---

## 2. 不可違反的設計原則(每一步都要守)

1. **Single source of truth**:`channel → SINR → rate → reward` 只有一份函數,baseline / HASAC / C-HASAC / eval 全部呼叫它。若有上傳 `baselines.py`,重用其 channel 與 rate 計算;否則照 §3 自建。
2. **三層資訊嚴格分流**(§0 第 2 點):actor 永遠**只**吃 (A)+z;critic 與 reward 可用 (C) 特權。
3. **不要把 neighbor CSI / 反事實放進 observation**(那是 reward 用的,訓練時 sim 算)。
4. **不要 raw sum-rate**:目標一律 **PF-weighted**,否則最佳解退化成「全給 channel 最好的 UE」。
5. **不要雙重計算干擾**:`rate` 裡的 SINR 已含干擾,reward 不要再單獨減 interference 項。
6. **z 不餵 critic**:critic 用 `share_obs`(已含全域),z 只給 actor。
7. **HASAC 與 C-HASAC 唯一差別 = actor 有沒有吃 z**;reward / critic / 演算法 / 超參完全一致。

---

## 3. 環境規格(sim)

### 3.1 場景
- `N_BS` 個 base station(預設 3,要能設 6)。每個 BS 服務一組 UE,**UE 數可不同**(預設每 BS 隨機 3–6 個,總 `N_UE`)。
- 2D 區域(預設 500m×500m)。BS 與 UE 隨機佈點;**確保有 cell-edge UE**(離服務 BS 遠、受鄰區強干擾)。可選:用真實 BS 拓樸(OpenCelliD)取代隨機佈點以提升真實度(非必要)。
- reuse-1:所有 BS 同頻,互相干擾。
- **per-BS sum-power 約束**:`Σ_u p_{i,u} ≤ P_max`(這同時避免 RB 退化問題)。

### 3.2 通道(frequency-flat 即可,可選加 fading)
```
PL_dB(d) = 32.4 + 21*log10(d_m) + 20*log10(3.5)   # 3.5 GHz, d in metres
g[j,u] = 10^(-(PL + shadow~N(0,4dB))/10)            # BS j -> UE u 的 power gain
```
- 可選 mobility:UE random walk(讓通道時間自相關)。**注意:full-buffer 下物理時間仍是 bandit,mobility 只讓 R2 的歷史推斷有意義,不改 reward。**

### 3.3 SINR / rate(per-UE)
UE `u` 由 BS `i` 服務,功率 `p_{i,u}`:
```
SINR_u = (p_{i,u} * g[i,u]) / ( Σ_{j≠i} P_j_at_u * g[j,u] + N0 )
   其中 P_j_at_u = BS j 在 u 所用資源上的總干擾功率(同頻 reuse-1 下可取 BS j 對 u 的總發射功率)
rate_u = log2(1 + SINR_u)        # bps/Hz
N0 = 每 RB 熱雜訊 ≈ dBm_to_W(-174 + 10*log10(180e3))
P_max = dBm_to_W(30)
```

### 3.4 PF 權重(carry-over,但用 weight trick 處理)
```
w_u = 1 / (R̄_u + ε)              # PF 權重
R̄_u ← (1-β) R̄_u + β * rate_u     # running average, β≈0.01
```
`R̄` 是 slowly-varying 的權重,**不放進 RL state 的 carry-over**(用 weight trick 貪婪處理)。

### 3.5 Reward = difference reward(訓練訊號,sim 用全資訊算)
```
# actual:所有 BS 用其動作 → 算 rate_u, C_u = w_u * rate_u (all u)
# 對每個 BS i 做反事實:把 BS i 功率全歸零,重算「不屬於 i 的 UE」的 rate
harm_i = Σ_{u ∉ i} w_u * ( rate_u^{(i靜音)} - rate_u^{(actual)} )   # ≥ 0,i 造成的傷害
r_i    = Σ_{u ∈ i} w_u * rate_u^{(actual)}  -  harm_i               # 自己賺的 - 害人的
```
- 每步要 `N_BS` 次反事實重算(把每個 BS 輪流歸零),很便宜。
- **這是訓練訊號,執行時不需要**(符合 deployment line)。
- **替代/ablation**:team reward `r_team = Σ_{all u} w_u rate_u`(所有 BS 共用)。先用 difference reward,之後 ablate team vs difference。

### 3.6 (可選,論文階段)queue + goodput 模式
加 per-UE buffer `Q_u`(`Q ← max(Q-rate,0)+arrival`),reward 改 `Σ w_u rate_u − λ Σ Q_u`,評測改 goodput + p99 delay。**final project 預設 full-buffer,先不做這個。**

---

## 4. 觀測規格(三層,務必分清)

```
# (A) actor 的 per-BS 本地觀測 o_i —— 變長 UE 集合,deployment-observable
o_i = { 每個 UE u∈i: [ achievable_rate_u(或 CQI), w_u, 上一步 p_{i,u} ] }
      + BS 級: [ n_ue_i, 上一步自身 throughput ]
# 不含任何鄰居/cross-cell 資訊。

# (B) encoder 輸入 —— RIC 可觀測 KPM(各 cell 一列),deployment-observable
kpm_c = [ load_c(=n_ue 或 RRC.ConnMean proxy), throughput_c, prb_util_c ]  for c in all cells
# encoder 對 cell 集合做 permutation-invariant pooling → z

# (C) critic 的 share_obs —— sim 特權,訓練時用、執行丟掉
share_obs = 全域真實狀態:所有 g[j,u]、所有 p、所有 UE 的 w/rate
```

---

## 5. 模型規格

### 5.1 Actor(per-BS,跨 BS **parameter sharing**,permutation-equivariant)
```
輸入: o_i (變長 UE 集合) [+ z (C-HASAC 才有)]
流程: 每 UE 過 shared MLP → per-UE embedding
      (可選) 幾層 self-attention over UE 集合 (permutation-equivariant)
      [z 與 BS 級特徵 broadcast concat 到每個 UE embedding]
      → 每 UE 一個 power logit
輸出(SAC,連續、隨機): 對 logits 出 squashed Gaussian (reparameterized)
      → 經 per-BS sum-power 投影 (e.g. softmax over UE * P_max,或 sigmoid 後若超 budget 等比縮放)
      → p_{i,u}
entropy: 對動作分布算,供 SAC 用
```
- 變長 UE / 不同 finish time **不是問題**:set-based head 天然吃變長,batch 時 pad+mask。
- **HASAC = 此 actor 不吃 z;C-HASAC = 吃 z。其餘全同。**

### 5.2 Critic(centralized,CTDE,sim 特權)
```
Q(share_obs, joint_action) → scalar       # 雙 Q (twin) + target networks
```
- `N_BS`、`N_UE` 若每集固定 → 直接 concat 成定長向量最簡單(**final project 建議固定**)。
- 若要變長 → critic 也用 set/attention encoder over (BS,UE) entities → pooled → Q。
- **critic 不吃 z**(已有 share_obs)。

### 5.3 Encoder f_θ(只 C-HASAC 有)
```
z = f_θ( {kpm_c : all cells} )
架構: 每 cell 過 MLP → attention/mean pooling (permutation-invariant) → z ∈ R^d  (d=8~16)
可選: 用 GNN,edge = inter-BS 距離 (geometry-aware);距離由 config/BS 座標來 (deployable)
更新: z 每 K 步重算一次、中間 hold (K=1 先,之後可試 K=10);off-policy 時存 kpm、更新時重算 z 讓梯度回傳
```

### 5.4 Temperature α
- 自動 entropy tuning,target entropy `H̄ = -dim(action)`(或 per-BS)。

---

## 6. 演算法:HASAC(off-policy)

可在官方 **HARL repo**(Zhong et al., `github.com/PKU-MARL/HARL`)上實作,或**從頭乾淨實作**(losses 標準,推薦後者較可控)。三個 loss:

```
Critic:  L_Q = E[ (Q(s,a) − y)² ],
         y = r + γ ( min_j Q_target_j(s', a') − α log π(a'|o') ),  a' ~ π
Actor:   L_π = E[ α log π(a|o[,z]) − Q(s, a) ]            # max Q + α·H
Temp:    L_α = E[ −α ( log π(a|o) + H̄ ) ]
```
- `s` = share_obs(critic),`o` = 本地觀測 [+z](actor)。**asymmetric actor-critic**。
- off-policy replay、target network、double-Q、polyak τ。
- multi-agent:parameter sharing(同一 actor 套到各 BS);CTDE centralized critic。sequential update(HAML)可做可不做,parameter sharing 下先用**同時更新**即可,要更貼 HASAC 再加 sequential。
- **off-policy 的理由**:UE 多、sim 取樣貴 → replay 重用更 sample-efficient。

### 6.1 C-HASAC 的 z 梯度流
```
z = f_θ(kpm)                          # 重算 (不存 stale z)
a_i = π_φ(o_i, z)
L_π 對 θ 的梯度經由 z 回傳 → 訓練 encoder (不需 manager reward)
(可選) aux: 小權重 reconstruction(decode z → 預測 kpm)防 z-collapse, β_aux≈0.01
```

---

## 7. Baselines(全部接同一個 §3 objective,同一批 held-out seed 評測)

| baseline | 資訊 | 角色 | 必做 |
|---|---|---|---|
| **full_power** | — | floor | ✅ |
| **PF (proportional fair)** | local CQI | 部署 default | ✅ |
| **vanilla HASAC** | (A) deployable,**無 z** | **主要對手** | ✅✅ |
| **C-HASAC (ours)** | (A)+z | 我們的方法 | ✅✅ |
| **PF-weighted WMMSE** | **full CSI** | **天花板(逼近,不期待贏)** | ✅ |
| grid-optimal(小規模) | full CSI | 絕對上界(量 gap) | 選做 |

WMMSE 用 §3 的 PF 權重當 weighted-sum-rate 的權重(= MaxWeight 內層用 WMMSE 解);full-CSI,明確標注「資訊比我們多,是 target 不是 peer」。

---

## 8. 實驗協議

### 8.1 訓練
- 固定 `N_BS`、UE 佈點分布;每個 episode 重抽 channel(full-buffer → 物理時間 bandit,預期且 OK)。
- 先把 **vanilla HASAC 訓到收斂**,再用**同超參**訓 C-HASAC。

### 8.2 評測
- 一批 **held-out seeds**(與訓練 seed 分開),所有方法跑**同一批 channel snapshot/軌跡**(common random numbers,配對)。
- 指標:**PF-weighted utility(主)**、Jain fairness、**gap to PF-WMMSE ceiling**、(若有 WMMSE 對 full-CSI 的 runtime)。5+ seeds,報 mean ± 95% CI。

### 8.3 主結果與 ablation(按重要性)
1. **C-HASAC vs vanilla HASAC**(唯一差別 z)→ 主結果,贏了就是 z 的功勞。
2. **z-probe**:C-HASAC 推論時把 `z←0`(或 shuffle),看掉多少 → 證明 z 真的在做事。
3. **team reward vs difference reward**。
4. **γ ablation(物理時間,0 vs 0.99)**:full-buffer 下**預期沒差**(確認 bandit,by design,當誠實附註)。
5. (選)encoder 有無 geometry edge;N_BS 從 3→6 的 scalability。

---

## 9. 預設超參(可直接用,GPU 一張中階卡即可)

```
γ = 0.99            # 即使 bandit 也設 0.99,跑 §8.3-4 ablation
τ (polyak) = 0.005
actor lr = 3e-4 ; critic lr = 3e-4 ; alpha lr = 3e-4
batch = 256 ; replay = 1e6 ; warmup steps = 5000
hidden = 256 (MLP) ; attention heads = 4 ; z_dim = 16
target entropy = -action_dim
train steps = 5e5 (先 1e5 驗證 pipeline)
update freq = 每 env step 更新 1 次 ; double-Q
seeds = {0,1,2,3,4}
```

---

## 10. 給執行端的 ordered TODO + 驗收條件

> 每一步做完先自測再進下一步。任何 reward/指標都呼叫同一個 objective 函數。

1. **環境**:照 §3 實作 `step()`(channel→SINR→rate→difference reward),含反事實計算與 PF 權重更新。
   - 驗收:full_power 與 PF-WMMSE 的 PF-utility 數量級合理(WMMSE 明顯優於 full_power);反事實 `harm_i ≥ 0`。
2. **Baselines**:full_power、PF、PF-weighted WMMSE(full CSI)。
   - 驗收:WMMSE > PF > full_power(在 PF-utility 上)。
3. **vanilla HASAC**:§5.1 actor(無 z)+ §5.2 critic + §6 三 loss。
   - 驗收:訓練收斂,PF-utility 穩定超過 PF baseline,逐步逼近 WMMSE。
4. **C-HASAC**:加 §5.3 encoder,actor 改吃 z,§6.1 梯度流。
   - 驗收:**C-HASAC > vanilla HASAC**(同超參、同 seed);**z-probe 顯示拿掉 z 明顯退化**。
5. **Ablations**:§8.3 全跑。
6. **產出**:結果表(§8.2 指標,mean±CI)+ 訓練曲線 + z-probe 圖 + 一份簡短 REPORT.md 說明每個結論。

### 最終要回答的一句話
> 「在 deployment-observable 的限制下(actor 只看本地 + RIC 的 KPM-derived z,無 cross-cell CSI),**多吃一個 learned 全域 context z 的 C-HASAC,是否贏過沒有 z 的 HASAC,並逼近 full-CSI 的 PF-WMMSE 天花板?**」

---

## 11. 明確「不要做」清單

- ❌ 不要把 neighbor/cross-cell CSI 放進 actor 的 observation(違反 deployment line;那是 reward 在 sim 算的)。
- ❌ 不要用 raw sum-rate 當目標(要 PF-weighted)。
- ❌ 不要在 reward 裡再單獨減 interference(已含在 rate)。
- ❌ 不要把 z 餵 critic。
- ❌ 不要做 autoregressive-over-UE 解碼(final project 用 one-shot equivariant head)。
- ❌ 不要拿 Nasir-Guo 當比較對象(它 input 需要額外蒐集,不可部署);只偷它的 difference reward。
- ❌ 不要期待贏 full-CSI WMMSE(它是天花板;贏的對象是 vanilla HASAC)。
