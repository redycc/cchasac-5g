# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## 專案現況（2026-06）

DL Final Project：**C-HASAC** — Contextual Heterogeneous-Agent SAC，用 learned latent context z 讓去中心化 BS 在 5G 多基站功率分配中協調。

**核心命題**：C-HASAC（actor 吃 z）是否贏過 vanilla HASAC（actor 不吃 z），唯一差別 = actor 有沒有吃 encoder 學出的 z。

> **注意**：早期的 H-HASAC（Hierarchical）已放棄。現在的主線是 `env_chasac.py` + `scripts/train_chasac.py`。

---

## HANDOFF 設計原則（不可違反）

參考 `HANDOFF_CHASAC_IMPL.md` 的完整版，以下是最核心的限制：

### 三層資訊嚴格分流
| 層 | 內容 | 誰可以用 |
|----|------|---------|
| **(A) BS-local 可觀測** | per-UE rate/PF_weight/power（+可選 RSRP_neighbor） | actor obs |
| **(B) RIC-observable KPM** | 各 cell 的 load/throughput/P_bs（+可選 BS 間距離） | encoder → z → actor |
| **(C) sim-only 特權** | 完整 CSI、反事實 harm_i、全域 g matrix | critic + reward（訓練時）|

**z 不餵 critic**，critic 只吃 share_obs（已含特權資訊）。
**不把 neighbor CSI 放進 actor obs**（RSRP_neighbor = 自家 channel gain 給全 BS，合規）。

### Deployment Line
- actor obs 只用在 O-RAN gNB 真實能看到的資訊
- z 的來源（KPM）在 O-RAN 中由 RIC xApp 透過 E2 介面取得並下發
- **絕對不要比 Nasir-Guo**（它用顯式 neighbor CSI 交換，違反 deployment line）

---

## 主要檔案

| 檔案 | 用途 |
|------|------|
| `env_chasac.py` | **唯一真相**：channel→SINR→rate→reward，obs_share/obs_kpm/obs_local |
| `scripts/train_chasac.py` | 訓練主腳本：`--use_z 0/1` 切 HASAC/C-HASAC，`--use_rsrp 0/1` 切是否加 RSRP |
| `progress.md` | **單一實驗紀錄**，每次實驗/改動後必須更新 |
| `HANDOFF_CHASAC_IMPL.md` | 完整設計文件，有疑問先查這裡 |
| `results/` | 訓練 log（`*_log.txt`）、numpy 結果（`*.npy`） |
| `tasks/` | bot 任務佇列、`disable_subprocess.flag`、`bot_offset.txt` |
| `logs/bot_listener.log` | Telegram bot 活動 log |

### 重要維度（2026-06 當前）
- `ue_feat` = 3（無 RSRP）或 6（`--use_rsrp 1`，+g[i][u] for all BSes）
- `kpm_dim` = 5（3 KPM + 2 BS間距離，`N_BS=3`）
- `share_dim` = 63（g×36 + p×12 + serv×12 + bs_dist×3）
- `z_dim` = 16（encoder 輸出）

---

## 訓練指令

```bash
# vanilla HASAC（無 z，baseline）
python3 scripts/train_chasac.py --use_z 0 --reward logpf --bc_steps 1000 \
  --mu_bound 5 --warmup 1000 --steps 200000 --tag hasac_z0

# C-HASAC（有 z）
python3 scripts/train_chasac.py --use_z 1 --reward logpf --bc_steps 1000 \
  --mu_bound 5 --warmup 1000 --steps 200000 --tag chasac_z1

# C-HASAC + RSRP_neighbor
python3 scripts/train_chasac.py --use_z 1 --use_rsrp 1 --reward logpf \
  --bc_steps 1000 --mu_bound 5 --warmup 1000 --steps 200000 --tag chasac_rsrp

# 背景執行（推薦）
nohup python3 scripts/train_chasac.py [args] > results/run_tag.txt 2>&1 &
```

**重要超參**：
- `--reward logpf`：目前最佳 reward（potential-based ΔΣlog(R̄_u+ε)），避免 difference reward 的靜默局部解
- `--mu_bound 5`：防止 SAC tanh 飽和崩潰到 −165
- `--bc_steps 1000`：輕量 BC warm-start，BC 才打開「z 使用開關」
- `--warmup 1000`：actor 前 1000 步隨機探索

---

## Eval 指標

- **PF-U**（canonical）= `Σ_u log(R̄_u + 1e-6)`，20-ep held-out scenarios
- **z-ablation**：
  - `drop_zero` = policy − policy_z←0（z 歸零後掉多少）
  - `drop_shuffle` = policy − policy_z←shuffle（z 換成別 episode 的 z，比 drop_zero 嚴格）
  - **Tim 提醒**：drop_zero 可能高估 z 使用量（z 變常數 offset 時 drop_zero 大但 drop_shuffle≈0）
- **floor** = equal_power ≈ −5.33
- **ceiling** = PF-WSR（full-CSI） ≈ +23.5

每次 run 結束的 FINAL 區塊會同時報 drop_zero 和 drop_shuffle。

---

## Telegram 群組操作

### Bot 架構
- `scripts/bot_listener.py`（PID 存在 `logs/bot_listener.log`）：監聽 @WinlabDLBot 的群組訊息
- `tasks/disable_subprocess.flag`：**存在時** bot 把訊息 queue 到 `tasks/incoming.log`，由主 session 接管
- `tasks/bot_offset.txt`：bot 的 Telegram update offset，重啟時讀取避免重複處理舊訊息

### 正常模式（bot 自動跑 Claude subprocess）
```bash
# 啟動/重啟 bot
nohup python3 scripts/bot_listener.py >> logs/bot_listener.log 2>&1 &

# 監控 bot 活動（在主 session 用 Monitor tool）
# command: tail -f logs/bot_listener.log | grep --line-buffered -E "query|Claude done|WARNING|ERROR"
```

### 主 session 接管模式（避免重複回覆）
當主 session 要直接回答群組時，先設 flag 避免 subprocess 同時回：
```bash
touch tasks/disable_subprocess.flag
```
接管後用 Monitor 監看 incoming.log：
```
# Monitor command:
tail -f tasks/incoming.log
```
解除接管：`rm tasks/disable_subprocess.flag`

### 發訊息到群組
```python
import sys; sys.path.insert(0, 'scripts')
from telegram_utils import send_message
send_message('<b>訊息</b>', parse_mode='HTML')
```

### 快速 status report
```bash
python3 scripts/telegram_report.py   # 發 progress.md（格式化）
```

---

## 訓練 Monitor（重要習慣）

每次啟動訓練後，立即用 Monitor tool 掛上去：

```
# 監控訓練 log（Monitor tool command）
tail -f results/YOUR_RUN_log.txt | grep --line-buffered -E "step|FINAL|NaN|Error|Killed"

# 同時監控多個 run
tail -f results/run_a_log.txt results/run_b_log.txt | grep --line-buffered -E "step|FINAL|NaN|Error"
```

---

## 良好習慣（使用者明確要求）

1. **progress.md 只記錄實驗結果**，不記系統維護類變更（重啟 bot、調 flag 等）
2. **實驗跑完結果自動發群組**，不等使用者提醒
3. **群組發摘要而非全文**：結果數字 + 關鍵觀察 + 下一步，控制在可讀長度
4. **修改 script 後必須更新 progress.md**（stop hook 會擋住）
5. **z←shuffle 和 z←0 都要報**：只報 z←0 可能高估 z 的使用程度
6. **BC dataset cache** 有 `kpm_dim` 和 `use_rsrp` 驗證欄位，改架構後自動重建
7. **不要隨意刪 deepmimo_cache/**、`results/bc_dataset.npz`

---

## 已知坑（C-HASAC 路線）

### SAC tanh 飽和崩潰（−165）
action `a=tanh(x)`，`mu→−∞` 時 power→0 → log(0) → −165。`--mu_bound 5` 根除。

### pure-RL 不用 z（Tim 觀察，獨立驗證）
純 RL 訓練時 drop_shuffle≈0，BC 進來才打開「z 使用開關」。
→ 沒有 BC 的 run 不太可能看到 meaningful drop_shuffle。

### Q 後段 overestimation 導致 actor 崩潰
40k 前常見 peak，之後 oscillation。目前已知緩解：mu_bound + logpf + BC warm-start。
根本解尚未找到（SAC Q-overestimation 問題）。

### BC dataset cache shape mismatch
改 `ue_feat`（`--use_rsrp`）或 `kpm_dim` 後，cache 自動失效重建。
如果手動刪 `results/bc_dataset.npz` 也沒問題，下次自動重建。
