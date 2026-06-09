# FinalProject Repo Map

## Main Line

這個 repo 目前真正主線不是早期 `H-HASAC`，而是較新的 **C-HASAC / vanilla HASAC** 對照：

- `env_chasac.py`
  - 現在最重要的環境真相來源。
  - 定義 scenario、SINR/rate、PF utility、difference/team/logpf reward。
  - 也內建 `equal_power` floor 與 `pf_wsr_ceiling` full-CSI ceiling。
- `scripts/train_chasac.py`
  - 乾淨版訓練入口。
  - `--use_z 0` = vanilla HASAC。
  - `--use_z 1` = C-HASAC。
  - 兩者唯一差別是 actor 有沒有吃 encoder 產生的 `z`。
- `scripts/monitor_chasac_runs.py`
  - 監控兩個指定 run 的 log，定時送 Telegram 摘要與警示。
- `results/`
  - 目前所有 run 的 log、stdout、checkpoint 輸出集中在這裡。
- `progress.md`
  - 歷史最完整，但混了多條實驗線。讀它要分辨時間點與實驗環境。

## Historical Lines

- `scripts/train_cc_hasac_v*.py`
  - Claude 早期大量試驗版本。
  - 主要在不同 reward、BC pretrain、encoder freeze、alpha 設定間反覆試。
  - 有研究價值，但不是現在最乾淨的主程式。
- `scripts/train_cc_hasac_goodput_v*.py`
  - 動態環境 / goodput 路線。
  - 方向是讓 RL 在真正 sequential 設定中比較有發揮空間。
- `scripts/train_h_hasac*.py`
  - 更舊的 hierarchical H-HASAC 主線。
  - 對應舊報告與 `HARL` 架構。

## Supporting Code

- `baseline.py`
  - 舊線的 baseline / WMMSE / grid-opt 共享計算核心。
  - 重點概念是 single source of truth。
- `envs/`
  - 舊環境集合，包含 `cc_env*`、`fiveg_env.py` 等。
- `HARL/`
  - 外部/改造過的 HARL framework。
- `H-HASAC/`
  - 更早期專案目錄。

## Telegram / Automation

- `scripts/bot_listener.py`
  - 群組 bot listener，會把 mention 派給 Claude/Codex。
- `scripts/telegram_report.py`
  - 把 `progress.md` 直接送到 Telegram。
- `scripts/telegram_utils.py`
  - 共用 Telegram 發送工具。
- `scripts/report_chasac_status.py`
  - 新增的即時分析腳本：直接讀兩個 run 的 log，生成短分析並送 Telegram。

## What To Trust Most

如果目標是判斷「現在這輪 C-HASAC vs HASAC 到底有沒有學到 `z` 的真實貢獻」，優先看這些：

1. `env_chasac.py`
2. `scripts/train_chasac.py`
3. `results/chasac_z1_*` / `results/hasac_z0_*`
4. `scripts/monitor_chasac_runs.py`
5. `progress.md` 中 2026-06-06 之後、明確標示 `env_chasac` / `train_chasac.py` 的段落
