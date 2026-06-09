#!/bin/bash
# monitor_bot.sh — 即時顯示 bot_listener 狀態 + subprocess 輸出
# Usage: bash scripts/monitor_bot.sh

LOG="logs/bot_listener.log"
PID_FILE=""

show_header() {
    clear
    echo "═══════════════════════════════════════════════════════"
    echo "  WinlabDLBot Monitor  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "═══════════════════════════════════════════════════════"
    BOT_PID=$(pgrep -f "bot_listener.py" | head -1)
    if [ -n "$BOT_PID" ]; then
        echo "  bot_listener: ✅ ALIVE  (PID $BOT_PID)"
    else
        echo "  bot_listener: ❌ DEAD   → 重啟: nohup python3 scripts/bot_listener.py >> logs/bot_listener.log 2>&1 &"
    fi
    SUB_PID=$(pgrep -f "claude.*--resume\|claude.*task_" | head -1)
    if [ -n "$SUB_PID" ]; then
        echo "  subprocess  : 🔵 RUNNING (PID $SUB_PID)"
    else
        echo "  subprocess  : ⬜ idle"
    fi
    echo "───────────────────────────────────────────────────────"
    echo "  [最後 40 行 log — 持續更新，Ctrl+C 退出]"
    echo "═══════════════════════════════════════════════════════"
}

# 初次顯示 header，然後 tail -f 追蹤 log
show_header
tail -n 40 -f "$LOG"
