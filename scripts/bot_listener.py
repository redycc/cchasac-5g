"""
WinlabDLBot group listener.
- 所有 @WinlabDLBot mention → 呼叫 Claude CLI 動態回答
Run with: nohup python3 scripts/bot_listener.py > logs/bot_listener.log 2>&1 &
"""
import json
import os
import subprocess
import threading
import time
import logging
import requests

BOT_TOKEN    = "8647990177:AAFlXBKYJkNDEeUmOikNO1Y3XE_0MjXRZeU"
BOT_USERNAME = "winlabdlbot"
API          = f"https://api.telegram.org/bot{BOT_TOKEN}"
PROJECT_DIR  = "/home/hyc1014/DL/FinalProject"
CLAUDE_BIN   = "/home/hyc1014/.local/bin/claude"
TASKS_DIR    = "/home/hyc1014/DL/FinalProject/tasks"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Telegram API helpers ───────────────────────────────────────────────────────

def split_message(text: str, limit: int = 4000) -> list[str]:
    chunks, current, current_len = [], [], 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def send_message(chat_id: int, text: str, reply_to_id: int | None = None) -> None:
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_to_id:
        payload["reply_to_message_id"] = reply_to_id
    for chunk in split_message(text):
        try:
            r = requests.post(f"{API}/sendMessage",
                              json={**payload, "text": chunk}, timeout=15)
            data = r.json()
            if not data.get("ok"):
                log.error("sendMessage failed: %s", data)
        except Exception as e:
            log.error("sendMessage error: %s", e)
        payload.pop("reply_to_message_id", None)


def get_updates(offset: int) -> list:
    try:
        r = requests.get(
            f"{API}/getUpdates",
            params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
            timeout=35,
        )
        return r.json().get("result", [])
    except Exception as e:
        log.warning("getUpdates error: %s", e)
        return []


# ── Claude CLI handler ─────────────────────────────────────────────────────────

def handle_query(chat_id: int, message_id: int, question: str) -> None:
    """Route all mentions to Claude CLI for dynamic answers."""

    def _run():
        os.makedirs(TASKS_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        task_file = os.path.join(TASKS_DIR, f"task_{ts}.md")
        task_content = (
            f"# Telegram 群組問題\n\n"
            f"**時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## 問題\n\n{question}\n\n"
            f"## 執行要求\n\n"
            f"請閱讀 progress.md 了解目前實驗狀態，然後用繁體中文回答這個問題。\n"
            f"- 若是進度/結果查詢：直接分析並回答，不要只貼 progress.md 原文\n"
            f"- 若是實作建議：評估可行性，可行則直接實作並更新 progress.md\n"
            f"- 純回答（非實作）請控制在 400 字以內\n"
        )
        with open(task_file, "w", encoding="utf-8") as f:
            f.write(task_content)
        log.info("Task file written: %s", task_file)

        send_message(
            chat_id,
            f"🤖 Claude Code 正在處理...\n💬 <b>問題</b>：{question}",
            reply_to_id=message_id,
        )

        try:
            result = subprocess.run(
                [
                    CLAUDE_BIN,
                    "--print",
                    "--output-format", "json",
                    "--dangerously-skip-permissions",
                    f"請執行任務檔 {task_file} 的內容。",
                ],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                timeout=600,
            )
            try:
                data = json.loads(result.stdout)
                session_id = data.get("session_id", "")
                output     = (data.get("result") or "").strip()
            except json.JSONDecodeError:
                session_id = ""
                output     = result.stdout.strip() or result.stderr.strip()

            resume_hint = (
                f"\n\n📌 <code>claude --resume {session_id}</code>"
                if session_id else ""
            )
            send_message(
                chat_id,
                f"✅ <b>Claude Code 回覆</b>{resume_hint}\n\n{output or '(無輸出)'}",
            )
            log.info("query done, session_id=%s rc=%d", session_id, result.returncode)
        except subprocess.TimeoutExpired:
            send_message(chat_id, "⏰ Claude Code 超時（10 分鐘），請手動確認。")
        except Exception as e:
            send_message(chat_id, f"❌ 執行失敗：{e}")
            log.error("claude subprocess error: %s", e)

    threading.Thread(target=_run, daemon=True).start()


# ── 主迴圈 ────────────────────────────────────────────────────────────────────

def get_mention_text(message: dict) -> str | None:
    """Return message text with the @BotUsername mention stripped, or None."""
    entities = message.get("entities", [])
    text = message.get("text", "")
    for entity in entities:
        if entity.get("type") == "mention":
            offset, length = entity["offset"], entity["length"]
            mention = text[offset:offset + length].lstrip("@").lower()
            if mention == BOT_USERNAME:
                cleaned = (text[:offset] + text[offset + length:]).strip()
                return cleaned
    return None


def main() -> None:
    log.info("WinlabDLBot listener started (@%s)", BOT_USERNAME)
    offset = 0
    while True:
        updates = get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message", {})
            if not message:
                continue

            text = get_mention_text(message)
            if text is None:
                continue  # not @mentioned

            chat_id    = message["chat"]["id"]
            message_id = message["message_id"]
            log.info("query text=%r", text[:80])
            handle_query(chat_id, message_id, text)

        if not updates:
            time.sleep(1)


if __name__ == "__main__":
    main()
