"""Send progress.md to Telegram group.

Set environment variables before running:
    export TG_BOT_TOKEN="<your-bot-token>"
    export TG_CHAT_ID="<your-group-chat-id>"
"""
import os
import re
import requests

BOT_TOKEN = os.environ["TG_BOT_TOKEN"]
CHAT_ID   = os.environ["TG_CHAT_ID"]
PROGRESS  = os.path.join(os.path.dirname(__file__), "..", "progress.md")


def md_to_html(text: str) -> str:
    lines, out = text.splitlines(), []
    for line in lines:
        if line.startswith("### "):
            line = f"<b>{line[4:]}</b>"
        elif line.startswith("## "):
            line = f"\n<b>— {line[3:]} —</b>"
        elif line.startswith("# "):
            line = f"<b>{line[2:]}</b>"
        elif re.fullmatch(r"-{3,}", line.strip()):
            line = "──────────────────────"
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        out.append(line)
    return "\n".join(out)


def send():
    with open(PROGRESS, encoding="utf-8") as f:
        raw = f.read()
    text = md_to_html(raw)
    resp = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
        timeout=15,
    )
    data = resp.json()
    if data.get("ok"):
        print("✅ 訊息已送出")
    else:
        print(f"❌ 失敗：{data}")


if __name__ == "__main__":
    send()
