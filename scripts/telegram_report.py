"""Send progress.md to Telegram group."""
import re
import requests

BOT_TOKEN = "8647990177:AAFlXBKYJkNDEeUmOikNO1Y3XE_0MjXRZeU"
CHAT_ID   = "-1003883274003"
PROGRESS  = "/home/hyc1014/DL/FinalProject/progress.md"


def md_to_html(text: str) -> str:
    lines, out = text.splitlines(), []
    for line in lines:
        # headings
        if line.startswith("### "):
            line = f"<b>{line[4:]}</b>"
        elif line.startswith("## "):
            line = f"\n<b>— {line[3:]} —</b>"
        elif line.startswith("# "):
            line = f"<b>{line[2:]}</b>"
        # horizontal rule
        elif re.fullmatch(r"-{3,}", line.strip()):
            line = "──────────────────────"
        # bold **text**
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        # strip markdown table rows (just keep as-is plain text)
        out.append(line)
    return "\n".join(out)


def split_message(text, limit=4000):
    """Split text into chunks at newline boundaries within limit."""
    chunks, current = [], []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def send():
    with open(PROGRESS, encoding="utf-8") as f:
        raw = f.read()

    text   = md_to_html(raw)
    url    = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    chunks = split_message(text)
    for i, chunk in enumerate(chunks):
        resp = requests.post(url, json={
            "chat_id":    CHAT_ID,
            "text":       chunk,
            "parse_mode": "HTML",
        }, timeout=15)
        data = resp.json()
        if data.get("ok"):
            print(f"✅ 訊息 {i+1}/{len(chunks)} 已送出")
        else:
            print(f"❌ 第 {i+1} 段失敗：{data}")
            break


if __name__ == "__main__":
    send()
