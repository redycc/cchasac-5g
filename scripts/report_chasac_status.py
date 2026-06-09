"""Analyze current CHASAC/HASAC runs and optionally send a Telegram summary."""
import argparse
import os
import re
from dataclasses import dataclass

from telegram_utils import send_message


PROJECT_DIR = "/home/hyc1014/DL/FinalProject"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

HEADER_RE = re.compile(r"use_z=(\d+).*reward=([A-Za-z0-9_+-]+).*bc_steps=(\d+)")
BC_RE = re.compile(r"\[BC\] it\s+(\d+)/(\d+)\s+\|\s+MSE\s+([0-9.]+)")
STEP_RE = re.compile(
    r"step\s+(\d+)\s+\|\s+PF-U\s+([-0-9.]+)\s+±\s+([-0-9.]+)\s+\|\s+alpha\s+([-0-9.]+)\s+\|\s+"
    r"pwr\s+([-0-9.]+)\s+\|\s+best\s+([-0-9.]+)"
)


@dataclass
class Run:
    tag: str
    label: str
    path: str


def parse_log(path: str) -> dict:
    info = {
        "exists": os.path.exists(path),
        "header": None,
        "bc": None,
        "latest": None,
        "mtime": os.path.getmtime(path) if os.path.exists(path) else None,
    }
    if not info["exists"]:
        return info
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if info["header"] is None:
                m = HEADER_RE.search(line)
                if m:
                    info["header"] = {
                        "use_z": int(m.group(1)),
                        "reward": m.group(2),
                        "bc_steps": int(m.group(3)),
                    }
            m = BC_RE.search(line)
            if m:
                info["bc"] = {
                    "last_it": int(m.group(1)),
                    "total": int(m.group(2)),
                    "mse": float(m.group(3)),
                }
                continue
            m = STEP_RE.search(line)
            if m:
                info["latest"] = {
                    "step": int(m.group(1)),
                    "pfu": float(m.group(2)),
                    "std": float(m.group(3)),
                    "alpha": float(m.group(4)),
                    "pwr": float(m.group(5)),
                    "best": float(m.group(6)),
                }
    return info


def build_report(run_a: Run, run_b: Run) -> str:
    parsed_a = parse_log(run_a.path)
    parsed_b = parse_log(run_b.path)
    lines = ["【Codex 分析】目前 CHASAC/HASAC run 狀態"]
    for run, parsed in ((run_a, parsed_a), (run_b, parsed_b)):
        if not parsed["exists"]:
            lines.append(f"- {run.label}: log 不存在")
            continue
        header = parsed["header"] or {}
        bc = parsed["bc"]
        latest = parsed["latest"]
        if latest is None:
            if bc is None:
                lines.append(f"- {run.label}: log 已存在，但還沒進到可解析 step")
            else:
                lines.append(
                    f"- {run.label}: BC {bc['last_it']}/{bc['total']} | MSE {bc['mse']:.4f}"
                )
            continue
        lines.append(
            f"- {run.label}: use_z={header.get('use_z', '?')} reward={header.get('reward', '?')} "
            f"BC={header.get('bc_steps', '?')} | step {latest['step']} | PF-U {latest['pfu']:.3f} ± {latest['std']:.3f} | "
            f"best {latest['best']:.3f} | pwr {latest['pwr']:.3f} | alpha {latest['alpha']:.4f}"
        )

    a = parsed_a["latest"]
    b = parsed_b["latest"]
    if a and b:
        if a["step"] == b["step"]:
            diff = a["pfu"] - b["pfu"]
            if abs(diff) < 1e-6:
                lines.append(f"同一步數 {a['step']} 目前幾乎平手。")
            elif diff > 0:
                lines.append(f"同一步數 {a['step']}，{run_a.label} 暫時領先 {abs(diff):.3f} PF-U。")
            else:
                lines.append(f"同一步數 {a['step']}，{run_b.label} 暫時領先 {abs(diff):.3f} PF-U。")
        lines.append(
            f"目前 best 對照：{run_a.label} {a['best']:.3f} vs {run_b.label} {b['best']:.3f}。"
        )
        if a["best"] > b["best"]:
            lines.append(
                f"初步判讀：有 z 的 run 曾摸到更高 peak，但若最新 PF-U 仍明顯震盪，就還不能當成穩定勝出。"
            )
        else:
            lines.append(
                f"初步判讀：目前無 z 的 run 並沒有被拉開，這輪重點要看後續是否出現穩定的 z 優勢。"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", default="chasac_z1_logpf_bc")
    ap.add_argument("--label-a", default="z1/C-HASAC")
    ap.add_argument("--tag-b", default="hasac_z0_logpf_bc")
    ap.add_argument("--label-b", default="z0/HASAC")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stdout-only", action="store_true")
    args = ap.parse_args()

    run_a = Run(args.tag_a, args.label_a, os.path.join(RESULTS_DIR, f"{args.tag_a}_log.txt"))
    run_b = Run(args.tag_b, args.label_b, os.path.join(RESULTS_DIR, f"{args.tag_b}_log.txt"))
    text = build_report(run_a, run_b)
    if args.stdout_only or args.dry_run:
        print(text)
        if args.dry_run:
            print("---")
        return
    send_message(text, dry_run=False)


if __name__ == "__main__":
    main()
