"""
Append one experiment result entry to progress.md.

Usage (from training scripts):
    from scripts.log_experiment import log_experiment
    log_experiment("H-HASAC v7", final_raw_ep=165.3, note="formula channel, beta=0.3")

CLI usage:
    python3 scripts/log_experiment.py --name "H-HASAC v7" --raw-ep 165.3 --note "formula"
"""
import argparse
import datetime
import os
import sys

PROGRESS = os.path.join(os.path.dirname(__file__), "..", "progress.md")


def log_experiment(name: str, final_raw_ep: float, note: str = ""):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"| {ts} | {name} | **{final_raw_ep:.1f}** | {note} |"

    with open(PROGRESS, "a", encoding="utf-8") as f:
        # If file doesn't end with newline, add one
        f.seek(0, 2)
        pos = f.tell()
        if pos > 0:
            f.seek(pos - 1)
            last = f.read(1)
            if last != "\n":
                f.write("\n")
        f.write(line + "\n")

    print(f"[log_experiment] Appended to progress.md: {line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--raw-ep", type=float, required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    log_experiment(args.name, args.raw_ep, args.note)
