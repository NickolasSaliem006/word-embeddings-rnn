"""Provided entry point: run the two completed student tasks in order."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for task in ("hw1_embedding", "hw1_rnn"):
        print(f"\n=== {task} ===", flush=True)
        subprocess.run([sys.executable, str(root / task / "main.py")], check=True)
