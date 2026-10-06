"""Run a list of kernel folders on Kaggle with bounded concurrency, pulling each one's outputs when it ends.

    py -3 scripts/kaggle_queue.py kaggle/lg-a-train kaggle/lg-b-train --concurrency 2

A ledger at checkpoints/kaggle/queue-ledger.jsonl records every push, final status, and pull.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_run import TERMINAL, pull, push, status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "checkpoints" / "kaggle" / "queue-ledger.jsonl"


def log(event: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with LEDGER.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)


def ref_for(folder: Path) -> str:
    return json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--poll", type=int, default=180)
    arguments = parser.parse_args()
    pending = list(arguments.folders)
    running: dict[str, Path] = {}
    while pending or running:
        while pending and len(running) < arguments.concurrency:
            folder = pending.pop(0)
            ref = ref_for(folder)
            try:
                push(folder)
                running[ref] = folder
                log({"event": "pushed", "ref": ref})
            except RuntimeError as error:
                log({"event": "push_failed", "ref": ref, "error": str(error)[:500]})
                if "concurrent" in str(error).lower() or "quota" in str(error).lower():
                    pending.insert(0, folder)
                    break
        time.sleep(arguments.poll)
        for ref in list(running):
            try:
                current = status(ref)
            except (RuntimeError, ValueError) as error:
                log({"event": "status_failed", "ref": ref, "error": str(error)[:300]})
                continue
            if current in TERMINAL:
                folder = running.pop(ref)
                dest = ROOT / "checkpoints" / "kaggle" / ref.split("/")[-1]
                try:
                    files = pull(ref, dest)
                    log({"event": "finished", "ref": ref, "status": current, "pulled_files": len(files)})
                except RuntimeError as error:
                    log({"event": "finished", "ref": ref, "status": current, "pull_error": str(error)[:300]})


if __name__ == "__main__":
    main()
