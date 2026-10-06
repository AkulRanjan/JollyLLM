"""Run a list of kernel folders on Kaggle with bounded concurrency, pulling each one's outputs when it ends.

    py -3 scripts/kaggle_queue.py kaggle/lg-a-train kaggle/lg-b-train --concurrency 2 --stop-on-error
    py -3 scripts/kaggle_queue.py --adopt akxran/lg-already-running ...   # also track kernels pushed earlier

Kaggle answers a push beyond the account's concurrent-session limit (or past the weekly GPU quota) with a
plain HTTP 400, so a rejected push goes back to the front of the queue and is retried on the next poll.
A ledger at checkpoints/kaggle/queue-ledger.jsonl records every push, retry, final status, and pull.
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
    parser.add_argument("folders", nargs="*", type=Path)
    parser.add_argument("--folders-file", type=Path, help="text file with one kernel folder per line")
    parser.add_argument("--adopt", nargs="*", default=[], help="refs already running on Kaggle to track and pull")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--poll", type=int, default=180)
    parser.add_argument("--max-push-attempts", type=int, default=400)
    parser.add_argument("--stop-on-error", action="store_true", help="push nothing new after any kernel errors")
    arguments = parser.parse_args()
    pending = list(arguments.folders)
    if arguments.folders_file:
        pending += [Path(line) for line in arguments.folders_file.read_text(encoding="utf-8").split() if line]
    running: dict[str, Path | None] = {ref: None for ref in arguments.adopt}
    attempts: dict[str, int] = {}
    halted = False
    while (pending and not halted) or running:
        while pending and not halted and len(running) < arguments.concurrency:
            folder = pending[0]
            ref = ref_for(folder)
            try:
                push(folder)
            except RuntimeError as error:
                attempts[ref] = attempts.get(ref, 0) + 1
                if attempts[ref] >= arguments.max_push_attempts:
                    pending.pop(0)
                    log({"event": "push_abandoned", "ref": ref, "error": str(error)[:300]})
                    continue
                if attempts[ref] == 1 or attempts[ref] % 20 == 0:
                    log({"event": "push_deferred", "ref": ref, "attempt": attempts[ref], "error": str(error)[:200]})
                break
            pending.pop(0)
            running[ref] = folder
            log({"event": "pushed", "ref": ref})
        time.sleep(arguments.poll)
        for ref in list(running):
            try:
                current = status(ref)
            except (RuntimeError, ValueError) as error:
                log({"event": "status_failed", "ref": ref, "error": str(error)[:300]})
                continue
            if current not in TERMINAL:
                continue
            running.pop(ref)
            dest = ROOT / "checkpoints" / "kaggle" / ref.split("/")[-1]
            try:
                files = pull(ref, dest)
                log({"event": "finished", "ref": ref, "status": current, "pulled_files": len(files)})
            except RuntimeError as error:
                log({"event": "finished", "ref": ref, "status": current, "pull_error": str(error)[:300]})
            if current != "complete" and arguments.stop_on_error:
                halted = True
                log({"event": "halted", "reason": f"{ref} ended with {current}", "still_pending": [str(p) for p in pending]})


if __name__ == "__main__":
    main()
