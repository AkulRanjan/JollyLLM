"""Push private Kaggle GPU kernels, poll them, and pull their outputs (Kaggle CLI 2.x).

    py -3 scripts/kaggle_run.py push kaggle/<slug>
    py -3 scripts/kaggle_run.py status akxran/<slug>
    py -3 scripts/kaggle_run.py wait akxran/<slug>
    py -3 scripts/kaggle_run.py pull akxran/<slug> checkpoints/kaggle/<slug>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TERMINAL = ("complete", "error", "cancelled", "canceled", "cancel_acknowledged")


def kernel_slug(stem: str, phase: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", f"lg-{stem}-{phase}".lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) > 50:
        raise ValueError(f"kernel slug {slug!r} is longer than 50 characters")
    return slug


def kernel_metadata(
    ref: str,
    title: str,
    notebook: str,
    *,
    gpu: bool = True,
    internet: bool = True,
    kernel_sources: list[str] | None = None,
) -> dict:
    return {
        "id": ref,
        "title": title,
        "code_file": notebook,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": internet,
        "machine_shape": "NvidiaTeslaT4" if gpu else "None",
        "dataset_sources": [],
        "kernel_sources": list(kernel_sources or []),
        "competition_sources": [],
        "model_sources": [],
    }


def parse_status(text: str) -> str:
    match = re.search(r'status\s*"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', text)
    if not match:
        raise ValueError(f"cannot parse kernel status from {text!r}")
    return match.group(1).lower()


def _kaggle(*args: str) -> str:
    # Kernel logs contain non-ASCII text; force UTF-8 so the CLI does not fail on Windows code pages.
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def push(folder: Path) -> str:
    metadata = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("is_private") is not True:
        raise ValueError("refusing to push a kernel that is not private")
    return _kaggle("kernels", "push", "-p", str(folder))


def status(ref: str) -> str:
    return parse_status(_kaggle("kernels", "status", ref))


def wait(ref: str, poll_seconds: int = 120, timeout_seconds: int = 13 * 3600) -> str:
    started = time.time()
    while True:
        current = status(ref)
        print(f"{time.strftime('%H:%M:%S')} {ref}: {current}", flush=True)
        if current in TERMINAL:
            return current
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"{ref} still {current} after {timeout_seconds}s")
        time.sleep(poll_seconds)


def pull(ref: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", ref, "-p", str(dest), "--force")
    return sorted(path for path in dest.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive private Kaggle kernels")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("push").add_argument("folder", type=Path)
    sub.add_parser("status").add_argument("ref")
    wait_parser = sub.add_parser("wait")
    wait_parser.add_argument("ref")
    wait_parser.add_argument("--poll", type=int, default=120)
    pull_parser = sub.add_parser("pull")
    pull_parser.add_argument("ref")
    pull_parser.add_argument("dest", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "push":
        print(push(arguments.folder))
    elif arguments.command == "status":
        print(status(arguments.ref))
    elif arguments.command == "wait":
        print(wait(arguments.ref, arguments.poll))
    else:
        print(json.dumps([str(path) for path in pull(arguments.ref, arguments.dest)], indent=2))


if __name__ == "__main__":
    main()
