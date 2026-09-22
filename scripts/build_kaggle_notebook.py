"""Build a single self-contained Kaggle notebook for the SC-2016 QLoRA DAPT run.

The notebook carries the project code, configs, and the validated statute graph as one
embedded archive, so nothing has to be uploaded to Kaggle. At run time it downloads the
SC-2016 judgments and the base model from Hugging Face at the revisions the configs pin,
re-runs ingest-sc2016, checks the corpus and training-manifest hashes against the local
ones recorded here at build time, and then runs train-dapt.

    py -3 scripts/build_kaggle_notebook.py

No judgment text is embedded: the source registry forbids redistribution, so the corpus
is always rebuilt from the original Hugging Face dataset inside the Kaggle session.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import sys
import tarfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "kaggle_sc2016_dapt.ipynb"
TRAINING_MANIFEST = ROOT / "data/clean/judgments/training.manifest.json"
BASE_GRAPH_FILES = ("data/graphs/india_statutes/graph.json", "data/graphs/india_statutes/graph.manifest.json")
EXPECTED_FIELDS = ("corpus_sha256", "content_sha256", "graph_content_sha256", "source_sha256", "split_counts")


def bundle_paths() -> list[Path]:
    paths = sorted((ROOT / "src/legal_graph").glob("*.py"))
    paths += sorted((ROOT / "configs").rglob("*.json"))
    paths += [ROOT / relative for relative in BASE_GRAPH_FILES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing bundle inputs (run the local pipeline first): {[str(path) for path in missing]}")
    return paths


def build_bundle(paths: list[Path]) -> bytes:
    """Return a deterministic tar.gz so an unchanged workspace yields an unchanged bundle hash."""

    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in paths:
                data = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(ROOT).as_posix())
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def expected_hashes() -> dict[str, object]:
    manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    return {field: manifest[field] for field in EXPECTED_FIELDS}


def markdown(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str, *, hidden: bool = False) -> dict[str, object]:
    metadata: dict[str, object] = {"jupyter": {"source_hidden": True}} if hidden else {}
    return {"cell_type": "code", "execution_count": None, "metadata": metadata, "outputs": [], "source": _lines(text)}


def _lines(text: str) -> list[str]:
    lines = textwrap.dedent(text).strip("\n").splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


INTRO = """
# LLM Graph: SC-2016 QLoRA DAPT on Kaggle

This notebook runs the project's `train-dapt` step on a Kaggle GPU. You don't need to upload anything.
The project code, configs, and statute graph are packed into the notebook. The SC-2016 judgments and the
base model are downloaded from Hugging Face at the exact revisions pinned in the repo.

**Choose one model per run** in the first code cell:

| `RUN` | Model | Role | Download |
| --- | --- | --- | --- |
| `"qwen3-8b"` (default) | Qwen/Qwen3-8B | Main model | 16.4 GB |
| `"saul-7b"` | Equall/Saul-7B-Instruct-v1 | Legal-domain comparison: same data, split, and settings | 29.0 GB |
| `"smoke"` | Qwen/Qwen2.5-1.5B-Instruct | Short pipeline check (20 steps) | 3.1 GB |

Run `qwen3-8b` first, then `saul-7b` as a separate run. A 1,500-step run takes hours on a T4, and Kaggle stops
sessions after 12 hours. Checkpoints are saved every 250 steps, so a run that is cut off still keeps its progress.

**Before running, open the right-hand panel → Session options:**

1. **Accelerator:** `GPU T4 x2`. Avoid P100: Kaggle's current PyTorch and bitsandbytes builds may not support it.
2. **Internet:** `On`. This requires a phone-verified Kaggle account.

Then use **Save Version → Save & Run All** so training continues after you close the tab. The adapters appear in
the version's **Output** tab. **Run All** works too, but only while the browser session stays open.

| Step | What it does |
| --- | --- |
| 1. Settings | Choose the model and optional overrides |
| 2. Environment | Check the GPU and Internet access; install or upgrade packages only when needed |
| 3. Unpack | Verify and extract the embedded project bundle, then resolve the training config |
| 4. Data | Fetch SC-2016 JSON and Markdown from Hugging Face with a sparse git fetch (~12 MB, pinned commit) |
| 5. Model | Check disk space, then download the base model from Hugging Face (pinned revision) |
| 6. Ingest | Run `ingest-sc2016` and check that the corpus hashes match the local build |
| 7. Train | Save lineage, then run `train-dapt` with checkpoints written straight to `/kaggle/working` |
| 8. Outputs | Summarise the run and flag non-finite losses |

**Data handling:** the source registry sets `redistribution_allowed: false` for SC-2016. Judgments and model
weights stay on the session's temporary disk and are never written to `/kaggle/working`, so they are not saved
with notebook versions. Only LoRA adapters and text-free manifests are saved. Keep the notebook private.

To regenerate this notebook after local code changes, run `py -3 scripts/build_kaggle_notebook.py`.
"""

SETTINGS = """
# 1. Settings: the only cell you should need to edit.
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# One model per run: "qwen3-8b" (main), then "saul-7b" (legal-domain comparison) in a separate run.
RUN = "qwen3-8b"
CONFIGS = {
    "qwen3-8b": "configs/training/qwen3-8b-sc2016-dapt-v1.json",
    "saul-7b": "configs/training/saul-7b-sc2016-dapt-v1.json",
    "smoke": "configs/training/qwen25-1.5b-sc2016-dapt-smoke.json",
}
CONFIG = CONFIGS[RUN]

# Optional changes to the chosen config, e.g. {"max_steps": 500, "checkpoint_every_steps": 100}.
# They are applied before the model download and get their own experiment_id, so no run is overwritten.
# To swap the base model, override "model_source" and "model_revision" together.
OVERRIDES = {}

ON_KAGGLE = Path("/kaggle/working").is_dir()
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path(tempfile.gettempdir())
WORK = SCRATCH / "llm-graph"  # code, judgments, and model weights: not saved with the notebook
OUT = Path("/kaggle/working") if ON_KAGGLE else SCRATCH / "llm-graph-output"  # saved outputs


def run(*args):
    \"\"\"Run a Python module inside the project workspace, streaming its output.\"\"\"
    pythonpath = os.pathsep.join(filter(None, (str(WORK / "src"), os.environ.get("PYTHONPATH"))))
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    process = subprocess.Popen(
        [sys.executable, *args], cwd=WORK, env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for line in process.stdout:
        print(line, end="")
    if process.wait() != 0:
        raise RuntimeError(f"exit code {process.returncode}: python {' '.join(args)}")


print("workspace:", WORK)
print("outputs:  ", OUT)
"""

ENVIRONMENT = """
# 2. Environment: GPU, Internet, and packages.
import importlib.util
import urllib.error
import urllib.request
from importlib.metadata import version as installed_version

from packaging.version import Version

WORK.mkdir(parents=True, exist_ok=True)

try:
    urllib.request.urlopen("https://huggingface.co/api/whoami-v2", timeout=20)
except urllib.error.HTTPError:
    pass  # 401 without a token still proves the Hub is reachable
except OSError as error:
    raise RuntimeError("No Internet. Session options -> Internet -> On (needs a verified account).") from error
print("Hugging Face Hub: reachable")

REQUIRED = {"torch": "torch", "transformers": "transformers", "peft": "peft", "bitsandbytes": "bitsandbytes",
            "accelerate": "accelerate", "huggingface_hub": "huggingface_hub", "pypdf": "pypdf"}
MINIMUM = {"transformers": "4.51"}  # first release with Qwen3
missing = [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]
outdated = [
    f"{package}>={minimum}" for package, minimum in MINIMUM.items()
    if package not in missing and Version(installed_version(package)) < Version(minimum)
]
if missing or outdated:
    print("installing:", ", ".join(missing + outdated))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing, *outdated], check=True)
    importlib.invalidate_caches()

run("-c", "import importlib.metadata as m, sys; print('python', sys.version.split()[0]); "
          + "; ".join(f"print({name!r}, m.version({name!r}))" for name in REQUIRED.values()))

# Probe in a subprocess, with the same torch that training uses, so this kernel holds no GPU memory.
run("-c", '''
import torch
if not torch.cuda.is_available():
    raise SystemExit("No GPU found. Session options -> Accelerator -> GPU T4 x2, then run again.")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {torch.cuda.device_count()} x {props.name}, {props.total_memory / 1e9:.1f} GB, capability {props.major}.{props.minor}")
if props.major < 7:
    print("WARNING: capability below 7.0 (P100?). Switch to GPU T4 x2 if training fails.")
''')
"""

UNPACK = """
# 3. Unpack the embedded project bundle, then resolve the training config used by every later step.
import base64
import hashlib
import io
import json
import shutil
import tarfile

payload = base64.b64decode("".join(BUNDLE_B64.split()))
if hashlib.sha256(payload).hexdigest() != BUNDLE_SHA256:
    raise RuntimeError("The embedded bundle is corrupted. Re-import the notebook.")
for stale in ("src", "configs"):
    shutil.rmtree(WORK / stale, ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    try:
        archive.extractall(WORK, filter="data")
    except TypeError:  # Python without tarfile extraction filters
        archive.extractall(WORK)
print(f"bundle {BUNDLE_SHA256[:16]}: {len(BUNDLE_FILES)} files extracted to {WORK}")

train_config = json.loads((WORK / CONFIG).read_text(encoding="utf-8"))
config_path = CONFIG
if OVERRIDES:
    if ("model_source" in OVERRIDES) != ("model_revision" in OVERRIDES):
        raise ValueError("override model_source and model_revision together, so the base model stays pinned")
    derived = {**train_config, **OVERRIDES, "lora": {**train_config["lora"], **OVERRIDES.get("lora", {})}}
    suffix = hashlib.sha256(json.dumps(OVERRIDES, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    derived["experiment_id"] = OVERRIDES.get("experiment_id", f"{train_config['experiment_id']}-kaggle-{suffix}")
    derived["output_dir"] = OVERRIDES.get("output_dir", f"checkpoints/{derived['experiment_id']}")
    if "model_source" in OVERRIDES and "local_model_path" not in OVERRIDES:
        name = derived["model_source"].split("/")[-1].lower()
        derived["local_model_path"] = f"models/{name}-{derived['model_revision'][:12]}"
    config_path = f"configs/training/{derived['experiment_id']}.json"
    (WORK / config_path).write_text(json.dumps(derived, indent=2) + "\\n", encoding="utf-8")
    train_config = derived
print(f"run {RUN!r}: {train_config['experiment_id']} | {train_config['model_source']} "
      f"@ {train_config['model_revision'][:12]} | config {config_path}")
"""

DATA = """
# 4. SC-2016 judgments from Hugging Face, at the revision recorded in the source registry.
import json
import re
import stat

from huggingface_hub import snapshot_download


def hf_token():
    \"\"\"Use an optional Kaggle secret named HF_TOKEN (Add-ons -> Secrets) to avoid anonymous rate limits.\"\"\"
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


registry = json.loads((WORK / "configs/sources/sc2016-source.json").read_text(encoding="utf-8"))
locators = [
    re.fullmatch(r"https://huggingface\\.co/datasets/(?P<repo>[^/]+/[^/]+)/tree/(?P<revision>[0-9a-f]{40})",
                 source["source_locator"])
    for source in registry["sources"]
]
locators = [match for match in locators if match]
if len(locators) != 1:
    raise RuntimeError("expected exactly one pinned Hugging Face dataset in the SC-2016 source registry")
SC2016_REPO, SC2016_REVISION = locators[0]["repo"], locators[0]["revision"]

sc2016_root = WORK / "data/raw/judgments/sc-2016"
SC2016_PATTERNS = ["extracted_jsons/*.json", "extracted_mds/*.md"]


def remove_tree(path):
    \"\"\"Delete a directory, including git's read-only pack files.\"\"\"
    def retry_writable(function, target, *_):
        os.chmod(target, stat.S_IWRITE)
        function(target)
    if path.exists():
        shutil.rmtree(path, **{("onexc" if sys.version_info >= (3, 12) else "onerror"): retry_writable})


def fetch_sc2016():
    \"\"\"Fetch the pinned commit with one shallow, sparse git fetch.

    Downloading 1,178 files one by one through the Hub API gets rate-limited (HTTP 429), so that API is
    only the fallback. Either way, ingest-sc2016 verifies the source checksum afterwards.
    \"\"\"
    git = ["git", "-C", str(sc2016_root), "-c", "core.autocrlf=false", "-c", "advice.detachedHead=false"]
    if (sc2016_root / ".git").is_dir():
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip() == SC2016_REVISION:
            return "git (already present)"
    remove_tree(sc2016_root)
    sc2016_root.mkdir(parents=True)
    commands = (
        git + ["init", "-q"],
        git + ["remote", "add", "origin", f"https://huggingface.co/datasets/{SC2016_REPO}"],
        git + ["sparse-checkout", "set", "--no-cone", *(f"/{pattern}" for pattern in SC2016_PATTERNS)],
        git + ["fetch", "-q", "--depth", "1", "origin", SC2016_REVISION],
        git + ["checkout", "-q", "FETCH_HEAD"],
    )
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    try:
        for command in commands:
            subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        return "git"
    except (OSError, subprocess.CalledProcessError) as error:
        print("git fetch failed; falling back to the Hub file API:", getattr(error, "stderr", None) or error)
    remove_tree(sc2016_root)
    snapshot_download(repo_id=SC2016_REPO, repo_type="dataset", revision=SC2016_REVISION, local_dir=sc2016_root,
                      allow_patterns=SC2016_PATTERNS, token=hf_token())
    return "hub"


method = fetch_sc2016()
print(SC2016_REPO, "@", SC2016_REVISION[:12], "via", method,
      "| json:", len(list((sc2016_root / "extracted_jsons").glob("*.json"))),
      "| markdown:", len(list((sc2016_root / "extracted_mds").glob("*.md"))))
"""

MODEL = """
# 5. Base model from Hugging Face, at the revision pinned in the training config.
from fnmatch import fnmatch

from huggingface_hub import HfApi

model_repo, model_revision = train_config["model_source"], train_config["model_revision"]
model_dir = WORK / train_config["local_model_path"]
MODEL_PATTERNS = ["*.json", "*.safetensors", "merges.txt", "tokenizer.model", "LICENSE"]

# Keep one base model on the temporary disk at a time; the others can be downloaded again.
models_root = WORK / "models"
for other in models_root.glob("*") if models_root.is_dir() else ():
    if other.is_dir() and other.resolve() != model_dir.resolve():
        print("removing other base model:", other.name)
        remove_tree(other)

info = HfApi().model_info(model_repo, revision=model_revision, files_metadata=True, token=hf_token())
needed = sum(item.size or 0 for item in info.siblings if any(fnmatch(item.rfilename, p) for p in MODEL_PATTERNS))
present = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()) if model_dir.is_dir() else 0
free = shutil.disk_usage(WORK).free
if needed - present > free:
    raise RuntimeError(f"{model_repo} needs {needed / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free under {WORK}. "
                       "Restart the session to clear the temporary disk.")
print(f"downloading {model_repo} @ {model_revision[:12]}: {needed / 1e9:.1f} GB ({free / 1e9:.0f} GB free)")
snapshot_download(repo_id=model_repo, revision=model_revision, local_dir=model_dir,
                  allow_patterns=MODEL_PATTERNS, token=hf_token())
size = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
print(f"weights staged at {model_dir}: {size / 1e9:.2f} GB")
"""

INGEST = """
# 6. Rebuild the corpus, graph, and training manifest; they must match the local build byte for byte.
run("-m", "legal_graph.cli", "ingest-sc2016", "--code-revision", f"kaggle-bundle-{BUNDLE_SHA256[:16]}")

training_manifest = json.loads((WORK / "data/clean/judgments/training.manifest.json").read_text(encoding="utf-8"))
mismatches = {field: (training_manifest.get(field), expected) for field, expected in EXPECTED.items()
              if training_manifest.get(field) != expected}
if mismatches:
    raise RuntimeError(
        "The Kaggle corpus differs from the local build this notebook was generated from: "
        f"{mismatches}. Re-run ingest-sc2016 locally, then regenerate the notebook."
    )
print("\\nlineage verified:", {field: str(value)[:16] for field, value in EXPECTED.items() if field != "split_counts"})
print("splits:", EXPECTED["split_counts"])
"""

TRAIN = """
# 7. Train. On Kaggle, checkpoints go straight into /kaggle/working, so a run cut off by the 12-hour limit
# still keeps every checkpoint saved so far.
import threading
import time

output_dir = WORK / train_config["output_dir"]
saved_dir = OUT / train_config["output_dir"]
if output_dir.is_dir() and any(output_dir.iterdir()):
    raise FileExistsError(f"{output_dir} already has results. Set OVERRIDES = {{\\"experiment_id\\": \\"...\\"}} to start a new run.")
if ON_KAGGLE:
    checkpoint_root = WORK / Path(train_config["output_dir"]).parts[0]
    if not checkpoint_root.is_symlink():
        remove_tree(checkpoint_root)
        (OUT / checkpoint_root.name).mkdir(parents=True, exist_ok=True)
        checkpoint_root.symlink_to(OUT / checkpoint_root.name, target_is_directory=True)

# Lineage first, so even an interrupted run records exactly what it was trained on.
lineage = OUT / "lineage" / train_config["experiment_id"]
lineage.mkdir(parents=True, exist_ok=True)
for relative in ("data/clean/judgments/training.manifest.json", "data/graphs/judgments/sc-2016/graph.manifest.json"):
    shutil.copy2(WORK / relative, lineage / Path(relative).name)
shutil.copy2(WORK / config_path, lineage / "training-config.json")


def report_progress(stop, started):
    \"\"\"train-dapt prints nothing until it finishes, so report each checkpoint as it lands.\"\"\"
    seen = set()
    while not stop.wait(60):
        for step_dir in sorted(output_dir.glob("step-*")):
            if step_dir.name not in seen:
                seen.add(step_dir.name)
                step, elapsed = int(step_dir.name.split("-")[1]), time.time() - started
                left = elapsed / step * (train_config["max_steps"] - step)
                print(f"[progress] {step_dir.name} of {train_config['max_steps']} saved after "
                      f"{elapsed / 60:.0f} min; about {left / 60:.0f} min left", flush=True)


stop = threading.Event()
threading.Thread(target=report_progress, args=(stop, time.time()), daemon=True).start()
try:
    run("-m", "legal_graph.cli", "train-dapt", "--config", config_path)
finally:
    stop.set()
"""

OUTPUTS = """
# 8. Outputs: adapters, run manifest, and lineage. Judgment text and model weights are deliberately left behind.
import math

if output_dir.resolve() != saved_dir.resolve():  # outside Kaggle, checkpoints were written to the workspace
    remove_tree(saved_dir)
    shutil.copytree(output_dir, saved_dir)

run_manifest = json.loads((saved_dir / "run.manifest.json").read_text(encoding="utf-8"))
for key in ("status", "experiment_id", "base_model_source", "cuda_device", "train_document_count",
            "train_block_count", "train_loss_first", "train_loss_last", "validation_loss", "checkpoint_steps"):
    print(f"{key:>22}: {run_manifest[key]}")
losses = (run_manifest["train_loss_first"], run_manifest["train_loss_last"], run_manifest["validation_loss"])
if not all(math.isfinite(value) for value in losses):
    print("\\nWARNING: non-finite loss. T4 trains in fp16; retry with a lower learning_rate in OVERRIDES.")
print("\\nadapters:", saved_dir)
print("lineage: ", lineage)
print("Validation loss is per token of each model's own tokenizer: compare Qwen and Saul on outcome metrics, not on it.")
"""

OUTRO = """
### Next steps

- **Download:** open the saved version's **Output** tab. Adapters are in `checkpoints/<experiment>/adapter`, and
  what each run was trained on is in `lineage/<experiment>/`.
- **Comparison run:** set `RUN = "saul-7b"` and save a new version. It trains on the same documents, split, and
  settings, so the two adapters differ only in the base model.
- **Reuse in another Kaggle notebook:** use **Add Input → Your Work → this notebook**. The adapters mount read-only
  under `/kaggle/input/…`, so they never need to be downloaded or re-uploaded.
- **Bring back to the local repo:** place `checkpoints/<experiment>` under the repo's `checkpoints/`.
  `run.manifest.json` records the corpus, manifest, and base-model revisions it was trained on.
"""


def main() -> None:
    paths = bundle_paths()
    bundle = build_bundle(paths)
    bundle_sha256 = hashlib.sha256(bundle).hexdigest()
    encoded = base64.b64encode(bundle).decode("ascii")
    wrapped = "\n".join(encoded[start : start + 120] for start in range(0, len(encoded), 120))
    files = [path.relative_to(ROOT).as_posix() for path in paths]
    bundle_cell = (
        "# Embedded project bundle, generated by scripts/build_kaggle_notebook.py. Regenerate it; don't edit by hand.\n"
        f"BUNDLE_SHA256 = {bundle_sha256!r}\n"
        f"BUNDLE_FILES = {json.dumps(files, indent=4)}\n"
        f"EXPECTED = {json.dumps(expected_hashes(), indent=4, sort_keys=True)}\n"
        f'BUNDLE_B64 = """\n{wrapped}\n"""\n'
    )
    notebook = {
        "cells": [
            markdown(INTRO),
            code(SETTINGS),
            code(bundle_cell, hidden=True),
            code(ENVIRONMENT),
            code(UNPACK),
            code(DATA),
            code(MODEL),
            code(INGEST),
            code(TRAIN),
            code(OUTPUTS),
            markdown(OUTRO),
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "notebook": str(OUTPUT.relative_to(ROOT)),
        "notebook_bytes": OUTPUT.stat().st_size,
        "bundle_files": len(files),
        "bundle_bytes": len(bundle),
        "bundle_sha256": bundle_sha256,
    }, indent=2))


if __name__ == "__main__":
    sys.exit(main())
