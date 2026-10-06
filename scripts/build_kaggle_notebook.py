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


V1_GRAPH = "data/graphs/judgments/sc-2016/graph.json"
GRAPH_BUILDS = {
    V1_GRAPH: {
        "args": [],
        "training_manifest": "data/clean/judgments/training.manifest.json",
        "statute_files": BASE_GRAPH_FILES,
        "provisions": "data/clean/india_statutes/provisions.jsonl",
    },
    "data/graphs/judgments/sc-2016-v2/graph.json": {
        "args": ["--extractor", "v2", "--base-graph", "data/graphs/india_statutes-v2/graph.json",
                 "--output-dir", "data/graphs/judgments/sc-2016-v2",
                 "--report-output", "data/reports/judgments/sc-2016-v2/extraction-report.json",
                 "--training-manifest", "data/clean/judgments/training.v2.manifest.json",
                 "--graph-version", "india-statutes-sc2016-0.2.0"],
        "training_manifest": "data/clean/judgments/training.v2.manifest.json",
        "statute_files": ("data/graphs/india_statutes-v2/graph.json", "data/graphs/india_statutes-v2/graph.manifest.json"),
        "provisions": "data/clean/india_statutes-v2/provisions.jsonl",
    },
}


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


def expected_hashes(training_manifest: Path = TRAINING_MANIFEST) -> dict[str, object]:
    manifest = json.loads(training_manifest.read_text(encoding="utf-8"))
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

# peft refuses to build LoRA layers when an outdated torchao is importable (Kaggle images have shipped
# torchao 0.10 with peft 0.20). Nothing here uses torchao, so remove it rather than pin around it.
if importlib.util.find_spec("torchao") is not None:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
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
run("-m", "legal_graph.cli", "ingest-sc2016", "--code-revision", f"kaggle-bundle-{BUNDLE_SHA256[:16]}", *INGEST_ARGS)

training_manifest = json.loads((WORK / TRAINING_MANIFEST_PATH).read_text(encoding="utf-8"))
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


OUTCOME_INTRO = """
# LLM Graph: SC-2016 outcome model on Kaggle ({phase})

Generated by `scripts/build_kaggle_notebook.py --outcome {stem} --phase {phase}` and pushed as a private kernel by
`scripts/kaggle_run.py`. The project code, configs, statute graph and public statute text are embedded. SC-2016
and the base model are fetched from Hugging Face at pinned revisions, and the corpus hashes are checked against
the local build before anything trains.

- **train:** runs `train-outcome` for `{stem}`, then freezes the run (`freeze-outcome`). Test labels are never read.
- **evaluate:** takes the frozen run from the training kernel's output, verifies the freeze hashes, and scores the
  held-out test split once (graph variants are also scored once with the graph tokens removed).
"""

OUTCOME_TRAIN = """
# 7. Train the outcome model and freeze it. Checkpoints go straight into /kaggle/working.
output_dir = WORK / train_config["output_dir"]
if ON_KAGGLE:
    checkpoint_root = WORK / Path(train_config["output_dir"]).parts[0]
    if not checkpoint_root.is_symlink():
        remove_tree(checkpoint_root)
        (OUT / checkpoint_root.name).mkdir(parents=True, exist_ok=True)
        checkpoint_root.symlink_to(OUT / checkpoint_root.name, target_is_directory=True)
if output_dir.is_dir() and any(output_dir.iterdir()):
    raise FileExistsError(f"{output_dir} already has results")

# Resume: if an earlier session of this run is attached as an input (Add Input -> Your Work) and stopped before
# finishing, continue from its latest-checkpoint instead of starting over. A finished run is never retrained.
run_name = Path(train_config["output_dir"]).name
previous = sorted(
    path for path in Path("/kaggle/input").rglob(run_name)
    if path.is_dir() and (path / "latest-checkpoint" / "state.json").exists()
) if ON_KAGGLE else []
finished = [path for path in previous if (path / "run.manifest.json").exists()]
if finished:
    raise FileExistsError(f"{run_name} already finished in an attached input: {finished[0]}")
resume_args = []
if previous:
    if len(previous) > 1:
        raise RuntimeError(f"more than one resumable copy of {run_name} is attached: {[str(p) for p in previous]}")
    shutil.copytree(previous[0], output_dir)
    resume_args = ["--resume"]
    state = json.loads((output_dir / "latest-checkpoint" / "state.json").read_text(encoding="utf-8"))
    print(f"resuming from {previous[0]} at epoch {state['epoch']} step {state['step']}")

lineage = OUT / "lineage" / train_config["experiment_id"]
lineage.mkdir(parents=True, exist_ok=True)
for relative in (TRAINING_MANIFEST_PATH, str(Path(train_config["graph_path"]).parent / "graph.manifest.json")):
    shutil.copy2(WORK / relative, lineage / Path(relative).name)
shutil.copy2(WORK / config_path, lineage / "outcome-config.json")
print("checkpoint every", train_config.get("checkpoint_every_examples", 0), "examples (0 = epoch-best only)")
run("-m", "legal_graph.cli", "train-outcome", "--config", config_path, *resume_args)
run("-m", "legal_graph.cli", "freeze-outcome", "--run", str(output_dir))
"""

OUTCOME_EVALUATE = """
# 7. Evaluate the frozen run once on the held-out test split.
output_dir = WORK / train_config["output_dir"]
if ON_KAGGLE:
    checkpoint_root = WORK / Path(train_config["output_dir"]).parts[0]
    if not checkpoint_root.is_symlink():
        remove_tree(checkpoint_root)
        (OUT / checkpoint_root.name).mkdir(parents=True, exist_ok=True)
        checkpoint_root.symlink_to(OUT / checkpoint_root.name, target_is_directory=True)
run_name = Path(train_config["output_dir"]).name
candidates = sorted(
    path for path in Path("/kaggle/input").rglob(run_name) if path.is_dir() and (path / "freeze.json").exists()
) if ON_KAGGLE else []
if len(candidates) != 1:
    raise FileNotFoundError(f"expected exactly one frozen {run_name} under /kaggle/input, found {[str(p) for p in candidates]}")
remove_tree(output_dir)
shutil.copytree(candidates[0], output_dir)
if not (output_dir / "freeze.json").exists():
    raise RuntimeError("the training run was not frozen; refusing to evaluate on test")
run("-m", "legal_graph.cli", "evaluate-outcome", "--run", str(output_dir))
if train_config["variant"] != "text":
    run("-m", "legal_graph.cli", "evaluate-outcome", "--run", str(output_dir), "--ablate-graph")
"""

OUTCOME_OUTPUTS = """
# 8. Summary. Only adapters, encoder weights, predictions, metrics and text-free manifests are saved.
import glob

for name in ("run.manifest.json", "test-evaluation.json", "test-evaluation-graph-ablated.json"):
    path = output_dir / name
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        keep = {key: data[key] for key in ("status", "variant", "selector", "best_epoch", "validation_metrics",
                                           "subgraph_coverage", "ablate_graph") if key in data}
        if "metrics" in data:
            keep["test_accuracy"] = data["metrics"]["accuracy"]
            keep["test_macro_f1"] = data["metrics"]["macro_f1"]
        print(name, json.dumps(keep, indent=2))
print("saved under", OUT)
"""


KAGGLE_SOURCE_LIMIT = 1_000_000
STATUTE_DATASET = "akxran/lg-statute-data"
STATUTE_DATA_FILES = (
    "data/graphs/india_statutes/graph.json",
    "data/graphs/india_statutes/graph.manifest.json",
    "data/clean/india_statutes/provisions.jsonl",
    "data/graphs/india_statutes-v2/graph.json",
    "data/graphs/india_statutes-v2/graph.manifest.json",
    "data/clean/india_statutes-v2/provisions.jsonl",
)

STATUTE_DATA = """
# 3b. Statute layers and provision text from the private Kaggle dataset, each checked against the SHA-256
# recorded when this notebook was built. Kaggle limits notebook source to 1 MB, so they are not embedded.
import hashlib

for relative, expected in STATUTE_FILES.items():
    flat = relative.replace("/", "__")
    found = sorted(Path("/kaggle/input").rglob(flat)) if ON_KAGGLE else []
    if not found:
        raise FileNotFoundError(f"{flat} not found: attach the dataset {STATUTE_DATASET} (Add Input -> Your Datasets)")
    payload = found[0].read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError(f"{flat} does not match the hash this notebook was built with")
    target = WORK / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
print(f"{len(STATUTE_FILES)} statute files verified from {STATUTE_DATASET}")
"""


def statute_data_cell() -> str:
    hashes = {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in STATUTE_DATA_FILES}
    return f"STATUTE_DATASET = {STATUTE_DATASET!r}\nSTATUTE_FILES = {json.dumps(hashes, indent=4)}\n" + STATUTE_DATA


def outcome_notebook(stem: str, phase: str) -> dict[str, object]:
    config = f"configs/training/outcome/{stem}.json"
    if not (ROOT / config).is_file():
        raise SystemExit(f"unknown outcome config {config}")
    graph_path = json.loads((ROOT / config).read_text(encoding="utf-8"))["graph_path"]
    if graph_path not in GRAPH_BUILDS:
        raise SystemExit(f"no Kaggle graph build is defined for {graph_path}")
    build = GRAPH_BUILDS[graph_path]
    # Code and configs only: statute data comes from the private dataset (Kaggle's 1 MB source limit).
    statute = {ROOT / relative for relative in STATUTE_DATA_FILES}
    paths = [path for path in bundle_paths() if path not in statute]
    bundle = build_bundle(paths)
    settings = SETTINGS.replace('RUN = "qwen3-8b"', f'RUN = "{stem}"').replace(
        'CONFIG = CONFIGS[RUN]', f'CONFIGS[RUN] = "{config}"\nCONFIG = CONFIGS[RUN]')
    cells = [
        markdown(OUTCOME_INTRO.format(stem=stem, phase=phase)),
        code(settings),
        code(_bundle_cell(paths, bundle, build), hidden=True),
        code(ENVIRONMENT),
        code(UNPACK),
        code(statute_data_cell()),
        code(DATA),
        code(MODEL),
        code(INGEST),
        code(OUTCOME_TRAIN if phase == "train" else OUTCOME_EVALUATE),
        code(OUTCOME_OUTPUTS),
    ]
    return _notebook(cells)


SELFTEST_STEMS = ("qwen25-0.5b-smoke-text", "qwen25-0.5b-smoke-graph-typed", "qwen25-0.5b-smoke-graph-typed-4bit")

SELFTEST = """
# 7. Self-test: the whole unit suite with torch/peft/bitsandbytes on a GPU, then three smoke trainings
# (text fp16, typed graph fp16, typed graph + PCST in 4-bit). Validation only: no test labels are read.
import time

run("-m", "unittest", "discover", "-s", "tests")
if ON_KAGGLE:
    checkpoint_root = WORK / "checkpoints"
    if not checkpoint_root.is_symlink():
        remove_tree(checkpoint_root)
        (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
        checkpoint_root.symlink_to(OUT / "checkpoints", target_is_directory=True)
timings = {}
for stem in SMOKE_STEMS:
    started = time.time()
    run("-m", "legal_graph.cli", "train-outcome", "--config", f"configs/training/outcome/{stem}.json")
    timings[stem] = round(time.time() - started, 1)
    run_dir = WORK / json.loads((WORK / f"configs/training/outcome/{stem}.json").read_text())["output_dir"]
    manifest = json.loads((run_dir / "run.manifest.json").read_text(encoding="utf-8"))
    print(stem, json.dumps({key: manifest[key] for key in ("validation_metrics", "subgraph_coverage", "best_epoch")}))
    print("trainable fraction", manifest["parameter_report"]["trainable_fraction_with_encoder"])
(OUT / "selftest-timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
print(timings)
"""


CPU_UNITTEST = """
# Unit suite on a CPU session (no GPU quota): torch, transformers and peft from the Kaggle image.
import base64, hashlib, importlib.util, io, os, subprocess, sys, tarfile, tempfile
from pathlib import Path

payload = base64.b64decode("".join(BUNDLE_B64.split()))
assert hashlib.sha256(payload).hexdigest() == BUNDLE_SHA256, "bundle corrupted"
work = Path(tempfile.gettempdir()) / "llm-graph-unittest"
work.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    archive.extractall(work)
if importlib.util.find_spec("torchao") is not None:  # see the GPU notebooks: outdated torchao breaks peft LoRA
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
env = dict(os.environ, PYTHONPATH=str(work / "src"), CUDA_VISIBLE_DEVICES="")
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=work, env=env,
                        capture_output=True, text=True)
print(result.stderr[-20000:])
Path("/kaggle/working/unittest-result.txt").write_text(result.stderr, encoding="utf-8")
if result.returncode != 0:
    raise SystemExit("unit suite failed")
"""


def cpu_unittest_notebook() -> dict[str, object]:
    paths = bundle_paths() + sorted((ROOT / "tests").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    bundle = build_bundle(paths)
    return _notebook([code(_bundle_cell(paths, bundle), hidden=True), code(CPU_UNITTEST)])


def selftest_notebook() -> dict[str, object]:
    paths = bundle_paths() + [ROOT / "data/clean/india_statutes/provisions.jsonl"]
    paths += sorted((ROOT / "tests").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    bundle = build_bundle(paths)
    smoke_config = f"configs/training/outcome/{SELFTEST_STEMS[0]}.json"
    settings = SETTINGS.replace('RUN = "qwen3-8b"', 'RUN = "selftest"').replace(
        'CONFIG = CONFIGS[RUN]', f'CONFIGS[RUN] = "{smoke_config}"\nCONFIG = CONFIGS[RUN]\nSMOKE_STEMS = {list(SELFTEST_STEMS)!r}')
    cells = [
        markdown("# LLM Graph: outcome-model self-test on a Kaggle GPU\n\nUnit suite plus three smoke trainings."),
        code(settings),
        code(_bundle_cell(paths, bundle), hidden=True),
        code(ENVIRONMENT),
        code(UNPACK),
        code(DATA),
        code(MODEL),
        code(INGEST),
        code(SELFTEST),
    ]
    return _notebook(cells)


def _bundle_cell(paths: list[Path], bundle: bytes, build: dict | None = None) -> str:
    build = build or GRAPH_BUILDS[V1_GRAPH]
    encoded = base64.b64encode(bundle).decode("ascii")
    wrapped = "\n".join(encoded[start : start + 120] for start in range(0, len(encoded), 120))
    files = [path.relative_to(ROOT).as_posix() for path in paths]
    return (
        "# Embedded project bundle, generated by scripts/build_kaggle_notebook.py. Regenerate it; don't edit by hand.\n"
        f"BUNDLE_SHA256 = {hashlib.sha256(bundle).hexdigest()!r}\n"
        f"BUNDLE_FILES = {json.dumps(files, indent=4)}\n"
        f"EXPECTED = {json.dumps(expected_hashes(ROOT / build['training_manifest']), indent=4, sort_keys=True)}\n"
        f"INGEST_ARGS = {json.dumps(build['args'])}\n"
        f"TRAINING_MANIFEST_PATH = {build['training_manifest']!r}\n"
        f'BUNDLE_B64 = """\n{wrapped}\n"""\n'
    )


def _notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    notebook = {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isGpuEnabled": True,
                       "isInternetEnabled": True, "language": "python", "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    return notebook


def write_outcome_kernel(stem: str, phase: str, out_dir: Path, owner: str, train_slug: str | None = None) -> Path:
    """Write <out_dir>/<notebook>.ipynb and kernel-metadata.json for a private GPU kernel."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from kaggle_run import kernel_metadata, kernel_slug

    slug = kernel_slug(stem, phase)
    notebook_name = f"{slug}.ipynb"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / notebook_name).write_text(json.dumps(outcome_notebook(stem, phase), indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8", newline="\n")
    size = (out_dir / notebook_name).stat().st_size
    if size >= KAGGLE_SOURCE_LIMIT:
        raise SystemExit(f"{notebook_name} is {size:,} bytes; Kaggle rejects kernel sources of 1 MB or more")
    sources = [f"{owner}/{train_slug or kernel_slug(stem, 'train')}"] if phase == "evaluate" else []
    metadata = kernel_metadata(f"{owner}/{slug}", slug, notebook_name, kernel_sources=sources,
                               dataset_sources=[STATUTE_DATASET])
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outcome", help="outcome config stem under configs/training/outcome")
    parser.add_argument("--selftest", action="store_true", help="write the GPU self-test kernel")
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--out", type=Path, help="kernel folder (default kaggle/<slug>)")
    parser.add_argument("--owner", default="akxran")
    parser.add_argument("--unittest-cpu", action="store_true", help="write the CPU-only unit-suite kernel")
    arguments = parser.parse_args()
    if arguments.unittest_cpu:
        sys.path.insert(0, str(ROOT / "scripts"))
        from kaggle_run import kernel_metadata

        slug = "lg-unittest-cpu"
        out_dir = arguments.out or ROOT / "kaggle" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{slug}.ipynb").write_text(json.dumps(cpu_unittest_notebook(), indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8", newline="\n")
        metadata = kernel_metadata(f"{arguments.owner}/{slug}", slug, f"{slug}.ipynb", gpu=False, internet=True)
        (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"kernel_folder": str(out_dir)}, indent=2))
        return
    if arguments.selftest:
        sys.path.insert(0, str(ROOT / "scripts"))
        from kaggle_run import kernel_metadata

        slug = "lg-outcome-selftest"
        out_dir = arguments.out or ROOT / "kaggle" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{slug}.ipynb").write_text(json.dumps(selftest_notebook(), indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8", newline="\n")
        metadata = kernel_metadata(f"{arguments.owner}/{slug}", slug, f"{slug}.ipynb")
        (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"kernel_folder": str(out_dir), "files": sorted(p.name for p in out_dir.iterdir())}, indent=2))
        return
    if arguments.outcome:
        sys.path.insert(0, str(ROOT / "scripts"))
        from kaggle_run import kernel_slug

        out_dir = arguments.out or ROOT / "kaggle" / kernel_slug(arguments.outcome, arguments.phase)
        folder = write_outcome_kernel(arguments.outcome, arguments.phase, out_dir, arguments.owner)
        print(json.dumps({"kernel_folder": str(folder), "files": sorted(p.name for p in folder.iterdir())}, indent=2))
        return
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
        "INGEST_ARGS = []\n"
        "TRAINING_MANIFEST_PATH = 'data/clean/judgments/training.manifest.json'\n"
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
