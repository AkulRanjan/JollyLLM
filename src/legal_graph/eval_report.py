"""Readable progress, tables and figures for outcome runs.

Everything here reads predictions or logs a run has already written; nothing scores the test split again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from .aws_judgments import AWS_DATASET
from .outcomes import LABELS, evaluate

SOURCE_NAMES = {
    "Shreyasrao/Indian-law-supreme-court-judgements-2016": "SC-2016 test",
    AWS_DATASET: "2017 test (AWS bucket)",
}
_SHORT = {"allowed": "allowed", "dismissed": "dismissed", "disposed": "disposed", "partly_allowed": "partly"}
_STEP = re.compile(r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+) elapsed (\d+)s")
_EPOCH = re.compile(r'\{"epoch": (\d+), "train_loss": [^}]*?"validation_accuracy": ([\d.]+), "validation_macro_f1": ([\d.]+)')


def progress_line(rows: Sequence[Mapping], total: int, elapsed_seconds: float, label: str = "scored") -> str:
    """One live line while a split is scored: cases done, time left, and the score so far."""
    metrics = evaluate([row["gold"] for row in rows], [row["predicted"] for row in rows])
    done = len(rows)
    left = elapsed_seconds / done * (total - done)
    return (f"{label}: {done}/{total} cases ({done / total:.0%}) | {elapsed_seconds / 60:.1f} min elapsed, "
            f"~{left / 60:.1f} min left | so far: accuracy {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}")


def running_curve(rows: Sequence[Mapping], every: int = 10) -> list[dict]:
    """Accuracy and macro-F1 over the first n scored cases, for growing n (in scoring order)."""
    sizes = sorted(set(range(every, len(rows) + 1, every)) | {len(rows)})
    points = []
    for size in sizes:
        metrics = evaluate([row["gold"] for row in rows[:size]], [row["predicted"] for row in rows[:size]])
        points.append({"cases": size, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    return points


def source_name(source: str) -> str:
    return SOURCE_NAMES.get(source, source)


def summary_lines(pass_name: str, payload: Mapping) -> list[str]:
    """Plain-text table of one evaluation pass: overall, then per dataset when the test split pools several."""
    blocks = [("all test cases", payload["metrics"])]
    blocks += [(source_name(name), metrics) for name, metrics in sorted(payload.get("metrics_by_source", {}).items())]
    lines = [f"== {pass_name}"]
    for name, metrics in blocks:
        per_class = " | ".join(f"{_SHORT[label]} {metrics['per_class'][label]['f1']:.3f}" for label in LABELS)
        lines.append(f"  {name:<24} n={metrics['eligible_case_count']:<4} accuracy {metrics['accuracy']:.3f}  "
                     f"macro-F1 {metrics['macro_f1']:.3f}   per-class F1: {per_class}")
    return lines


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _passes(run_dir: Path) -> list[tuple[str, dict, list[dict]]]:
    passes = []
    for suffix in ("", "-graph-ablated"):
        result = run_dir / f"test-evaluation{suffix}.json"
        if result.exists():
            payload = json.loads(result.read_text(encoding="utf-8"))
            if suffix:
                name = "graph removed"
            else:
                name = "text" if payload["variant"] == "text" else f"{payload['variant']} (with graph)"
            passes.append((name, payload, _read_jsonl(run_dir / f"test-predictions{suffix}.jsonl")))
    return passes


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_evaluation(run_dir: Path, figures_dir: Path) -> dict:
    """Tables and three figures for an evaluated run: running score, confusion matrices, per-class F1."""
    run_dir, figures_dir = Path(run_dir), Path(figures_dir)
    passes = _passes(run_dir)
    if not passes:
        raise FileNotFoundError(f"{run_dir} has no test-evaluation.json")
    plt = _pyplot()
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = passes[0][1]["experiment_id"]
    lines = [line for name, payload, _ in passes for line in summary_lines(name, payload)]
    figures = []

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, payload, rows in passes:
        curve = running_curve(rows)
        for axis, key in zip(axes, ("accuracy", "macro_f1")):
            line, = axis.plot([p["cases"] for p in curve], [p[key] for p in curve], label=name)
            axis.axhline(payload["metrics"][key], color=line.get_color(), linestyle="--", linewidth=0.8)
    for axis, title in zip(axes, ("Accuracy", "Macro-F1")):
        axis.set(title=f"{title} over the first n test cases", xlabel="test cases scored (in scoring order)", ylim=(0, 1))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f"{stem}: the frozen model's score settling as cases are counted (dashed = final)")
    figure.tight_layout()
    figures.append(figures_dir / f"{stem}-running-score.png")
    figure.savefig(figures[-1], dpi=110)
    plt.close(figure)

    main = passes[0][1]
    blocks = [(source_name(n), m) for n, m in sorted(main.get("metrics_by_source", {}).items())] or [("all test cases", main["metrics"])]
    figure, axes = plt.subplots(1, len(blocks), figsize=(4.6 * len(blocks), 4.2), squeeze=False)
    for axis, (name, metrics) in zip(axes[0], blocks):
        matrix = [[metrics["confusion_matrix"][gold][predicted] for predicted in LABELS] for gold in LABELS]
        axis.imshow(matrix, cmap="Blues")
        peak = max(max(row) for row in matrix) or 1
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                axis.text(j, i, count, ha="center", va="center", color="white" if count > peak / 2 else "black")
        ticks = [_SHORT[label] for label in LABELS]
        axis.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=ticks, yticklabels=ticks,
                 xlabel="predicted", ylabel="actual", title=f"{name} (n={metrics['eligible_case_count']})")
    figure.suptitle(f"{stem}: confusion matrices ({passes[0][0]})")
    figure.tight_layout()
    figures.append(figures_dir / f"{stem}-confusion.png")
    figure.savefig(figures[-1], dpi=110)
    plt.close(figure)

    sources = [None] + sorted(main.get("metrics_by_source", {}))
    figure, axes = plt.subplots(1, len(sources), figsize=(4.6 * len(sources), 3.8), squeeze=False)
    width = 0.8 / len(passes)
    for axis, source in zip(axes[0], sources):
        for offset, (name, payload, _) in enumerate(passes):
            metrics = payload["metrics"] if source is None else payload["metrics_by_source"][source]
            axis.bar([i + offset * width for i in range(len(LABELS))],
                     [metrics["per_class"][label]["f1"] for label in LABELS], width, label=name)
        axis.set(xticks=[i + width * (len(passes) - 1) / 2 for i in range(len(LABELS))],
                 xticklabels=[_SHORT[label] for label in LABELS], ylim=(0, 1), ylabel="F1",
                 title="all test cases" if source is None else source_name(source))
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"{stem}: F1 per outcome class")
    figure.tight_layout()
    figures.append(figures_dir / f"{stem}-per-class-f1.png")
    figure.savefig(figures[-1], dpi=110)
    plt.close(figure)
    return {"lines": lines, "figures": [str(path) for path in figures]}


def _log_text(path: Path) -> str:
    """Kaggle's downloaded log is a JSON list of stream records; a live `kernels logs` capture is plain text."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return "".join(record.get("data", "") for record in json.loads(raw))
    except (json.JSONDecodeError, AttributeError):
        return raw


def training_curve(log_path: Path) -> dict:
    """Per-step loss and per-epoch validation parsed from a training log.

    The printed loss is the epoch's running mean, so the loss over each interval between two prints is recovered as
    (mean_b * step_b - mean_a * step_a) / (step_b - step_a); it restarts at each epoch.
    """
    text = _log_text(log_path)
    steps, previous = [], {}
    for match in _STEP.finditer(text):
        epoch, step, total, mean, elapsed = int(match[1]), int(match[2]), int(match[3]), float(match[4]), int(match[5])
        last_step, last_mean = previous.get(epoch, (0, 0.0))
        if step <= last_step:
            continue  # a resumed session repeats lines it already logged
        window = (mean * step - last_mean * last_step) / (step - last_step)
        previous[epoch] = (step, mean)
        steps.append({"epoch": epoch, "step": step, "global_step": (epoch - 1) * total + step,
                      "running_mean_loss": mean, "interval_loss": window, "elapsed_seconds": elapsed})
    epochs = [{"epoch": int(m[1]), "validation_accuracy": float(m[2]), "validation_macro_f1": float(m[3])}
              for m in _EPOCH.finditer(text.replace('\\"', '"'))]
    return {"steps": steps, "epochs": epochs}


def _smoothed(steps: Sequence[Mapping]) -> list[float]:
    """Mean interval loss over the last four prints (~100 steps), never across an epoch boundary."""
    values = []
    for index in range(len(steps)):
        window = [s["interval_loss"] for s in steps[max(0, index - 3): index + 1] if s["epoch"] == steps[index]["epoch"]]
        values.append(sum(window) / len(window))
    return values


def render_training_comparison(logs: Mapping[str, Path], figures_dir: Path, name: str = "training-comparison") -> dict:
    """Smoothed training loss of several runs on one axis (e.g. text vs graph on the same data)."""
    plt = _pyplot()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    peak = 0.0
    for run_name, path in logs.items():
        steps = training_curve(path)["steps"]
        if steps:
            smoothed = _smoothed(steps)
            axis.plot([s["global_step"] for s in steps], smoothed, label=run_name)
            peak = max(peak, max(smoothed))
    axis.set(title="Training loss, smoothed over ~100 steps", xlabel="training examples seen", ylabel="loss",
             ylim=(0, min(3.0, peak * 1.05 or 1.0)))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    target = figures_dir / f"{name}.png"
    figure.savefig(target, dpi=110)
    plt.close(figure)
    return {"lines": [], "figures": [str(target)]}


def render_training_curve(log_path: Path, figures_dir: Path, name: str) -> dict:
    """Training loss over steps (smoothed over ~100 steps) and validation scores per epoch."""
    curve = training_curve(log_path)
    if not curve["steps"]:
        return {"lines": [f"{name}: no training steps in {log_path} (a run without training?)"], "figures": []}
    plt = _pyplot()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    steps = curve["steps"]
    smoothed = _smoothed(steps)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [s["global_step"] for s in steps]
    axes[0].plot(x, [s["interval_loss"] for s in steps], alpha=0.25, label="per 25 steps")
    axes[0].plot(x, smoothed, label="smoothed (~100 steps)")
    axes[0].set(title="Training loss (class-weighted)", xlabel="training examples seen", ylabel="loss")
    axes[0].set_ylim(0, min(3.0, max(s["interval_loss"] for s in steps) * 1.05))
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    if curve["epochs"]:
        epochs = [e["epoch"] for e in curve["epochs"]]
        axes[1].plot(epochs, [e["validation_macro_f1"] for e in curve["epochs"]], marker="o", label="macro-F1")
        axes[1].plot(epochs, [e["validation_accuracy"] for e in curve["epochs"]], marker="s", label="accuracy")
        axes[1].set_xticks(epochs)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "no epoch has finished yet\n(validation runs at the end of each epoch)",
                     ha="center", va="center", transform=axes[1].transAxes, color="gray")
    axes[1].set(title="Validation after each epoch", xlabel="epoch", ylim=(0, 1))
    axes[1].grid(alpha=0.3)
    figure.suptitle(name)
    figure.tight_layout()
    target = figures_dir / f"{name}-training-curve.png"
    figure.savefig(target, dpi=110)
    plt.close(figure)
    last = steps[-1]
    lines = [f"{name}: {last['global_step']} training steps, last ~100-step loss {smoothed[-1]:.3f}, "
             f"{last['elapsed_seconds'] / 3600:.2f} h"]
    lines += [f"  epoch {e['epoch']}: validation accuracy {e['validation_accuracy']:.3f}, macro-F1 {e['validation_macro_f1']:.3f}"
              for e in curve["epochs"]]
    return {"lines": lines, "figures": [str(target)]}
