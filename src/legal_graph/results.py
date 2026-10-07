"""Collect measured outcome runs, compare each with its text-only control, and place it against the forecast."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .comparison import aggregate_seeds, compare_to_projection, paired_bootstrap

_SEED = re.compile(r"-s(\d+)-v\d+$")


def discover_runs(root: Path) -> dict[str, Path]:
    """experiment_id -> run directory, for every run.manifest.json under checkpoints/ (local or pulled from Kaggle)."""
    runs: dict[str, Path] = {}
    for manifest in sorted((root / "checkpoints").rglob("run.manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("method") != "graph_outcome_qlora_v1":
            continue
        experiment = payload["experiment_id"]
        current = runs.get(experiment)
        if current is None:
            runs[experiment] = manifest.parent
            continue
        # A training kernel's output and the evaluation kernel's copy both hold the run; keep the evaluated one.
        evaluated = [path for path in (current, manifest.parent) if (path / "test-evaluation.json").exists()]
        if len(evaluated) == 2:
            raise ValueError(f"experiment {experiment} was evaluated twice: {current} and {manifest.parent}")
        if evaluated:
            runs[experiment] = evaluated[0]
    return runs


def control_key(experiment_id: str) -> str:
    """Model family plus seed, e.g. qwen3-4b-graph-typed-g2-s2-v1 -> qwen3-4b-s2; qwen3-8b-text-v1 -> qwen3-8b."""
    family = "-".join(experiment_id.split("-")[:2])
    seed = _SEED.search(experiment_id)
    return f"{family}-s{seed.group(1)}" if seed else family


def _predictions(run: Path, name: str) -> dict[str, dict] | None:
    path = run / name
    if not path.exists():
        return None
    return {row["case_id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def collect(root: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    runs = discover_runs(root)
    rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    required_view = settings.get("required_input_view")
    for experiment, run in sorted(runs.items()):
        config_path = run / "config.json"
        view = json.loads(config_path.read_text(encoding="utf-8")).get("input_view", "prefix-with-disposition-masking-v1") \
            if config_path.exists() else None
        if experiment.startswith("diag-"):
            excluded[experiment] = "diagnostic run (not part of the reported comparison)"
            continue
        if experiment in settings.get("excluded_runs", {}):
            excluded[experiment] = settings["excluded_runs"][experiment]
            continue
        if required_view and view != required_view:
            excluded[experiment] = f"input view {view!r} is not the required {required_view!r}"
            continue
        history = json.loads((run / "run.manifest.json").read_text(encoding="utf-8")).get("history", [])
        if any(not math.isfinite(float(entry.get("train_loss", 0.0))) for entry in history):
            excluded[experiment] = "diverged: non-finite training loss"
            continue
    runs = {experiment: run for experiment, run in runs.items() if experiment not in excluded}
    for experiment, run in sorted(runs.items()):
        manifest = json.loads((run / "run.manifest.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "experiment_id": experiment,
            "variant": manifest["variant"],
            "selector": manifest.get("selector"),
            "base_model": manifest.get("base_model_source"),
            "frozen": (run / "freeze.json").exists(),
            "validation": manifest["validation_metrics"],
            "best_epoch": manifest.get("best_epoch"),
            "subgraph_coverage_validation": (manifest.get("subgraph_coverage") or {}).get("validation"),
            "trainable_fraction": manifest["parameter_report"].get("trainable_fraction_with_encoder"),
        }
        for suffix, key in (("", "test"), ("-graph-ablated", "test_graph_ablated")):
            path = run / f"test-evaluation{suffix}.json"
            if path.exists():
                metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                row[key] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                            "per_class_f1": {label: values["f1"] for label, values in metrics["per_class"].items()},
                            "cases": metrics["eligible_case_count"]}
        control_id = settings["controls"].get(control_key(experiment))
        if "test" in row and control_id and control_id != experiment and control_id in runs:
            mine = _predictions(run, "test-predictions.jsonl")
            theirs = _predictions(runs[control_id], "test-predictions.jsonl")
            if mine and theirs:
                if sorted(mine) != sorted(theirs):
                    raise ValueError(f"{experiment} and {control_id} were scored on different test cases")
                ids = sorted(mine)
                row["versus_text_control"] = {"control": control_id, **paired_bootstrap(
                    [mine[i]["gold"] for i in ids], [mine[i]["predicted"] for i in ids], [theirs[i]["predicted"] for i in ids],
                    metric=settings["primary_metric"], resamples=int(settings["bootstrap_resamples"]))}
        rows.append(row)

    seeds: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if _SEED.search(row["experiment_id"]):
            seeds.setdefault(_SEED.sub("", row["experiment_id"]), []).append(row)
    seed_summary = {
        group: {
            "runs": [member["experiment_id"] for member in members],
            "validation_macro_f1": aggregate_seeds([member["validation"]["macro_f1"] for member in members]),
            **({"test_macro_f1": aggregate_seeds([member["test"]["macro_f1"] for member in members]),
                "test_accuracy": aggregate_seeds([member["test"]["accuracy"] for member in members])}
               if all("test" in member for member in members) else {}),
        }
        for group, members in sorted(seeds.items())
    }

    forecast = []
    projection_path = root / settings["projection_report"]
    if projection_path.exists():
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        by_projection = {settings["projection_variant_for"][row["experiment_id"]]: row["test"]["accuracy"]
                         for row in rows if "test" in row and row["experiment_id"] in settings["projection_variant_for"]}
        forecast = compare_to_projection(by_projection, projection)
    return {
        "schema_version": "1.0.0",
        "status": "measured",
        "runs": rows,
        "seed_groups": seed_summary,
        "measured_vs_projected": forecast,
        "excluded_runs": excluded,
        "notes": settings.get("note", ""),
    }


def markdown_table(report: Mapping[str, Any]) -> str:
    lines = ["| Experiment | Variant | Val. acc | Val. mF1 | Test acc | Test mF1 | Δ mF1 vs text [95% CI] | Graph-ablated test mF1 |",
             "|---|---|---|---|---|---|---|---|"]
    for row in report["runs"]:
        test = row.get("test", {})
        delta = row.get("versus_text_control")
        ablated = row.get("test_graph_ablated", {})
        lines.append("| {} | {} | {:.3f} | {:.3f} | {} | {} | {} | {} |".format(
            row["experiment_id"], row["variant"], row["validation"]["accuracy"], row["validation"]["macro_f1"],
            f"{test['accuracy']:.3f}" if test else "–", f"{test['macro_f1']:.3f}" if test else "–",
            f"{delta['delta']:+.3f} [{delta['lower']:+.3f}, {delta['upper']:+.3f}]" if delta else "–",
            f"{ablated['macro_f1']:.3f}" if ablated else "–"))
    return "\n".join(lines) + "\n"


def write_report(root: Path, settings_path: Path) -> dict[str, Any]:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    report = collect(root, settings)
    output = root / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "measured-results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "measured-results.md").write_text(markdown_table(report), encoding="utf-8")
    return {"runs": len(report["runs"]), "with_test": sum("test" in row for row in report["runs"]), "output_dir": str(output)}
