"""Deterministic, clearly-labelled forecast of SC-2016 outcome accuracy per planned variant.

Nothing here is a measurement. The module takes the real corpus label supports, the
real measured Naive Bayes baseline, and a config of per-class recall assumptions, and
turns them into projected confusion matrices, metrics, uncertainty bands, and a
sample-size analysis so the planned experiments can be judged before they are run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .outcomes import LABELS, evaluate, infer_disposition


class ProjectionConfigurationError(ValueError):
    pass


_SPLIT_FIELDS = {"train": "training_splits", "validation": "validation_splits", "test": "held_out_splits"}


@dataclass(frozen=True, slots=True)
class Variant:
    variant_id: str
    label: str
    family: str
    evidence: str
    graph_evidence: str
    note: str
    class_recall: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    projection_id: str
    evaluator_version: str
    corpus_path: Path
    training_data_manifest: Path
    measured_validation_metrics: Path
    output_dir: Path
    primary_variant_id: str
    measured_variant_id: str
    confusability_prior: Mapping[str, Mapping[str, float]]
    variants: tuple[Variant, ...]
    learning_curve: Mapping[str, object]
    simulation: Mapping[str, object]


def load_config(path: Path) -> ProjectionConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectionConfigurationError(f"cannot load {path}: {error}") from error
    if data.get("schema_version") != "1.0.0" or data.get("method") != "class_recall_uplift_projection_v1":
        raise ProjectionConfigurationError("config must specify schema 1.0.0 and class_recall_uplift_projection_v1")
    raw_variants = data.get("variants")
    if not isinstance(raw_variants, list) or len(raw_variants) < 2:
        raise ProjectionConfigurationError("config must declare at least two variants")
    variants: list[Variant] = []
    for entry in raw_variants:
        try:
            recall = {label: float(entry["class_recall"][label]) for label in LABELS}
            variant = Variant(
                variant_id=str(entry["variant_id"]),
                label=str(entry["label"]),
                family=str(entry["family"]),
                evidence=str(entry["evidence"]),
                graph_evidence=str(entry["graph_evidence"]),
                note=str(entry["note"]),
                class_recall=recall,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProjectionConfigurationError(f"invalid variant entry: {error}") from error
        if variant.evidence not in {"measured", "projected"}:
            raise ProjectionConfigurationError(f"{variant.variant_id}: evidence must be measured or projected")
        if any(not 0.0 <= value <= 1.0 for value in variant.class_recall.values()):
            raise ProjectionConfigurationError(f"{variant.variant_id}: class recall must lie in [0, 1]")
        variants.append(variant)
    identifiers = [variant.variant_id for variant in variants]
    if len(set(identifiers)) != len(identifiers):
        raise ProjectionConfigurationError("variant ids must be unique")
    try:
        config = ProjectionConfig(
            projection_id=str(data["projection_id"]),
            evaluator_version=str(data["evaluator_version"]),
            corpus_path=Path(str(data["corpus_path"])),
            training_data_manifest=Path(str(data["training_data_manifest"])),
            measured_validation_metrics=Path(str(data["measured_validation_metrics"])),
            output_dir=Path(str(data["output_dir"])),
            primary_variant_id=str(data["primary_variant_id"]),
            measured_variant_id=str(data["measured_variant_id"]),
            confusability_prior=_confusability(data["confusability_prior"]),
            variants=tuple(variants),
            learning_curve=dict(data["learning_curve"]),
            simulation=dict(data["simulation"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectionConfigurationError(f"invalid projection config: {error}") from error
    references = (config.primary_variant_id, config.measured_variant_id, str(config.simulation.get("comparison_baseline_variant_id")))
    for required in references:
        if required not in identifiers:
            raise ProjectionConfigurationError(f"unknown variant reference {required!r}")
    return config


def _confusability(raw: object) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(raw, Mapping):
        raise ProjectionConfigurationError("confusability_prior must be an object")
    prior: dict[str, dict[str, float]] = {}
    for label in LABELS:
        row = raw.get(label)
        if not isinstance(row, Mapping) or set(row) != set(LABELS) - {label}:
            raise ProjectionConfigurationError(f"confusability_prior[{label}] must cover every other label exactly once")
        weights = {str(key): float(value) for key, value in row.items()}
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ProjectionConfigurationError(f"confusability_prior[{label}] must hold positive weights")
        total = sum(weights.values())
        prior[label] = {key: value / total for key, value in weights.items()}
    return prior


def label_supports(corpus_path: Path, manifest: Mapping[str, object]) -> dict[str, dict[str, int]]:
    """Count how many cases in each split carry a recoverable disposition label."""
    supports = {split: {label: 0 for label in LABELS} for split in _SPLIT_FIELDS}
    unlabelled = {split: 0 for split in _SPLIT_FIELDS}
    lookup: dict[str, str] = {}
    for split, field in _SPLIT_FIELDS.items():
        for origin in _splits(manifest, field):
            lookup[origin] = split
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                origin_split = str(row["origin_split"])
                text = str(row["text"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ProjectionConfigurationError(f"invalid corpus record at line {line_number}") from error
            split = lookup.get(origin_split)
            if split is None:
                continue
            label = infer_disposition(text[int(len(text) * 0.8) :])
            if label in LABELS:
                supports[split][label] += 1
            else:
                unlabelled[split] += 1
    for split, counts in supports.items():
        counts["_unlabelled"] = unlabelled[split]
    return supports


def _splits(manifest: Mapping[str, object], name: str) -> frozenset[str]:
    values = manifest.get(name)
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ProjectionConfigurationError(f"invalid manifest field {name}")
    return frozenset(values)


def projected_confusion(
    supports: Mapping[str, int],
    class_recall: Mapping[str, float],
    confusability: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, int]]:
    """Turn per-class recall into an integer confusion matrix that preserves each support total."""
    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for actual in LABELS:
        support = int(supports.get(actual, 0))
        if support <= 0:
            continue
        correct = min(support, max(0, _round_half_up(support * float(class_recall[actual]))))
        confusion[actual][actual] = correct
        errors = support - correct
        if errors <= 0:
            continue
        weights = confusability[actual]
        allocation = _largest_remainder({label: weights[label] * errors for label in weights}, errors)
        for predicted, count in allocation.items():
            confusion[actual][predicted] += count
    return confusion


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _largest_remainder(shares: Mapping[str, float], total: int) -> dict[str, int]:
    floors = {key: int(math.floor(value)) for key, value in shares.items()}
    remainder = total - sum(floors.values())
    ordered = sorted(shares, key=lambda key: (-(shares[key] - floors[key]), key))
    for key in ordered[:remainder]:
        floors[key] += 1
    return floors


def confusion_to_sequences(confusion: Mapping[str, Mapping[str, int]]) -> tuple[list[str], list[str]]:
    gold: list[str] = []
    predicted: list[str] = []
    for actual in LABELS:
        for forecast in LABELS:
            count = int(confusion[actual][forecast])
            gold.extend([actual] * count)
            predicted.extend([forecast] * count)
    return gold, predicted


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, float]:
    """Wilson score interval, which stays inside [0, 1] at the sample sizes this corpus offers."""
    if trials <= 0:
        raise ProjectionConfigurationError("wilson_interval requires a positive trial count")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return {
        "point": proportion,
        "lower": max(0.0, centre - spread),
        "upper": min(1.0, centre + spread),
        "half_width": spread,
    }


def learning_curve(settings: Mapping[str, object], anchor_accuracy: float) -> list[dict[str, float | int | bool]]:
    """Power-law data-scaling curve pinned to the projected accuracy at the current labelled count."""
    ceiling = float(settings["ceiling"])
    exponent = float(settings["exponent"])
    anchor = int(settings["anchor_labelled_cases"])
    raw_points = settings["labelled_case_points"]
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, str) or not raw_points:
        raise ProjectionConfigurationError("learning curve requires labelled_case_points")
    points = [int(value) for value in raw_points]
    if not 0 < anchor_accuracy < ceiling <= 1 or exponent <= 0 or anchor <= 0:
        raise ProjectionConfigurationError("learning curve requires 0 < anchor accuracy < ceiling <= 1")
    scale = (ceiling - anchor_accuracy) * anchor**exponent
    return [
        {
            "labelled_cases": count,
            "projected_accuracy": round(ceiling - scale / count**exponent, 6),
            "is_current_corpus": count == anchor,
        }
        for count in sorted({*points, anchor})
    ]


def simulate(accuracy: float, trials: int, sample_size: int, seed: int) -> dict[str, object]:
    """Sampling noise a single held-out run would show if the projected accuracy were exact."""
    generator = random.Random(seed)
    draws = sorted(sum(generator.random() < accuracy for _ in range(sample_size)) / sample_size for _ in range(trials))
    return {
        "trials": trials,
        "sample_size": sample_size,
        "assumed_accuracy": accuracy,
        "mean": sum(draws) / len(draws),
        "p05": draws[int(0.05 * len(draws))],
        "p25": draws[int(0.25 * len(draws))],
        "median": draws[len(draws) // 2],
        "p75": draws[int(0.75 * len(draws))],
        "p95": draws[int(0.95 * len(draws))],
        "within_5_points": sum(abs(value - accuracy) <= 0.05 for value in draws) / len(draws),
        "histogram": _histogram(draws, sample_size),
    }


def _histogram(draws: Sequence[float], sample_size: int) -> list[dict[str, float | int]]:
    counts: dict[int, int] = {}
    for value in draws:
        correct = round(value * sample_size)
        counts[correct] = counts.get(correct, 0) + 1
    return [
        {"accuracy": correct / sample_size, "correct": correct, "share": counts[correct] / len(draws)}
        for correct in sorted(counts)
    ]


def separation_probability(
    candidate_accuracy: float, reference_accuracy: float, sample_size: int, trials: int, seed: int
) -> float:
    """Chance a single run of this size ranks the candidate above the comparison variant."""
    generator = random.Random(seed)
    wins = 0
    for _ in range(trials):
        candidate = sum(generator.random() < candidate_accuracy for _ in range(sample_size))
        reference = sum(generator.random() < reference_accuracy for _ in range(sample_size))
        wins += candidate > reference
    return wins / trials


def run(config: ProjectionConfig, workspace_root: Path) -> dict[str, object]:
    corpus_path = workspace_root / config.corpus_path
    manifest_path = workspace_root / config.training_data_manifest
    measured_path = workspace_root / config.measured_validation_metrics
    output_dir = workspace_root / config.output_dir
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "validated":
        raise ProjectionConfigurationError("training manifest must be validated")
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    supports = label_supports(corpus_path, manifest)

    variant_rows: list[dict[str, object]] = []
    for variant in config.variants:
        splits: dict[str, object] = {}
        for split in ("validation", "test"):
            counts = {label: supports[split][label] for label in LABELS}
            confusion = projected_confusion(counts, variant.class_recall, config.confusability_prior)
            gold, predicted = confusion_to_sequences(confusion)
            metrics = evaluate(gold, predicted)
            correct = sum(confusion[label][label] for label in LABELS)
            splits[split] = {
                "eligible_case_count": len(gold),
                "correct": correct,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class": metrics["per_class"],
                "confusion_matrix": confusion,
                "accuracy_interval": wilson_interval(correct, len(gold)),
            }
        variant_rows.append(
            {
                "variant_id": variant.variant_id,
                "label": variant.label,
                "family": variant.family,
                "evidence": variant.evidence,
                "graph_evidence": variant.graph_evidence,
                "note": variant.note,
                "class_recall": dict(variant.class_recall),
                "splits": splits,
            }
        )

    by_id = {str(row["variant_id"]): row for row in variant_rows}
    measured_summary = {
        "accuracy": measured["accuracy"],
        "macro_f1": measured["macro_f1"],
        "eligible_case_count": measured["eligible_case_count"],
        "per_class": measured["per_class"],
        "confusion_matrix": measured["confusion_matrix"],
    }
    by_id[config.measured_variant_id]["measured_validation_metrics"] = measured_summary
    primary_test = _split_row(by_id[config.primary_variant_id], "test")
    comparison_id = str(config.simulation["comparison_baseline_variant_id"])
    comparison_test = _split_row(by_id[comparison_id], "test")

    ladder_start = float(measured["accuracy"])
    uplift: list[dict[str, object]] = []
    previous = ladder_start
    for row in variant_rows:
        if row["family"] in {"trivial", "ceiling"} or row["variant_id"] == config.measured_variant_id:
            continue
        current = float(_split_row(row, "test")["accuracy"])
        uplift.append(
            {
                "variant_id": row["variant_id"],
                "label": row["label"],
                "from_accuracy": previous,
                "to_accuracy": current,
                "delta": current - previous,
                "cumulative_delta": current - ladder_start,
            }
        )
        previous = current

    trials = int(config.simulation["trials"])
    seed = int(config.simulation["seed"])
    primary_accuracy = float(primary_test["accuracy"])
    sample_size = int(primary_test["eligible_case_count"])
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "projected_not_measured",
        "projection_id": config.projection_id,
        "evaluator_version": config.evaluator_version,
        "method": "class_recall_uplift_projection_v1",
        "disclaimer": "Every non-measured number is a planning forecast derived from the config assumptions, not an experimental result.",
        "corpus_sha256": _sha256_file(corpus_path),
        "training_manifest_sha256": _sha256_file(manifest_path),
        "measured_metrics_sha256": _sha256_file(measured_path),
        "label_schema": list(LABELS),
        "split_counts": manifest.get("split_counts"),
        "label_supports": supports,
        "measured_baseline": measured_summary,
        "primary_variant_id": config.primary_variant_id,
        "headline": {
            "primary_projected_test_accuracy": primary_accuracy,
            "primary_projected_test_macro_f1": primary_test["macro_f1"],
            "primary_projected_validation_accuracy": _split_row(by_id[config.primary_variant_id], "validation")["accuracy"],
            "measured_baseline_accuracy": measured["accuracy"],
            "absolute_gain_over_measured_baseline": primary_accuracy - ladder_start,
            "accuracy_interval": primary_test["accuracy_interval"],
        },
        "variants": variant_rows,
        "uplift_decomposition": uplift,
        "learning_curve": learning_curve(config.learning_curve, primary_accuracy),
        "sampling_noise": simulate(primary_accuracy, trials, sample_size, seed),
        "separation_power": {
            "comparison_variant_id": comparison_id,
            "comparison_accuracy": comparison_test["accuracy"],
            "points": [
                {
                    "sample_size": size,
                    "probability_primary_ranks_higher": separation_probability(
                        primary_accuracy, float(comparison_test["accuracy"]), size, 2000, seed + size
                    ),
                }
                for size in (50, 100, 200, 400, 800)
            ],
        },
    }
    payload["content_sha256"] = _payload_sha256(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "outcome-projection.json"
    report_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {
        "status": payload["status"],
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "primary_variant_id": config.primary_variant_id,
        "primary_projected_test_accuracy": primary_accuracy,
        "primary_projected_test_macro_f1": primary_test["macro_f1"],
        "measured_baseline_accuracy": measured["accuracy"],
    }


def _split_row(variant_row: Mapping[str, object], split: str) -> Mapping[str, object]:
    splits = variant_row["splits"]
    if not isinstance(splits, Mapping):
        raise ProjectionConfigurationError("variant row is missing split projections")
    row = splits[split]
    if not isinstance(row, Mapping):
        raise ProjectionConfigurationError(f"variant row is missing the {split} projection")
    return row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    material = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
