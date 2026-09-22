"""Leakage-aware pre-training outcome baseline for the SC-2016 corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


class OutcomeConfigurationError(ValueError):
    pass


LABELS = ("allowed", "dismissed", "disposed", "partly_allowed")
_OUTCOME_PATTERNS = (
    ("partly_allowed", re.compile(r"\b(?:appeal|petition|application|writ(?:\s+petition)?|special leave petition|civil appeal|criminal appeal)s?\s+(?:(?:is|are|stands?)\s+)?(?:partly|partially)\s+allowed\b", re.IGNORECASE)),
    ("allowed", re.compile(r"\b(?:appeal|petition|application|writ(?:\s+petition)?|special leave petition|civil appeal|criminal appeal)s?\s+(?:(?:is|are|stands?)\s+)?allowed\b", re.IGNORECASE)),
    ("dismissed", re.compile(r"\b(?:appeal|petition|application|writ(?:\s+petition)?|special leave petition|civil appeal|criminal appeal)s?\s+(?:(?:is|are|stands?)\s+)?dismissed\b", re.IGNORECASE)),
    ("disposed", re.compile(r"\b(?:appeal|petition|application|writ(?:\s+petition)?|special leave petition|civil appeal|criminal appeal)s?\s+(?:(?:is|are|stands?)\s+)?disposed(?:\s+of)?\b", re.IGNORECASE)),
)
_TOKEN_PATTERN = re.compile(r"[a-z][a-z-]{2,}")


@dataclass(frozen=True, slots=True)
class OutcomeBaselineConfig:
    experiment_id: str
    evaluator_version: str
    corpus_path: Path
    training_data_manifest: Path
    output_dir: Path
    input_character_limit: int
    vocabulary_size: int
    alpha: float


@dataclass(frozen=True, slots=True)
class CaseInput:
    case_id: str
    document_id: str
    title: str
    origin_split: str
    text: str


def load_config(path: Path) -> OutcomeBaselineConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OutcomeConfigurationError(f"cannot load {path}: {error}") from error
    if data.get("schema_version") != "1.0.0" or data.get("method") != "train_only_multinomial_nb_v1":
        raise OutcomeConfigurationError("config must specify schema 1.0.0 and train_only_multinomial_nb_v1")
    input_view = data.get("input_view")
    if not isinstance(input_view, Mapping) or input_view.get("id") != "prefix-with-disposition-masking-v1":
        raise OutcomeConfigurationError("config requires the approved disposition-masked input view")
    try:
        config = OutcomeBaselineConfig(
            experiment_id=str(data["experiment_id"]),
            evaluator_version=str(data["evaluator_version"]),
            corpus_path=Path(str(data["corpus_path"])),
            training_data_manifest=Path(str(data["training_data_manifest"])),
            output_dir=Path(str(data["output_dir"])),
            input_character_limit=int(input_view["character_limit"]),
            vocabulary_size=int(data["vocabulary_size"]),
            alpha=float(data["alpha"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OutcomeConfigurationError(f"invalid outcome baseline config: {error}") from error
    if config.input_character_limit <= 0 or config.vocabulary_size <= 0 or config.alpha <= 0:
        raise OutcomeConfigurationError("input_character_limit, vocabulary_size, and alpha must be positive")
    return config


def infer_disposition(text: str) -> str | None:
    """Return the latest explicit final-disposition phrase, if the text has one."""
    matches: list[tuple[int, int, str]] = []
    for label, pattern in _OUTCOME_PATTERNS:
        matches.extend((match.start(), match.end(), label) for match in pattern.finditer(text))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1], item[2]))[2]


def build_input_view(text: str, character_limit: int) -> str:
    """Expose only a leading document view with all explicit disposition phrases masked."""
    view = text[:character_limit]
    for _, pattern in _OUTCOME_PATTERNS:
        view = pattern.sub(" outcome_phrase ", view)
    return view


def run(config: OutcomeBaselineConfig, workspace_root: Path) -> dict[str, object]:
    corpus_path = workspace_root / config.corpus_path
    manifest_path = workspace_root / config.training_data_manifest
    output_dir = workspace_root / config.output_dir
    manifest = _load_manifest(manifest_path, corpus_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    cases = _read_cases(corpus_path)
    train = _labelled_cases(cases, _splits(manifest, "training_splits"))
    validation = _labelled_cases(cases, _splits(manifest, "validation_splits"))
    test = _unlabelled_cases(cases, _splits(manifest, "held_out_splits"))
    if not train or not validation or not test:
        raise OutcomeConfigurationError("train, validation, and held-out test collections must all be non-empty")
    predictor = _fit_predictor(train, config)
    validation_outputs = [_prediction(case, predictor, config) for case, _ in validation]
    validation_metrics = _metrics([label for _, label in validation], [str(item["predicted_label"]) for item in validation_outputs])
    test_outputs = [_prediction(case, predictor, config) for case in test]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "test-predictions.jsonl"
    _write_jsonl(predictions_path, test_outputs)
    validation_path = output_dir / "validation-metrics.json"
    _write_json(validation_path, validation_metrics)
    manifest_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "pre_training_baseline",
        "experiment_id": config.experiment_id,
        "evaluator_version": config.evaluator_version,
        "method": "train_only_multinomial_nb_v1",
        "corpus_sha256": _sha256_file(corpus_path),
        "training_manifest_sha256": _sha256_file(manifest_path),
        "input_view_id": "prefix-with-disposition-masking-v1",
        "training_cases_labelled": len(train),
        "validation_cases_labelled": len(validation),
        "test_cases_predicted": len(test_outputs),
        "test_labels_accessed": False,
        "test_labels_evaluated": False,
        "prediction_path": str(predictions_path),
        "prediction_sha256": _sha256_file(predictions_path),
        "validation_metrics_path": str(validation_path),
        "validation_metrics_sha256": _sha256_file(validation_path),
        "predicted_label_counts": dict(sorted(Counter(str(item["predicted_label"]) for item in test_outputs).items())),
        "mean_prediction_confidence": sum(float(item["confidence"]) for item in test_outputs) / len(test_outputs),
        "comparison_protocol": {
            "held_out_test_status": "not_read_for_labels",
            "future_candidate_requirement": "Use the same input view and four-label schema; evaluate only after candidate freeze.",
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1", "confusion_matrix"],
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    manifest_payload["content_sha256"] = _payload_sha256(manifest_payload)
    report_path = output_dir / "baseline.manifest.json"
    _write_json(report_path, manifest_payload)
    return {
        "status": manifest_payload["status"],
        "prediction_path": str(predictions_path),
        "validation_metrics_path": str(validation_path),
        "manifest_path": str(report_path),
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_cases_predicted": len(test_outputs),
        "test_labels_accessed": False,
    }


def _load_manifest(path: Path, corpus_path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OutcomeConfigurationError(f"cannot read training manifest: {error}") from error
    if not isinstance(manifest, Mapping) or manifest.get("status") != "validated":
        raise OutcomeConfigurationError("training manifest must be validated")
    if manifest.get("corpus_sha256") != _sha256_file(corpus_path):
        raise OutcomeConfigurationError("corpus checksum does not match training manifest")
    if "test" not in _splits(manifest, "held_out_splits"):
        raise OutcomeConfigurationError("training manifest must hold out test")
    return manifest


def _splits(manifest: Mapping[str, object], name: str) -> frozenset[str]:
    values = manifest.get(name)
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise OutcomeConfigurationError(f"invalid manifest field {name}")
    return frozenset(values)


def _read_cases(path: Path) -> tuple[CaseInput, ...]:
    cases: list[CaseInput] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
                case = CaseInput(
                    case_id=str(row["case_id"]),
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    origin_split=str(row["origin_split"]),
                    text=str(row["text"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise OutcomeConfigurationError(f"invalid corpus record at line {line_number}") from error
            if not case.text.strip():
                raise OutcomeConfigurationError(f"empty text at line {line_number}")
            cases.append(case)
    return tuple(cases)


def _labelled_cases(cases: Sequence[CaseInput], allowed_splits: frozenset[str]) -> tuple[tuple[CaseInput, str], ...]:
    labelled: list[tuple[CaseInput, str]] = []
    for case in cases:
        if case.origin_split not in allowed_splits:
            continue
        label = infer_disposition(case.text[int(len(case.text) * 0.8) :])
        if label in LABELS:
            labelled.append((case, label))
    return tuple(labelled)


def _unlabelled_cases(cases: Sequence[CaseInput], held_out_splits: frozenset[str]) -> tuple[CaseInput, ...]:
    return tuple(case for case in cases if case.origin_split in held_out_splits)


def _fit_predictor(train: Sequence[tuple[CaseInput, str]], config: OutcomeBaselineConfig) -> Mapping[str, object]:
    document_frequency: Counter[str] = Counter()
    for case, _ in train:
        document_frequency.update(set(_tokens(build_input_view(case.text, config.input_character_limit))))
    vocabulary = frozenset(token for token, _ in document_frequency.most_common(config.vocabulary_size))
    class_documents: Counter[str] = Counter(label for _, label in train)
    class_tokens: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    class_totals: Counter[str] = Counter()
    for case, label in train:
        tokens = [token for token in _tokens(build_input_view(case.text, config.input_character_limit)) if token in vocabulary]
        class_tokens[label].update(tokens)
        class_totals[label] += len(tokens)
    return {
        "vocabulary": vocabulary,
        "class_documents": class_documents,
        "class_tokens": class_tokens,
        "class_totals": class_totals,
        "document_count": len(train),
        "alpha": config.alpha,
    }


def _prediction(case: CaseInput, predictor: Mapping[str, object], config: OutcomeBaselineConfig) -> dict[str, object]:
    vocabulary = predictor["vocabulary"]
    class_documents = predictor["class_documents"]
    class_tokens = predictor["class_tokens"]
    class_totals = predictor["class_totals"]
    document_count = int(predictor["document_count"])
    alpha = float(predictor["alpha"])
    tokens = [token for token in _tokens(build_input_view(case.text, config.input_character_limit)) if token in vocabulary]
    frequencies = Counter(tokens)
    scores: dict[str, float] = {}
    for label in LABELS:
        prior = (int(class_documents[label]) + alpha) / (document_count + alpha * len(LABELS))
        denominator = int(class_totals[label]) + alpha * len(vocabulary)
        score = math.log(prior)
        score += sum(count * math.log((int(class_tokens[label][token]) + alpha) / denominator) for token, count in frequencies.items())
        scores[label] = score
    maximum = max(scores.values())
    probabilities = {label: math.exp(score - maximum) for label, score in scores.items()}
    normaliser = sum(probabilities.values())
    probabilities = {label: value / normaliser for label, value in probabilities.items()}
    label = max(LABELS, key=lambda candidate: (probabilities[candidate], candidate))
    input_view = build_input_view(case.text, config.input_character_limit)
    return {
        "case_id": case.case_id,
        "document_id": case.document_id,
        "title": case.title,
        "origin_split": case.origin_split,
        "input_view_id": "prefix-with-disposition-masking-v1",
        "input_sha256": hashlib.sha256(input_view.encode("utf-8")).hexdigest(),
        "predicted_label": label,
        "confidence": round(probabilities[label], 8),
        "label_probabilities": {candidate: round(probabilities[candidate], 8) for candidate in LABELS},
    }


def evaluate(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, object]:
    """Public entry point for the shared four-label evaluator."""
    return _metrics(gold, predicted)


def _metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, object]:
    if len(gold) != len(predicted) or not gold:
        raise OutcomeConfigurationError("metrics require equally sized non-empty gold and prediction values")
    confusion = {label: {candidate: 0 for candidate in LABELS} for label in LABELS}
    for actual, forecast in zip(gold, predicted, strict=True):
        confusion[actual][forecast] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in LABELS if actual != label)
        false_negative = sum(confusion[label][forecast] for forecast in LABELS if forecast != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": sum(confusion[label].values()), "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "metric_version": "sc2016-outcome-evaluator-1.0.0",
        "label_schema": list(LABELS),
        "eligible_case_count": len(gold),
        "accuracy": sum(actual == forecast for actual, forecast in zip(gold, predicted, strict=True)) / len(gold),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    material = {key: value for key, value in payload.items() if key != "content_sha256"}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
