"""Split-aware, leakage-controlled outcome examples for the model runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .models import GraphSnapshot, NodeType
from .outcomes import LABELS, infer_disposition, input_view
from .training import _sha256_file

_FIELDS = {"train": "training_splits", "validation": "validation_splits", "test": "held_out_splits"}


@dataclass(frozen=True, slots=True)
class OutcomeExample:
    case_id: str
    document_id: str
    split: str
    view_text: str
    label: str | None
    decision_date: date | None
    year: int | None = None
    source: str = ""  # the dataset the case came from, e.g. SC-2016 or the AWS bucket

    @property
    def as_of(self) -> date | None:
        """Retrieval cut-off: the decision date, or 1 January of the judgment year when the date is unknown.

        The fallback is deliberately conservative: it admits nothing enacted or decided later in that year.
        """
        if self.decision_date is not None:
            return self.decision_date
        return date(self.year, 1, 1) if self.year else None


def load_outcome_examples(
    corpus_path: Path,
    manifest_path: Path,
    split: str,
    character_limit: int,
    *,
    allow_test_labels: bool = False,
    input_view_id: str = "prefix-with-disposition-masking-v1",
) -> tuple[OutcomeExample, ...]:
    """Masked-view examples for one split; test labels stay hidden unless explicitly allowed."""
    if split not in _FIELDS:
        raise ValueError(f"unknown split {split!r}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "validated" or "test" not in manifest.get("held_out_splits", []):
        raise ValueError("training manifest must be validated and hold out test")
    if manifest.get("corpus_sha256") != _sha256_file(corpus_path):
        raise ValueError("corpus checksum does not match training manifest")
    allowed = set(manifest[_FIELDS[split]])
    hide = split == "test" and not allow_test_labels
    examples: list[OutcomeExample] = []
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["origin_split"] not in allowed:
                continue
            text = str(row["text"])
            label = infer_disposition(text[int(len(text) * 0.8) :])
            if label not in LABELS:
                label = None
            if label is None and not hide:
                continue
            view = input_view(input_view_id, text, character_limit)
            if view is None:
                continue  # no reliable judgment-body start: excluded rather than shown a headnote
            raw_date = row.get("decision_date")
            raw_year = str((row.get("attributes") or {}).get("year", "")).strip()
            examples.append(
                OutcomeExample(
                    case_id=str(row["case_id"]),
                    document_id=str(row["document_id"]),
                    split=split,
                    view_text=view,
                    label=None if hide else label,
                    decision_date=date.fromisoformat(raw_date) if raw_date else None,
                    year=int(raw_year) if raw_year.isdigit() else None,
                    source=str((row.get("attributes") or {}).get("dataset", "")),
                )
            )
    return tuple(examples)


def node_feature_texts(graph: GraphSnapshot, provisions_path: Path, max_chars: int = 400) -> dict[str, str]:
    """Short descriptive text per node; case nodes contribute their title only, never outcome text."""
    provision_text: dict[str, str] = {}
    if provisions_path.exists():
        with provisions_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    provision_text[str(row["node_id"])] = str(row.get("text", ""))[:max_chars]
    texts: dict[str, str] = {}
    for node in graph.nodes:
        if node.node_type is NodeType.PROVISION:
            texts[node.node_id] = f"{node.label}. {provision_text.get(node.node_id, '')}".strip()
        else:
            texts[node.node_id] = node.label
    return texts
