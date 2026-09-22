"""Tests for the split-safe pre-training outcome baseline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_graph.outcomes import OutcomeBaselineConfig, build_input_view, infer_disposition, run


class OutcomeBaselineTests(unittest.TestCase):
    def test_masks_explicit_disposition(self) -> None:
        text = "The appeal is allowed. Other context remains."
        self.assertEqual(infer_disposition(text), "allowed")
        self.assertNotIn("appeal is allowed", build_input_view(text, 100).lower())



    def test_writes_held_out_predictions_without_gold_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            records = [
                _record("train-allow", "train", "facts " * 40 + "The appeal is allowed."),
                _record("train-dismiss", "train", "facts " * 40 + "The appeal is dismissed."),
                _record("validation-allow", "validation", "facts " * 40 + "The appeal is allowed."),
                _record("test-dismiss", "test", "facts " * 40 + "The appeal is dismissed."),
            ]
            corpus.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            manifest = root / "manifest.json"
            import hashlib
            manifest.write_text(json.dumps({
                "status": "validated",
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "training_splits": ["train"],
                "validation_splits": ["validation"],
                "held_out_splits": ["test"],
            }), encoding="utf-8")
            result = run(OutcomeBaselineConfig(
                experiment_id="test",
                evaluator_version="test-1",
                corpus_path=Path("corpus.jsonl"),
                training_data_manifest=Path("manifest.json"),
                output_dir=Path("out"),
                input_character_limit=1000,
                vocabulary_size=100,
                alpha=1.0,
            ), root)
            predictions = [json.loads(line) for line in (root / result["prediction_path"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(predictions), 1)
            self.assertNotIn("gold_label", predictions[0])
            self.assertNotIn("actual_label", predictions[0])
            self.assertFalse(result["test_labels_accessed"])


def _record(case_id: str, split: str, text: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "document_id": f"doc:{case_id}",
        "title": case_id,
        "origin_split": split,
        "text": text,
    }
