from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.outcome_examples import load_outcome_examples, node_feature_texts
from legal_graph.training import _sha256_file

BODY = "Facts of the matter. " * 50


def _write(root: Path, extra_rows: tuple[dict, ...] = ()) -> tuple[Path, Path]:
    rows = [
        {"case_id": "c1", "document_id": "d1", "origin_split": "train", "decision_date": "2016-01-02",
         "text": "Section 482 of the CrPC was invoked. " + BODY + "The appeal is allowed."},
        {"case_id": "c2", "document_id": "d2", "origin_split": "train", "decision_date": None, "text": BODY + "Ordered accordingly."},
        {"case_id": "c3", "document_id": "d3", "origin_split": "test", "decision_date": "2016-03-04",
         "text": "The appeal is dismissed early. " + BODY + "The appeal is dismissed."},
        *extra_rows,
    ]
    corpus = root / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest = root / "m.json"
    manifest.write_text(json.dumps({
        "artifact_type": "judgment_training_corpus",
        "status": "validated",
        "corpus_sha256": _sha256_file(corpus),
        "training_splits": ["train"],
        "validation_splits": ["validation"],
        "held_out_splits": ["test"],
    }), encoding="utf-8")
    return corpus, manifest


class ExampleTests(unittest.TestCase):
    def test_train_drops_unlabelled_and_masks_view(self) -> None:
        with TemporaryDirectory() as directory:
            corpus, manifest = _write(Path(directory))
            train = load_outcome_examples(corpus, manifest, "train", 6000)
        self.assertEqual([(example.case_id, example.label) for example in train], [("c1", "allowed")])
        self.assertNotIn("allowed", train[0].view_text.replace("outcome_phrase", ""))
        self.assertEqual(str(train[0].decision_date), "2016-01-02")

    def test_test_labels_hidden_by_default(self) -> None:
        with TemporaryDirectory() as directory:
            corpus, manifest = _write(Path(directory))
            hidden = load_outcome_examples(corpus, manifest, "test", 6000)
            shown = load_outcome_examples(corpus, manifest, "test", 6000, allow_test_labels=True)
        self.assertEqual([example.label for example in hidden], [None])
        self.assertEqual([example.label for example in shown], ["dismissed"])
        self.assertIn("outcome_phrase", hidden[0].view_text)

    def test_rejects_corpus_hash_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            corpus, manifest = _write(Path(directory))
            corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_outcome_examples(corpus, manifest, "train", 6000)

    def test_view_respects_character_limit(self) -> None:
        with TemporaryDirectory() as directory:
            corpus, manifest = _write(Path(directory))
            train = load_outcome_examples(corpus, manifest, "train", 50)
        self.assertLessEqual(len(train[0].view_text), 50)

    def test_case_feature_text_is_title_only(self) -> None:
        from tests.test_query_graph import _graph

        with TemporaryDirectory() as directory:
            provisions = Path(directory) / "p.jsonl"
            provisions.write_text(json.dumps({"node_id": "provision:india:crpc:482:sha256-x", "text": "Saving of inherent powers."}) + "\n", encoding="utf-8")
            texts = node_feature_texts(_graph(), provisions)
        self.assertEqual(texts["case:train"], "A v B")
        self.assertEqual(texts["provision:india:crpc:482:sha256-x"], "CrPC section 482. Saving of inherent powers.")


class AsOfDateTests(unittest.TestCase):
    def test_decision_date_wins(self) -> None:
        from datetime import date
        from legal_graph.outcome_examples import OutcomeExample

        self.assertEqual(OutcomeExample("c", "d", "train", "v", "allowed", date(2016, 5, 1), 2016).as_of, date(2016, 5, 1))

    def test_undated_case_falls_back_to_first_day_of_its_year(self) -> None:
        from datetime import date
        from legal_graph.outcome_examples import OutcomeExample

        self.assertEqual(OutcomeExample("c", "d", "train", "v", "allowed", None, 2016).as_of, date(2016, 1, 1))
        self.assertIsNone(OutcomeExample("c", "d", "train", "v", "allowed", None).as_of)

    def test_loader_reads_the_year_attribute(self) -> None:
        with TemporaryDirectory() as directory:
            corpus, manifest = _write(Path(directory), ({
                "case_id": "c9", "document_id": "d9", "origin_split": "train", "decision_date": None,
                "attributes": {"year": "2016"}, "text": BODY + "The appeal is dismissed."},))
            train = {e.case_id: e for e in load_outcome_examples(corpus, manifest, "train", 6000)}
        self.assertEqual(str(train["c9"].as_of), "2016-01-01")
        self.assertIsNone(train["c1"].year)
