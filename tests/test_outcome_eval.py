from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from legal_graph.outcome_eval import freeze_run, verify_freeze


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run(root: Path) -> Path:
    run = root / "run"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "encoder.pt").write_bytes(b"encoder")
    (run / "config.json").write_text("{}", encoding="utf-8")
    return run


class FreezeTests(unittest.TestCase):
    def test_freeze_then_verify_passes(self) -> None:
        with TemporaryDirectory() as directory:
            run = _run(Path(directory))
            self.assertEqual(freeze_run(run)["status"], "frozen")
            verify_freeze(run)

    def test_change_after_freeze_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            run = _run(Path(directory))
            freeze_run(run)
            (run / "encoder.pt").write_bytes(b"retrained")
            with self.assertRaises(ValueError):
                verify_freeze(run)

    def test_new_adapter_file_after_freeze_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            run = _run(Path(directory))
            freeze_run(run)
            (run / "adapter" / "extra.bin").write_bytes(b"x")
            with self.assertRaises(ValueError):
                verify_freeze(run)

    def test_double_freeze_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            run = _run(Path(directory))
            freeze_run(run)
            with self.assertRaises(FileExistsError):
                freeze_run(run)

    def test_unfrozen_run_cannot_be_verified(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify_freeze(_run(Path(directory)))


@unittest.skipUnless(_has("torch") and _has("transformers") and _has("peft"), "ML stack not installed")
class FrozenEvaluationTests(unittest.TestCase):
    def test_train_freeze_evaluate_once_and_ablate(self) -> None:
        from legal_graph.outcome_eval import evaluate_frozen
        from legal_graph.outcome_training import load_outcome_config, train_outcome
        from tests.test_outcome_training import CharTokenizer, tiny_model, write_workspace

        with TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}):
            root = Path(directory)
            train_outcome(load_outcome_config(write_workspace(root, "graph_typed")), root,
                          model=tiny_model(), tokenizer=CharTokenizer())
            run = root / "run"
            with self.assertRaises(ValueError):
                evaluate_frozen(run, root, base_model=tiny_model(), tokenizer=CharTokenizer())
            freeze_run(run)
            result = evaluate_frozen(run, root, base_model=tiny_model(), tokenizer=CharTokenizer())
            self.assertEqual(result["status"], "measured_on_test")
            self.assertEqual(result["metrics"]["eligible_case_count"], 1)
            self.assertTrue((run / "test-predictions.jsonl").exists())
            with self.assertRaises(FileExistsError):
                evaluate_frozen(run, root, base_model=tiny_model(), tokenizer=CharTokenizer())
            ablated = evaluate_frozen(run, root, ablate_graph=True, base_model=tiny_model(), tokenizer=CharTokenizer())
            self.assertTrue(ablated["ablate_graph"])
            verify_freeze(run)
