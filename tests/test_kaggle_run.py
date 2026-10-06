from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KaggleRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load("kaggle_run")

    def test_metadata_is_private_gpu_notebook(self) -> None:
        meta = self.module.kernel_metadata("akxran/lg-x-train", "lg-x-train", "lg-x-train.ipynb")
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertEqual(meta["kernel_sources"], [])

    def test_slug_is_kaggle_safe(self) -> None:
        self.assertEqual(self.module.kernel_slug("qwen25-0.5b-smoke-text", "train"), "lg-qwen25-0-5b-smoke-text-train")
        with self.assertRaises(ValueError):
            self.module.kernel_slug("x" * 60, "train")

    def test_status_parsing(self) -> None:
        self.assertEqual(self.module.parse_status('akxran/k has status "complete"'), "complete")
        self.assertEqual(self.module.parse_status("akxran/k has status KernelWorkerStatus.RUNNING"), "running")

    def test_push_refuses_public_kernel(self) -> None:
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "kernel-metadata.json").write_text(json.dumps({"is_private": False}), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.module.push(folder)


@unittest.skipUnless((ROOT / "data/clean/judgments/training.manifest.json").exists()
                     and (ROOT / "data/clean/india_statutes/provisions.jsonl").exists(), "local pipeline outputs absent")
class OutcomeNotebookTests(unittest.TestCase):
    def test_writes_train_and_evaluate_kernels(self) -> None:
        builder = _load("build_kaggle_notebook")
        with TemporaryDirectory() as directory:
            train = builder.write_outcome_kernel("qwen3-4b-graph-typed-s1", "train", Path(directory) / "t", "akxran")
            evaluate = builder.write_outcome_kernel("qwen3-4b-graph-typed-s1", "evaluate", Path(directory) / "e", "akxran")
            train_meta = json.loads((train / "kernel-metadata.json").read_text(encoding="utf-8"))
            eval_meta = json.loads((evaluate / "kernel-metadata.json").read_text(encoding="utf-8"))
            notebook = json.loads((train / train_meta["code_file"]).read_text(encoding="utf-8"))
        source = "".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("train-outcome", source)
        self.assertIn("freeze-outcome", source)
        self.assertNotIn("evaluate-outcome", source)
        self.assertIn("configs/training/outcome/qwen3-4b-graph-typed-s1.json", source)
        self.assertEqual(eval_meta["kernel_sources"], ["akxran/lg-qwen3-4b-graph-typed-s1-train"])
        self.assertTrue(train_meta["is_private"] and eval_meta["is_private"])
