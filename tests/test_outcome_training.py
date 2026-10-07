from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from legal_graph.outcome_training import config_from_dict, config_to_dict, load_outcome_config


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


ML_STACK = _has("torch") and _has("transformers") and _has("peft")


class CharTokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False,
                 max_length: int | None = None, **_: object) -> dict[str, list[int]]:
        ids = [2 + (ord(character) % 120) for character in text]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def tiny_model():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(11)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
    ))


VALIDATION_ROW = {
    "case_id": "c4", "document_id": "d4", "origin_split": "validation", "decision_date": "2016-05-01",
    "text": "Section 482 of the CrPC. " + "x " * 40 + "The appeal is dismissed.",
}


def write_workspace(root: Path, variant: str, *, with_validation: bool = True, extra_rows: tuple = ()) -> Path:
    from legal_graph.manifests import graph_to_dict
    from tests.test_outcome_examples import _write
    from tests.test_query_graph import _graph

    corpus, manifest = _write(root, ((VALIDATION_ROW,) if with_validation else ()) + tuple(extra_rows))
    (root / "graph.json").write_text(json.dumps(graph_to_dict(_graph())), encoding="utf-8")
    config = {
        "schema_version": "1.0.0", "method": "graph_outcome_qlora_v1", "experiment_id": f"tiny-{variant}",
        "variant": variant, "selector": "expansion", "model_source": "tiny", "model_revision": "none",
        "local_model_path": "unused", "corpus_path": str(corpus), "training_data_manifest": str(manifest),
        "graph_path": str(root / "graph.json"), "provisions_path": str(root / "none.jsonl"),
        "output_dir": str(root / "run"), "input_character_limit": 300, "max_prompt_tokens": 400,
        "epochs": 2, "learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "gradient_accumulation": 1,
        "seed": 7, "quantize_4bit": False, "min_cuda_memory_bytes": 0, "max_train_examples": 8,
        "lora": {"rank": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
        "graph": {"num_tokens": 2, "hidden_dim": 8, "layers": 2, "node_budget": 6, "expansion_hops": 2, "node_feature_tokens": 8},
    }
    path = root / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class ConfigTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            config = load_outcome_config(write_workspace(Path(directory), "graph_typed"))
        self.assertEqual(config_from_dict(config_to_dict(config)), config)

    def test_rejects_unknown_variant_and_misused_random_selector(self) -> None:
        with TemporaryDirectory() as directory:
            data = json.loads(write_workspace(Path(directory), "graph_typed").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            config_from_dict({**data, "variant": "graph_magic"})
        with self.assertRaises(ValueError):
            config_from_dict({**data, "selector": "random"})
        self.assertEqual(config_from_dict({**data, "variant": "graph_random"}).selector, "random")


@unittest.skipUnless(ML_STACK, "ML stack not installed")
class TinyTrainingTests(unittest.TestCase):
    def test_refuses_without_validation_cases(self) -> None:
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_outcome_config(write_workspace(root, "graph_typed", with_validation=False))
            with self.assertRaises(ValueError):
                train_outcome(config, root, model=tiny_model(), tokenizer=CharTokenizer())

    def test_parameter_cap_is_enforced(self) -> None:
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_outcome_config(write_workspace(root, "graph_typed"))  # encoder pushes the tiny model past 2%
            with self.assertRaises(RuntimeError):
                train_outcome(config, root, model=tiny_model(), tokenizer=CharTokenizer())

    def test_graph_typed_run_writes_artifacts(self) -> None:
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}):
            root = Path(directory)
            result = train_outcome(load_outcome_config(write_workspace(root, "graph_typed")), root,
                                   model=tiny_model(), tokenizer=CharTokenizer())
            run = root / "run"
            self.assertEqual(result["status"], "trained_not_frozen")
            self.assertFalse(result["test_labels_accessed"])
            self.assertTrue((run / "encoder.pt").exists())
            self.assertTrue((run / "adapter" / "adapter_config.json").exists())
            self.assertEqual(result["validation_examples"], 1)
            self.assertEqual(len(result["history"]), 2)
            self.assertEqual(result["subgraph_coverage"]["validation"]["nonempty_share"], 1.0)
            reloaded = load_outcome_config(run / "config.json")
            self.assertEqual(reloaded.variant, "graph_typed")

    def test_text_run_has_no_encoder_or_coverage(self) -> None:
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}):
            root = Path(directory)
            result = train_outcome(load_outcome_config(write_workspace(root, "text")), root,
                                   model=tiny_model(), tokenizer=CharTokenizer())
            self.assertFalse((root / "run" / "encoder.pt").exists())
            self.assertIsNone(result["subgraph_coverage"]["train"])
            self.assertEqual(result["parameter_report"]["encoder_parameters"], 0)

    def test_refuses_non_empty_output(self) -> None:
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run").mkdir()
            (root / "run" / "keep.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_outcome(load_outcome_config(write_workspace(root, "text")), root,
                              model=tiny_model(), tokenizer=CharTokenizer())

    def test_scoring_covers_all_labels(self) -> None:
        import torch
        from legal_graph.outcome_examples import OutcomeExample
        from legal_graph.outcome_training import OutcomeModel

        wrapper = OutcomeModel(tiny_model().eval(), CharTokenizer(), None, 600, torch.device("cpu"))
        example = OutcomeExample("c", "d", "validation", "Some judgment text.", "allowed", None)
        scores = wrapper.score(example, None)
        self.assertEqual(set(scores), {"allowed", "dismissed", "disposed", "partly_allowed"})
        self.assertTrue(all(value < 0 for value in scores.values()))
        self.assertIsNone(wrapper.prefix_for(None))


TRAIN_ROWS = tuple(
    {"case_id": f"t{i}", "document_id": f"dt{i}", "origin_split": "train", "decision_date": "2016-02-01",
     "text": f"Section 482 of the CrPC case {i}. " + "y " * 30 + ("The appeal is allowed." if i % 2 else "The appeal is dismissed.")}
    for i in range(6)
)


@unittest.skipUnless(ML_STACK, "ML stack not installed")
class ResumeTests(unittest.TestCase):
    def _config(self, root: Path):
        path = write_workspace(root, "graph_typed", extra_rows=TRAIN_ROWS)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["checkpoint_every_examples"] = 2
        path.write_text(json.dumps(data), encoding="utf-8")
        return load_outcome_config(path)

    def test_interrupted_run_resumes_to_the_same_result(self) -> None:
        import torch
        from legal_graph import outcome_training
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir,                 mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}):
            first, second = Path(first_dir), Path(second_dir)
            reference = train_outcome(self._config(first), first, model=tiny_model(), tokenizer=CharTokenizer())

            original = outcome_training.OutcomeModel.loss
            calls = {"n": 0}

            def crash_after_ten(self, example, sub):
                calls["n"] += 1
                if calls["n"] == 10:  # epoch 2, after the step-2 and step-4 checkpoints of that epoch
                    raise RuntimeError("simulated session end")
                return original(self, example, sub)

            config = self._config(second)
            with mock.patch.object(outcome_training.OutcomeModel, "loss", crash_after_ten):
                with self.assertRaises(RuntimeError):
                    train_outcome(config, second, model=tiny_model(), tokenizer=CharTokenizer())
            self.assertTrue((second / "run" / "latest-checkpoint" / "state.json").exists())
            with self.assertRaises(FileExistsError):
                train_outcome(config, second, model=tiny_model(), tokenizer=CharTokenizer())
            resumed = train_outcome(config, second, model=tiny_model(), tokenizer=CharTokenizer(), resume=True)

            self.assertTrue(resumed["resumed"])
            self.assertEqual(resumed["history"][-1]["train_loss"], reference["history"][-1]["train_loss"])
            self.assertEqual(resumed["validation_metrics"], reference["validation_metrics"])
            self.assertFalse((second / "run" / "latest-checkpoint").exists())
            first_encoder = torch.load(first / "run" / "encoder.pt")
            second_encoder = torch.load(second / "run" / "encoder.pt")
            for key in first_encoder:
                self.assertTrue(torch.equal(first_encoder[key], second_encoder[key]), key)
            with self.assertRaises(FileExistsError):
                train_outcome(config, second, model=tiny_model(), tokenizer=CharTokenizer(), resume=True)

    def test_resume_refuses_a_changed_config(self) -> None:
        from dataclasses import replace
        from legal_graph import outcome_training
        from legal_graph.outcome_training import train_outcome

        with TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}):
            root = Path(directory)
            config = self._config(root)
            original = outcome_training.OutcomeModel.loss
            calls = {"n": 0}

            def crash(self, example, sub):
                calls["n"] += 1
                if calls["n"] == 5:
                    raise RuntimeError("stop")
                return original(self, example, sub)

            with mock.patch.object(outcome_training.OutcomeModel, "loss", crash):
                with self.assertRaises(RuntimeError):
                    train_outcome(config, root, model=tiny_model(), tokenizer=CharTokenizer())
            with self.assertRaises(ValueError):
                train_outcome(replace(config, learning_rate=0.5), root, model=tiny_model(), tokenizer=CharTokenizer(), resume=True)

    def test_checkpoint_interval_must_respect_accumulation(self) -> None:
        with TemporaryDirectory() as directory:
            data = json.loads(write_workspace(Path(directory), "text").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            config_from_dict({**data, "gradient_accumulation": 4, "checkpoint_every_examples": 6})
        self.assertEqual(config_from_dict({**data, "gradient_accumulation": 4, "checkpoint_every_examples": 8}).checkpoint_every_examples, 8)


class LearningRateScheduleTests(unittest.TestCase):
    def test_constant_without_warmup_is_identity(self) -> None:
        from legal_graph.outcome_training import _lr_lambda

        factor = _lr_lambda(100, 0.0, "constant")
        self.assertEqual([factor(0), factor(50), factor(99)], [1.0, 1.0, 1.0])

    def test_warmup_then_linear_decay(self) -> None:
        from legal_graph.outcome_training import _lr_lambda

        factor = _lr_lambda(100, 0.1, "linear")
        self.assertAlmostEqual(factor(0), 0.1)
        self.assertAlmostEqual(factor(9), 1.0)
        self.assertAlmostEqual(factor(10), 1.0)
        self.assertAlmostEqual(factor(55), 0.5)
        self.assertEqual(factor(100), 0.0)

    def test_invalid_schedule_settings_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            data = json.loads(write_workspace(Path(directory), "text").read_text(encoding="utf-8"))
        for bad in ({"max_grad_norm": 0}, {"warmup_ratio": 1.0}, {"lr_schedule": "cosine"}):
            with self.assertRaises(ValueError):
                config_from_dict({**data, **bad})


@unittest.skipUnless(ML_STACK, "ML stack not installed")
class NonFiniteTests(unittest.TestCase):
    def _run(self, root: Path, nan_calls):
        import torch
        from legal_graph import outcome_training
        from legal_graph.outcome_training import train_outcome

        path = write_workspace(root, "text", extra_rows=TRAIN_ROWS)
        original = outcome_training.OutcomeModel.loss
        calls = {"n": 0}

        def maybe_nan(self, example, sub):
            calls["n"] += 1
            loss = original(self, example, sub)
            return loss * float("nan") if nan_calls(calls["n"]) else loss

        with mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}),                 mock.patch.object(outcome_training.OutcomeModel, "loss", maybe_nan):
            return train_outcome(load_outcome_config(path), root, model=tiny_model(), tokenizer=CharTokenizer())

    def test_isolated_nan_loss_is_skipped_and_counted(self) -> None:
        import math

        with TemporaryDirectory() as directory:
            result = self._run(Path(directory), lambda n: n == 2)
        self.assertEqual(result["nonfinite_events"], 1)
        self.assertTrue(all(math.isfinite(h["train_loss"]) for h in result["history"]))

    def test_persistent_nan_stops_the_run(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError) as raised:
                self._run(Path(directory), lambda n: n >= 3)
        self.assertIn("diverged", str(raised.exception))


class ClassWeightTests(unittest.TestCase):
    def test_inverse_frequency_weights_average_to_one(self) -> None:
        from legal_graph.outcome_training import class_weights

        labels = ["allowed"] * 6 + ["dismissed"] * 3 + ["partly_allowed"]
        weights = class_weights(labels, "inverse_frequency")
        self.assertAlmostEqual(sum(weights[label] for label in labels) / len(labels), 1.0)
        self.assertGreater(weights["partly_allowed"], weights["dismissed"])
        self.assertGreater(weights["dismissed"], weights["allowed"])
        self.assertAlmostEqual(weights["partly_allowed"] / weights["allowed"], 6.0)

    def test_none_means_unweighted(self) -> None:
        from legal_graph.outcome_training import class_weights

        self.assertEqual(class_weights(["allowed", "dismissed"], "none"), {})

    def test_unknown_scheme_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            data = json.loads(write_workspace(Path(directory), "text").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            config_from_dict({**data, "class_weighting": "focal"})
