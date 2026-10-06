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


def write_workspace(root: Path, variant: str, *, with_validation: bool = True) -> Path:
    from legal_graph.manifests import graph_to_dict
    from tests.test_outcome_examples import _write
    from tests.test_query_graph import _graph

    corpus, manifest = _write(root, (VALIDATION_ROW,) if with_validation else ())
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
            config = load_outcome_config(write_workspace(root, "text"))
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
