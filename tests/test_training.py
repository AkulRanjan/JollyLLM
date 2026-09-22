from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.training import TrainingConfigurationError, _sha256_file, load_documents


class TrainingDataTests(unittest.TestCase):
    def test_loads_only_manifest_allowed_splits(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text("\n".join((
                json.dumps({"case_id": "case:train", "origin_split": "train", "text": "train"}),
                json.dumps({"case_id": "case:validation", "origin_split": "validation", "text": "validation"}),
                json.dumps({"case_id": "case:test", "origin_split": "test", "text": "test"}),
            )) + "\n", encoding="utf-8")
            manifest = root / "training.manifest.json"
            manifest.write_text(json.dumps({
                "artifact_type": "judgment_training_corpus",
                "status": "validated",
                "corpus_sha256": _sha256_file(corpus),
                "training_splits": ["train"],
                "validation_splits": ["validation"],
                "held_out_splits": ["test"],
            }), encoding="utf-8")
            train = load_documents(corpus, manifest, "train", 10, 7)
            validation = load_documents(corpus, manifest, "validation", 10, 7)
        self.assertEqual([item["case_id"] for item in train], ["case:train"])
        self.assertEqual([item["case_id"] for item in validation], ["case:validation"])

    def test_rejects_checksum_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text(json.dumps({"case_id": "case:train", "origin_split": "train", "text": "text"}) + "\n", encoding="utf-8")
            manifest = root / "training.manifest.json"
            manifest.write_text(json.dumps({
                "artifact_type": "judgment_training_corpus",
                "status": "validated",
                "corpus_sha256": "0" * 64,
                "training_splits": ["train"],
                "validation_splits": ["validation"],
                "held_out_splits": ["test"],
            }), encoding="utf-8")
            with self.assertRaises(TrainingConfigurationError):
                load_documents(corpus, manifest, "train", 1, 7)


class CollatorTests(unittest.TestCase):
    """The padding value collides with a real token whenever a tokenizer has no
    dedicated pad token and pad_token falls back to eos_token (Llama/Mistral, so
    Saul-7B). Masking must follow each block's true length, not token identity."""

    def _collate(self, blocks: list[list[int]], pad_id: int) -> dict:
        import torch

        from legal_graph.training import _collator

        return _collator(torch, pad_id)([torch.tensor(block, dtype=torch.long) for block in blocks])

    def test_real_eos_token_is_not_masked_when_pad_equals_eos(self) -> None:
        batch = self._collate([[7, 8, 2]], pad_id=2)
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertEqual(batch["labels"].tolist(), [[7, 8, 2]])

    def test_only_padding_positions_are_masked(self) -> None:
        batch = self._collate([[7, 2, 9, 2], [4, 5]], pad_id=2)
        self.assertEqual(batch["input_ids"].tolist(), [[7, 2, 9, 2], [4, 5, 2, 2]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        self.assertEqual(batch["labels"].tolist(), [[7, 2, 9, 2], [4, 5, -100, -100]])

    def test_distinct_pad_token_still_masks_padding(self) -> None:
        batch = self._collate([[7, 8, 9], [4]], pad_id=0)
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 0, 0]])
        self.assertEqual(batch["labels"].tolist(), [[7, 8, 9], [4, -100, -100]])


class BaseModelPinningTests(unittest.TestCase):
    """Adapters must name the pinned upstream repo, not the scratch directory the
    weights happened to be staged in, or they only load on the training machine."""

    def test_rewrites_staging_path_to_pinned_source(self) -> None:
        from peft import LoraConfig

        from legal_graph.training import _pin_base_model

        configs = {"default": LoraConfig(r=8, base_model_name_or_path="/kaggle/temp/llm-graph/models/saullm-7b-instruct")}
        _pin_base_model(configs, "Equall/Saul-7B-Instruct-v1", "2133ba7923533934e78f73848045299dd74f08d2")
        self.assertEqual(configs["default"].base_model_name_or_path, "Equall/Saul-7B-Instruct-v1")
        self.assertEqual(configs["default"].revision, "2133ba7923533934e78f73848045299dd74f08d2")

    def test_applies_to_every_adapter(self) -> None:
        from peft import LoraConfig

        from legal_graph.training import _pin_base_model

        configs = {name: LoraConfig(r=8) for name in ("default", "other")}
        _pin_base_model(configs, "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
        for adapter_config in configs.values():
            self.assertEqual(adapter_config.base_model_name_or_path, "Qwen/Qwen3-8B")
            self.assertEqual(adapter_config.revision, "b968826d9c46dd6066d109eabc6255188de91218")
