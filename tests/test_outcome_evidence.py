from __future__ import annotations

from datetime import date
import importlib.util
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from legal_graph.outcome_examples import OutcomeExample
from legal_graph.outcome_prompt import prompt_token_ids
from legal_graph.outcome_training import config_from_dict


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def write_provisions(path: Path) -> Path:
    rows = [
        {"node_id": "provision:india:ipc:302:sha256-a", "act_id": "ipc", "section": "302",
         "text": "302. Punishment for murder. Whoever commits murder shall be punished with death."},
        {"node_id": "provision:india:bns:103:sha256-b", "act_id": "bns", "section": "103",
         "text": "103. Punishment for murder. Whoever commits murder shall be punished with death or imprisonment."},
        {"node_id": "provision:india:crpc:482:sha256-c", "act_id": "crpc", "section": "482",
         "text": "482. Saving of inherent powers of High Court to quash proceedings."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


class Bm25EvidenceTests(unittest.TestCase):
    def test_returns_relevant_provisions_in_force_on_the_decision_date(self) -> None:
        from legal_graph.outcome_evidence import Bm25Evidence

        with TemporaryDirectory() as directory:
            evidence = Bm25Evidence(write_provisions(Path(directory) / "p.jsonl"), top_k=2)
        in_2016 = evidence.provisions("the accused committed murder and was punished", date(2016, 5, 1))
        in_2024 = evidence.provisions("the accused committed murder and was punished", date(2024, 9, 1))
        self.assertIn("provision:india:ipc:302:sha256-a", in_2016)
        self.assertNotIn("provision:india:bns:103:sha256-b", in_2016)  # BNS is not in force in 2016
        self.assertIn("provision:india:bns:103:sha256-b", in_2024)

    def test_evidence_text_and_empty_query(self) -> None:
        from legal_graph.outcome_evidence import EVIDENCE_HEADER, Bm25Evidence

        with TemporaryDirectory() as directory:
            evidence = Bm25Evidence(write_provisions(Path(directory) / "p.jsonl"), top_k=1)
        text = evidence(OutcomeExample("c", "d", "validation", "petition to quash proceedings inherent powers", "allowed", date(2016, 1, 1)))
        self.assertTrue(text.startswith(EVIDENCE_HEADER))
        self.assertIn("CRPC: 482.", text)
        self.assertEqual(evidence(OutcomeExample("c", "d", "validation", "zzz qqq", "allowed", None)), "")


class WordTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]}


class EvidencePromptTests(unittest.TestCase):
    def test_no_evidence_leaves_the_prompt_unchanged(self) -> None:
        tokenizer = WordTokenizer()
        plain = prompt_token_ids(tokenizer, "word " * 50, 80)
        self.assertEqual(prompt_token_ids(tokenizer, "word " * 50, 80, evidence="", max_evidence_tokens=10), plain)

    def test_evidence_is_capped_and_shares_the_budget(self) -> None:
        tokenizer = WordTokenizer()
        ids = prompt_token_ids(tokenizer, "word " * 500, 80, evidence="ev " * 100, max_evidence_tokens=10)
        self.assertEqual(len(ids), 80)
        self.assertEqual(ids[:10], [tokenizer.vocab["ev"]] * 10)


class FrozenConfigTests(unittest.TestCase):
    def _base(self) -> dict:
        from tests.test_outcome_training import write_workspace

        with TemporaryDirectory() as directory:
            return json.loads(write_workspace(Path(directory), "text").read_text(encoding="utf-8"))

    def test_bm25_prompt_requires_evidence_settings(self) -> None:
        data = self._base()
        with self.assertRaises(ValueError):
            config_from_dict({**data, "variant": "bm25_prompt"})
        config = config_from_dict({**data, "variant": "bm25_prompt", "evidence": {"top_k": 3, "max_tokens": 256}})
        self.assertEqual((config.bm25_top_k, config.max_evidence_tokens), (3, 256))

    def test_other_variants_reject_evidence(self) -> None:
        with self.assertRaises(ValueError):
            config_from_dict({**self._base(), "evidence": {"top_k": 3, "max_tokens": 256}})

    def test_zero_shot_loads(self) -> None:
        self.assertEqual(config_from_dict({**self._base(), "variant": "zero_shot"}).variant, "zero_shot")


@unittest.skipUnless(_has("torch") and _has("transformers") and _has("peft"), "ML stack not installed")
class FrozenRunTests(unittest.TestCase):
    def test_bm25_prompt_run_scores_validation_then_test_once(self) -> None:
        from legal_graph.outcome_eval import evaluate_frozen, freeze_run
        from legal_graph.outcome_training import load_outcome_config, train_outcome
        from tests.test_outcome_training import CharTokenizer, tiny_model, write_workspace

        with TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"LEGAL_GRAPH_SKIP_PARAM_CAP": "1"}):
            root = Path(directory)
            path = write_workspace(root, "text")
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(variant="bm25_prompt", provisions_path=str(write_provisions(root / "p.jsonl")),
                        evidence={"top_k": 2, "max_tokens": 120}, max_prompt_tokens=600)
            path.write_text(json.dumps(data), encoding="utf-8")
            result = train_outcome(load_outcome_config(path), root, model=tiny_model(), tokenizer=CharTokenizer())
            run = root / "run"
            self.assertTrue(result["frozen_base_model"])
            self.assertEqual(result["train_examples"], 0)
            self.assertIsNotNone(result["evidence_corpus_sha256"])
            self.assertFalse((run / "adapter").exists())
            freeze_run(run)
            evaluated = evaluate_frozen(run, root, base_model=tiny_model(), tokenizer=CharTokenizer())
            self.assertEqual(evaluated["status"], "measured_on_test")
            with self.assertRaises(ValueError):
                evaluate_frozen(run, root, ablate_graph=True, base_model=tiny_model(), tokenizer=CharTokenizer())


class UndatedEvidenceTests(unittest.TestCase):
    def test_undated_case_uses_year_fallback(self) -> None:
        from legal_graph.outcome_evidence import Bm25Evidence

        with TemporaryDirectory() as directory:
            evidence = Bm25Evidence(write_provisions(Path(directory) / "p.jsonl"), top_k=3)
        text = evidence(OutcomeExample("c", "d", "validation", "murder punished with death", "allowed", None, 2016))
        self.assertNotIn("BNS:", text)
        self.assertIn("IPC: 302.", text)
