from __future__ import annotations

import unittest

from legal_graph.outcome_prompt import ANSWER_PREFIX, continuation_log_prob, label_token_ids, pick_label, prompt_token_ids


class WordTokenizer:
    """Whitespace tokenizer for deterministic tests."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]}


class PromptTests(unittest.TestCase):
    def test_prompt_truncates_view_but_keeps_answer_prefix(self) -> None:
        tokenizer = WordTokenizer()
        ids = prompt_token_ids(tokenizer, "word " * 500, max_prompt_tokens=60)
        self.assertEqual(len(ids), 60)
        tail = tokenizer(ANSWER_PREFIX)["input_ids"]
        self.assertEqual(ids[-len(tail):], tail)

    def test_prompt_without_room_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prompt_token_ids(WordTokenizer(), "text", max_prompt_tokens=5)

    def test_label_ids_are_distinct_and_multi_token_for_partly(self) -> None:
        ids = label_token_ids(WordTokenizer())
        self.assertEqual(len(set(ids.values())), 4)
        self.assertEqual(len(ids["partly_allowed"]), 2)

    def test_continuation_sums_all_tokens(self) -> None:
        self.assertAlmostEqual(continuation_log_prob([[-1.0, -2.0], [-3.0, -0.5]], [1, 1]), -2.5)

    def test_pick_label_breaks_ties_in_schema_order(self) -> None:
        self.assertEqual(pick_label({"dismissed": -1.0, "allowed": -1.0, "disposed": -3.0, "partly_allowed": -9.0}), "allowed")
        self.assertEqual(pick_label({"dismissed": -0.5, "allowed": -1.0, "disposed": -3.0, "partly_allowed": -9.0}), "dismissed")
