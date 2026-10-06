"""Prompt template, label verbalisers, and candidate scoring for the outcome task."""

from __future__ import annotations

from typing import Mapping, Sequence

from .outcomes import LABELS

INSTRUCTION = (
    "Read the opening of an Indian Supreme Court judgment. Phrases that state the final "
    "disposition have been replaced by outcome_phrase. Answer with the final outcome: "
    "allowed, dismissed, disposed, or partly allowed.\n\nJudgment:\n"
)
ANSWER_PREFIX = "\n\nOutcome:"
VERBALIZERS = {label: " " + label.replace("_", " ") for label in LABELS}


def prompt_token_ids(tokenizer, view_text: str, max_prompt_tokens: int) -> list[int]:
    """Instruction + judgment view truncated to fit + answer prefix, which is never truncated."""
    head = tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
    tail = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    body = tokenizer(view_text, add_special_tokens=False)["input_ids"]
    room = max_prompt_tokens - len(head) - len(tail)
    if room <= 0:
        raise ValueError("max_prompt_tokens leaves no room for the judgment view")
    return list(head) + list(body[:room]) + list(tail)


def label_token_ids(tokenizer) -> dict[str, tuple[int, ...]]:
    ids = {label: tuple(tokenizer(text, add_special_tokens=False)["input_ids"]) for label, text in VERBALIZERS.items()}
    if any(not value for value in ids.values()) or len(set(ids.values())) != len(ids):
        raise ValueError("label verbalisers must tokenise to non-empty, distinct sequences")
    return ids


def continuation_log_prob(log_probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """Total log-probability of a candidate continuation (sum over all of its tokens)."""
    if len(log_probs) != len(targets):
        raise ValueError("one log-probability row is required per target token")
    return float(sum(row[target] for row, target in zip(log_probs, targets)))


def pick_label(scores: Mapping[str, float]) -> str:
    """Highest-scoring label; ties go to the earlier label in the schema order."""
    return max(LABELS, key=lambda label: (scores[label], -LABELS.index(label)))
