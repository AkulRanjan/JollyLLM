"""Framework-independent validation of graph soft-token prefix assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


Embedding = Sequence[float]


@dataclass(frozen=True, slots=True)
class PrefixAssembly:
    """Validated model-ready sequence representation before tensor conversion."""

    embeddings: tuple[tuple[float, ...], ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    graph_token_count: int
    instruction_token_count: int
    document_token_count: int


def assemble_prefix(
    graph_soft_tokens: Sequence[Embedding],
    instruction_embeddings: Sequence[Embedding],
    document_embeddings: Sequence[Embedding],
    document_labels: Sequence[int],
    *,
    ignore_index: int = -100,
) -> PrefixAssembly:
    """Validate and concatenate `[graph] || [instruction] || [document]` embeddings.

    Graph and instruction positions receive ignored labels because they are input
    context. The caller supplies labels only for document/task positions.
    """

    if not graph_soft_tokens:
        raise ValueError("at least one graph soft token is required; use an explicit null token for an empty graph")
    if len(document_embeddings) != len(document_labels):
        raise ValueError("document_embeddings and document_labels must have equal lengths")

    all_embeddings = [*graph_soft_tokens, *instruction_embeddings, *document_embeddings]
    if not all_embeddings:
        raise ValueError("prefix assembly requires at least one embedding")
    dimension = len(all_embeddings[0])
    if dimension == 0:
        raise ValueError("embedding dimension must be positive")
    for index, embedding in enumerate(all_embeddings):
        if len(embedding) != dimension:
            raise ValueError(
                f"embedding at combined index {index} has dimension {len(embedding)}; expected {dimension}"
            )

    context_length = len(graph_soft_tokens) + len(instruction_embeddings)
    return PrefixAssembly(
        embeddings=tuple(tuple(float(value) for value in embedding) for embedding in all_embeddings),
        attention_mask=(1,) * len(all_embeddings),
        labels=(ignore_index,) * context_length + tuple(document_labels),
        graph_token_count=len(graph_soft_tokens),
        instruction_token_count=len(instruction_embeddings),
        document_token_count=len(document_embeddings),
    )
