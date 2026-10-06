"""Relation-aware graph encoder that turns a query subgraph into K soft tokens."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .query_graph import QuerySubgraph


class RGCNLayer(nn.Module):
    """h_v' = W_0 h_v + sum_r sum_{u in N_r(v)} (1 / c_{v,r}) W_r h_u."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int) -> None:
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.relation_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.relation_weight.view(num_relations * in_dim, out_dim))

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        out = self.self_loop(h)
        if edge_index.numel() == 0:
            return out
        source, target = edge_index[0], edge_index[1]
        messages = torch.bmm(h[source].unsqueeze(1), self.relation_weight[edge_type]).squeeze(1)
        num_relations = self.relation_weight.shape[0]
        key = target * num_relations + edge_type
        counts = torch.zeros(h.shape[0] * num_relations, device=h.device, dtype=h.dtype)
        counts = counts.index_add(0, key, torch.ones_like(key, dtype=h.dtype))
        messages = messages / counts[key].unsqueeze(1)
        return out.index_add(0, target, messages)


class GraphPrefixEncoder(nn.Module):
    """R-GCN over the query subgraph, query-conditioned pooling, and an MLP projector."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        model_dim: int,
        num_tokens: int,
        num_relations: int,
        typed: bool,
        num_layers: int = 2,
        output_rms: float = 0.02,
    ) -> None:
        super().__init__()
        self.typed = typed
        relations = num_relations if typed else 1
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(RGCNLayer(hidden_dim, hidden_dim, relations) for _ in range(num_layers))
        self.pool_queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.query_proj = nn.Linear(model_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        # Two-layer MLP projector; the inner width stays at hidden_dim so the encoder adds
        # O(hidden * model_dim) parameters rather than O(model_dim^2) to the trainable budget.
        self.projector = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, model_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.scale = nn.Parameter(torch.tensor(float(output_rms)))
        self.null_tokens = nn.Parameter(torch.randn(num_tokens, model_dim) * output_rms)

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_type: Tensor, query: Tensor) -> Tensor:
        if node_features.shape[0] == 0:
            return self.null_tokens
        if not self.typed:
            edge_type = torch.zeros_like(edge_type)
        h = torch.relu(self.input_proj(node_features))
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index, edge_type))
        queries = self.pool_queries + self.query_proj(query.to(h.dtype)).unsqueeze(0)
        scores = queries @ self.key_proj(h).T / math.sqrt(h.shape[-1])
        pooled = torch.softmax(scores, dim=-1) @ h
        return self.norm(self.projector(pooled)) * self.scale


def subgraph_tensors(sub: QuerySubgraph, feature_lookup: Mapping[str, Tensor], device) -> tuple[Tensor, Tensor, Tensor]:
    """Node feature matrix, edge index [2, E], and edge types [E] for one query subgraph."""
    if not sub.node_ids:
        width = next(iter(feature_lookup.values())).shape[-1] if feature_lookup else 1
        return (
            torch.zeros(0, width, device=device),
            torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )
    x = torch.stack([feature_lookup[node_id] for node_id in sub.node_ids]).to(device)
    if sub.edges:
        index = torch.tensor([[s for s, _, _ in sub.edges], [t for _, t, _ in sub.edges]], dtype=torch.long, device=device)
        types = torch.tensor([r for _, _, r in sub.edges], dtype=torch.long, device=device)
    else:
        index = torch.zeros(2, 0, dtype=torch.long, device=device)
        types = torch.zeros(0, dtype=torch.long, device=device)
    return x, index, types


@torch.no_grad()
def embed_node_texts(texts: Mapping[str, str], tokenizer, embedding: nn.Module, max_tokens: int, device) -> dict[str, Tensor]:
    """Mean of the frozen LM input embeddings over each node's short descriptive text."""
    out: dict[str, Tensor] = {}
    for node_id, text in texts.items():
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
        ids = list(ids)[:max_tokens] or [tokenizer.eos_token_id]
        vectors = embedding(torch.tensor(ids, device=device))
        out[node_id] = vectors.float().mean(dim=0).cpu()
    return out
