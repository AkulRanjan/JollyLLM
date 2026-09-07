"""Deterministic split-safe subgraph retrieval for the initial vertical slice.

This module deliberately does not claim to be a PCST solver. It implements a
prize-guided connected expansion that exercises the same versioned request and
bundle contracts. A future PCST implementation can replace the selector behind
the same public function once its solver dependency is approved.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
from hashlib import sha256
from typing import Iterable

from .models import EdgeRecord, GraphSnapshot, NodeRecord, RetrievalConfig, RetrievalRequest, SubgraphBundle


def retrieve_subgraph(
    graph: GraphSnapshot,
    request: RetrievalRequest,
    config: RetrievalConfig,
) -> SubgraphBundle:
    """Select a deterministic connected, split-safe subgraph from explicit seeds."""

    if config.node_budget < 1:
        raise ValueError("node_budget must be at least 1")
    if config.expansion_hops < 0:
        raise ValueError("expansion_hops cannot be negative")

    nodes = graph.nodes_by_id
    eligible_edges = tuple(edge for edge in graph.edges if _edge_is_eligible(edge, nodes, request))
    eligible_nodes = {node_id for node_id, node in nodes.items() if _node_is_eligible(node, request.as_of_date)}
    seeds = sorted(
        (node_id for node_id in request.seed_node_ids if node_id in eligible_nodes),
        key=lambda node_id: _node_sort_key(node_id, request),
    )
    cache_key = _cache_key(graph, request, config)

    if not seeds:
        return SubgraphBundle(
            graph_version=graph.version,
            query_id=request.query_id,
            retrieval_strategy=config.strategy,
            selected_node_ids=(),
            selected_edge_ids=(),
            cache_key=cache_key,
            status="no_result",
            diagnostics={"reason": "no_eligible_seed"},
        )

    adjacency = _build_adjacency(eligible_edges)
    root = seeds[0]
    selected = _connected_prize_guided_expansion(root, adjacency, request, config)
    selected_edges = tuple(
        edge.edge_id
        for edge in eligible_edges
        if edge.source_id in selected and edge.target_id in selected
    )

    return SubgraphBundle(
        graph_version=graph.version,
        query_id=request.query_id,
        retrieval_strategy=config.strategy,
        selected_node_ids=tuple(selected),
        selected_edge_ids=selected_edges,
        cache_key=cache_key,
        status="ok",
        diagnostics={
            "root_seed": root,
            "eligible_edge_count": str(len(eligible_edges)),
            "selected_node_count": str(len(selected)),
        },
    )


def _node_is_eligible(node: NodeRecord, as_of_date: date | None) -> bool:
    if as_of_date is None:
        return True
    if node.available_from and node.available_from > as_of_date:
        return False
    if node.available_to and node.available_to < as_of_date:
        return False
    if node.decision_date and node.decision_date > as_of_date:
        return False
    return True


def _edge_is_eligible(edge: EdgeRecord, nodes: dict[str, NodeRecord], request: RetrievalRequest) -> bool:
    if edge.origin_split not in request.allowed_origin_splits:
        return False
    if request.allowed_relation_types is not None and edge.relation_type not in request.allowed_relation_types:
        return False
    if edge.source_id not in nodes or edge.target_id not in nodes:
        return False
    if not _node_is_eligible(nodes[edge.source_id], request.as_of_date):
        return False
    if not _node_is_eligible(nodes[edge.target_id], request.as_of_date):
        return False
    if request.as_of_date is not None:
        if edge.available_from and edge.available_from > request.as_of_date:
            return False
        if edge.available_to and edge.available_to < request.as_of_date:
            return False
        if edge.source_decision_date and edge.source_decision_date > request.as_of_date:
            return False
    return True


def _build_adjacency(edges: Iterable[EdgeRecord]) -> dict[str, list[tuple[str, str]]]:
    adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        # Connectivity selection treats allowed directed edges as traversable in either direction.
        # The original direction remains in selected_edge_ids for the GNN input.
        adjacency[edge.source_id].append((edge.target_id, edge.edge_id))
        adjacency[edge.target_id].append((edge.source_id, edge.edge_id))
    return dict(adjacency)


def _connected_prize_guided_expansion(
    root: str,
    adjacency: dict[str, list[tuple[str, str]]],
    request: RetrievalRequest,
    config: RetrievalConfig,
) -> list[str]:
    selected = [root]
    visited = {root}
    frontier: deque[tuple[str, int]] = deque([(root, 0)])

    while frontier and len(selected) < config.node_budget:
        parent, depth = frontier.popleft()
        if depth >= config.expansion_hops:
            continue
        neighbours = sorted(adjacency.get(parent, ()), key=lambda item: _node_sort_key(item[0], request))
        for neighbour, _edge_id in neighbours:
            if neighbour in visited:
                continue
            visited.add(neighbour)
            selected.append(neighbour)
            frontier.append((neighbour, depth + 1))
            if len(selected) >= config.node_budget:
                break
    return selected


def _node_sort_key(node_id: str, request: RetrievalRequest) -> tuple[float, str]:
    # Negative prize sorts high-relevance nodes first and node_id makes ties reproducible.
    return (-request.node_prizes.get(node_id, 0.0), node_id)


def _cache_key(graph: GraphSnapshot, request: RetrievalRequest, config: RetrievalConfig) -> str:
    fields = [
        graph.version,
        request.query_id,
        ",".join(sorted(request.seed_node_ids)),
        ",".join(f"{key}:{value:.12g}" for key, value in sorted(request.node_prizes.items())),
        request.as_of_date.isoformat() if request.as_of_date else "none",
        ",".join(sorted(split.value for split in request.allowed_origin_splits)),
        ",".join(sorted(relation.value for relation in request.allowed_relation_types or ())),
        config.strategy,
        str(config.node_budget),
        str(config.expansion_hops),
        str(config.require_connected),
    ]
    return "sha256:" + sha256("|".join(fields).encode("utf-8")).hexdigest()
