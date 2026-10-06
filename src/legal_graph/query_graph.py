"""Split- and time-safe query subgraphs for the graph-augmented outcome model.

Seeds come only from the masked input view. Retrieval sees training-split edges only,
nothing decided or in force after the query's decision date, and never the query's
own case node, whose edges were extracted from the full judgment text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from .extraction import extract_indian_references, resolve_candidates
from .models import GraphSnapshot, RelationType, RetrievalConfig, RetrievalRequest, Split
from .retrieval import _edge_is_eligible, _node_is_eligible, retrieve_subgraph

RELATIONS: tuple[RelationType, ...] = tuple(RelationType)
NUM_RELATION_TYPES = 2 * len(RELATIONS)
SELECTORS = ("expansion", "pcst", "random")
_ALLOWED = frozenset({Split.TRAIN})


@dataclass(frozen=True, slots=True)
class QuerySubgraph:
    query_id: str
    node_ids: tuple[str, ...]
    edges: tuple[tuple[int, int, int], ...]
    seed_node_ids: tuple[str, ...]
    status: str


def view_seed_node_ids(view_text: str, document_id: str, graph: GraphSnapshot) -> tuple[str, ...]:
    """Graph nodes cited in the visible text, in order of first mention."""
    candidates = extract_indian_references(view_text, document_id)
    results = resolve_candidates(candidates, (node.node_id for node in graph.nodes))
    seen: list[str] = []
    for result in results:
        if result.resolved_target_id and result.resolved_target_id not in seen:
            seen.append(result.resolved_target_id)
    return tuple(seen)


def build_query_subgraph(
    graph: GraphSnapshot,
    *,
    query_id: str,
    document_id: str,
    view_text: str,
    as_of_date: date | None,
    exclude_node_ids: frozenset[str],
    config: RetrievalConfig,
    selector: str = "expansion",
    random_seed: int | None = None,
) -> QuerySubgraph:
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector {selector!r}")
    view = _without(graph, exclude_node_ids)
    nodes = view.nodes_by_id
    seeds = tuple(
        seed
        for seed in view_seed_node_ids(view_text, document_id, view)
        if seed in nodes and _node_is_eligible(nodes[seed], as_of_date)
    )
    if not seeds:
        return QuerySubgraph(query_id, (), (), (), "empty")
    request = RetrievalRequest(
        query_id=query_id,
        seed_node_ids=seeds,
        node_prizes={seed: 1.0 for seed in seeds},
        as_of_date=as_of_date,
        allowed_origin_splits=_ALLOWED,
    )
    eligible_edges = tuple(edge for edge in view.edges if _edge_is_eligible(edge, nodes, request))
    selected = _expand(view, request, config, seeds)
    if selector == "pcst":
        from .pcst import select_pcst

        selected = select_pcst(seeds, eligible_edges, nodes, budget=config.node_budget, max_hops=config.expansion_hops)
    elif selector == "random":
        selected = _random_nodes(view, as_of_date, len(selected), query_id, random_seed if random_seed is not None else 0)
    index = {node_id: position for position, node_id in enumerate(selected)}
    triples: set[tuple[int, int, int]] = set()
    for edge in eligible_edges:
        if edge.source_id in index and edge.target_id in index:
            relation = RELATIONS.index(edge.relation_type)
            triples.add((index[edge.source_id], index[edge.target_id], relation))
            triples.add((index[edge.target_id], index[edge.source_id], relation + len(RELATIONS)))
    return QuerySubgraph(query_id, tuple(selected), tuple(sorted(triples)), seeds, "ok" if selected else "empty")


def _without(graph: GraphSnapshot, excluded: frozenset[str]) -> GraphSnapshot:
    if not excluded:
        return graph
    return GraphSnapshot(
        graph.version,
        graph.schema_version,
        tuple(node for node in graph.nodes if node.node_id not in excluded),
        tuple(edge for edge in graph.edges if edge.source_id not in excluded and edge.target_id not in excluded),
    )


def _expand(graph: GraphSnapshot, request: RetrievalRequest, config: RetrievalConfig, seeds: tuple[str, ...]) -> list[str]:
    """Union of per-seed connected expansions sharing the node budget across seeds."""
    per_seed = max(1, config.node_budget // len(seeds))
    selected: list[str] = []
    for seed in seeds:
        single = RetrievalRequest(
            query_id=request.query_id,
            seed_node_ids=(seed,),
            node_prizes=request.node_prizes,
            as_of_date=request.as_of_date,
            allowed_origin_splits=request.allowed_origin_splits,
        )
        bundle = retrieve_subgraph(graph, single, RetrievalConfig(node_budget=per_seed, expansion_hops=config.expansion_hops))
        for node_id in bundle.selected_node_ids or (seed,):
            if node_id not in selected and len(selected) < config.node_budget:
                selected.append(node_id)
    return selected


def _random_nodes(graph: GraphSnapshot, as_of_date: date | None, count: int, query_key: str, seed: int) -> list[str]:
    """Random-retrieval control: same node count, eligible nodes, no relevance signal."""
    reachable = {edge.source_id for edge in graph.edges if edge.origin_split in _ALLOWED}
    reachable |= {edge.target_id for edge in graph.edges if edge.origin_split in _ALLOWED}
    pool = sorted(
        node.node_id for node in graph.nodes if node.node_id in reachable and _node_is_eligible(node, as_of_date)
    )
    generator = random.Random(f"{seed}|{query_key}")
    return generator.sample(pool, min(count, len(pool)))
