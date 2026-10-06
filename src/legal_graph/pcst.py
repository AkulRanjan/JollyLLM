"""Greedy prize-collecting Steiner tree selection (pcst_greedy_v1).

An approximation, not an exact PCST solver: seeds are joined to the growing tree along
shortest paths while their prize exceeds the path cost, profitable neighbours are
added best-first, and unprofitable non-seed leaves are pruned.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Mapping

from .models import EdgeRecord, NodeRecord

STRATEGY = "pcst_greedy_v1"


def select_pcst(
    seeds: tuple[str, ...],
    edges: tuple[EdgeRecord, ...],
    nodes: Mapping[str, NodeRecord],
    *,
    budget: int,
    max_hops: int,
    edge_cost: float = 0.1,
    neighbour_prize: float = 0.25,
) -> list[str]:
    if not seeds or budget < 1:
        return []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.source_id in nodes and edge.target_id in nodes:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
    seed_set = set(seeds)
    prize: dict[str, float] = defaultdict(float, {seed: 1.0 for seed in seeds})
    for node_id in list(adjacency):
        if node_id not in seed_set:
            prize[node_id] = neighbour_prize * len(adjacency[node_id] & seed_set)

    tree = [seeds[0]]
    in_tree = {seeds[0]}
    while len(tree) < budget:
        best: tuple[float, str, list[str]] | None = None
        for seed in sorted(seed_set - in_tree):
            path = _shortest_path(in_tree, seed, adjacency, 2 * max_hops)
            if path is None:
                continue
            new = [node for node in path if node not in in_tree]
            gain = sum(prize[node] for node in new) - edge_cost * len(new)
            if gain > 0 and len(tree) + len(new) <= budget and (best is None or gain > best[0]):
                best = (gain, seed, new)
        if best is None:
            break
        for node in best[2]:
            tree.append(node)
            in_tree.add(node)

    frontier = sorted(
        {neighbour for node in tree for neighbour in adjacency.get(node, ())} - in_tree,
        key=lambda node: (-(prize[node] - edge_cost), node),
    )
    for node in frontier:
        if len(tree) >= budget or prize[node] - edge_cost <= 0:
            break
        tree.append(node)
        in_tree.add(node)

    pruned = True
    while pruned:
        pruned = False
        for node in list(tree):
            degree = len(adjacency.get(node, set()) & in_tree)
            if node not in seed_set and degree <= 1 and prize[node] < edge_cost:
                tree.remove(node)
                in_tree.discard(node)
                pruned = True
    return tree


def _shortest_path(sources: set[str], target: str, adjacency: Mapping[str, set[str]], limit: int) -> list[str] | None:
    parents: dict[str, str | None] = {source: None for source in sorted(sources)}
    queue = deque((source, 0) for source in sorted(sources))
    while queue:
        node, depth = queue.popleft()
        if node == target:
            path: list[str] = []
            current: str | None = node
            while current is not None and current not in sources:
                path.append(current)
                current = parents[current]
            return list(reversed(path))
        if depth >= limit:
            continue
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour not in parents:
                parents[neighbour] = node
                queue.append((neighbour, depth + 1))
    return None
