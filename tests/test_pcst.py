from __future__ import annotations

import unittest

from legal_graph.models import EdgeRecord, NodeRecord, NodeType, RelationType, Split
from legal_graph.pcst import select_pcst


def _node(node_id: str) -> NodeRecord:
    return NodeRecord(node_id, NodeType.PROVISION, node_id, ("d",))


def _edge(index: int, source: str, target: str) -> EdgeRecord:
    return EdgeRecord(f"e{index}", RelationType.REFERS_TO_PROVISION, source, target, ("x",), Split.TRAIN)


NODES = {key: _node(key) for key in ("s1", "s2", "hub", "c1", "c2", "c3", "far")}
EDGES = (_edge(1, "c1", "s1"), _edge(2, "c1", "s2"), _edge(3, "hub", "s1"), _edge(4, "c2", "s1"), _edge(5, "c3", "far"))


class PcstTests(unittest.TestCase):
    def test_connects_seeds_through_shared_case(self) -> None:
        tree = select_pcst(("s1", "s2"), EDGES, NODES, budget=10, max_hops=2)
        self.assertEqual(tree[0], "s1")
        self.assertTrue({"s1", "s2", "c1"} <= set(tree))

    def test_excludes_unconnected_nodes(self) -> None:
        tree = select_pcst(("s1", "s2"), EDGES, NODES, budget=10, max_hops=2)
        self.assertNotIn("far", tree)
        self.assertNotIn("c3", tree)

    def test_respects_budget_and_is_deterministic(self) -> None:
        first = select_pcst(("s1", "s2"), EDGES, NODES, budget=2, max_hops=2)
        second = select_pcst(("s1", "s2"), EDGES, NODES, budget=2, max_hops=2)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 2)

    def test_single_isolated_seed(self) -> None:
        self.assertEqual(select_pcst(("hub",), (), NODES, budget=5, max_hops=2), ["hub"])

    def test_unprofitable_connector_is_pruned(self) -> None:
        tree = select_pcst(("s1",), EDGES, NODES, budget=10, max_hops=2, neighbour_prize=0.05)
        self.assertEqual(tree, ["s1"])
