from __future__ import annotations

from datetime import date
import unittest

from legal_graph.models import EdgeRecord, GraphSnapshot, NodeRecord, NodeType, RelationType, RetrievalConfig, Split
from legal_graph.query_graph import NUM_RELATION_TYPES, RELATIONS, build_query_subgraph, view_seed_node_ids


def _graph() -> GraphSnapshot:
    crpc = NodeRecord("act:india:crpc", NodeType.ACT, "CrPC", ("d",), available_from=date(1974, 4, 1))
    bns = NodeRecord("act:india:bns", NodeType.ACT, "BNS", ("d",), available_from=date(2023, 12, 25))
    s482 = NodeRecord("provision:india:crpc:482:sha256-x", NodeType.PROVISION, "CrPC section 482", ("d",), available_from=date(1974, 4, 1))
    s103 = NodeRecord("provision:india:bns:103:sha256-y", NodeType.PROVISION, "BNS section 103", ("d",), available_from=date(2023, 12, 25))
    train_case = NodeRecord("case:train", NodeType.CASE, "A v B", ("d",), decision_date=date(2016, 1, 5))
    query_case = NodeRecord("case:query", NodeType.CASE, "C v D", ("d",), decision_date=date(2016, 6, 1))
    test_case = NodeRecord("case:test", NodeType.CASE, "E v F", ("d",), decision_date=date(2016, 2, 1))
    later_case = NodeRecord("case:later", NodeType.CASE, "G v H", ("d",), decision_date=date(2016, 9, 1))
    edges = (
        EdgeRecord("e1", RelationType.CONTAINS, "act:india:crpc", s482.node_id, ("x",), Split.TRAIN),
        EdgeRecord("e2", RelationType.CONTAINS, "act:india:bns", s103.node_id, ("x",), Split.TRAIN),
        EdgeRecord("e3", RelationType.REFERS_TO_PROVISION, "case:train", s482.node_id, ("x",), Split.TRAIN, source_decision_date=date(2016, 1, 5)),
        EdgeRecord("e4", RelationType.REFERS_TO_PROVISION, "case:query", s482.node_id, ("x",), Split.VALIDATION, source_decision_date=date(2016, 6, 1)),
        EdgeRecord("e5", RelationType.REFERS_TO_PROVISION, "case:test", s482.node_id, ("x",), Split.TEST, source_decision_date=date(2016, 2, 1)),
        EdgeRecord("e6", RelationType.REFERS_TO_PROVISION, "case:later", s482.node_id, ("x",), Split.TRAIN, source_decision_date=date(2016, 9, 1)),
    )
    return GraphSnapshot("g", "1.0.0", (crpc, bns, s482, s103, train_case, query_case, test_case, later_case), edges)


VIEW = "The petition under Section 482 of the CrPC and s. 103 BNS was filed."


class SeedTests(unittest.TestCase):
    def test_seeds_resolve_from_view_text(self) -> None:
        seeds = view_seed_node_ids(VIEW, "doc:query", _graph())
        self.assertEqual(seeds, ("provision:india:crpc:482:sha256-x", "provision:india:bns:103:sha256-y"))

    def test_view_without_citations_has_no_seeds(self) -> None:
        self.assertEqual(view_seed_node_ids("No statute is named here.", "doc:q", _graph()), ())


class SubgraphTests(unittest.TestCase):
    def _build(self, view: str = VIEW, selector: str = "expansion"):
        return build_query_subgraph(
            _graph(),
            query_id="q",
            document_id="doc:query",
            view_text=view,
            as_of_date=date(2016, 6, 1),
            exclude_node_ids=frozenset({"case:query"}),
            config=RetrievalConfig(node_budget=10, expansion_hops=2),
            selector=selector,
            random_seed=3,
        )

    def test_only_train_split_and_past_nodes_are_reachable(self) -> None:
        sub = self._build()
        self.assertEqual(sub.status, "ok")
        self.assertIn("case:train", sub.node_ids)
        self.assertNotIn("case:query", sub.node_ids)
        self.assertNotIn("case:test", sub.node_ids)
        self.assertNotIn("case:later", sub.node_ids)
        self.assertNotIn("provision:india:bns:103:sha256-y", sub.node_ids)
        self.assertEqual(sub.seed_node_ids, ("provision:india:crpc:482:sha256-x",))

    def test_edges_are_index_triples_with_inverse_relations(self) -> None:
        sub = self._build()
        forward = RELATIONS.index(RelationType.REFERS_TO_PROVISION)
        types = {relation for _, _, relation in sub.edges}
        self.assertIn(forward, types)
        self.assertIn(forward + len(RELATIONS), types)
        self.assertTrue(all(0 <= relation < NUM_RELATION_TYPES for relation in types))
        self.assertTrue(all(0 <= s < len(sub.node_ids) and 0 <= t < len(sub.node_ids) for s, t, _ in sub.edges))

    def test_no_citation_gives_empty_subgraph(self) -> None:
        sub = self._build(view="Nothing to resolve.")
        self.assertEqual((sub.status, sub.node_ids, sub.edges), ("empty", (), ()))

    def test_random_selector_matches_size_and_is_seeded(self) -> None:
        first, second = self._build(selector="random"), self._build(selector="random")
        self.assertEqual(first.node_ids, second.node_ids)
        self.assertEqual(len(first.node_ids), len(self._build().node_ids))
        self.assertNotIn("case:test", first.node_ids)
        self.assertNotIn("case:later", first.node_ids)

    def test_pcst_selector_respects_eligibility(self) -> None:
        sub = self._build(selector="pcst")
        self.assertEqual(sub.status, "ok")
        self.assertNotIn("case:test", sub.node_ids)
        self.assertNotIn("case:later", sub.node_ids)
        self.assertEqual(sub.node_ids[0], "provision:india:crpc:482:sha256-x")

    def test_unknown_selector_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._build(selector="magic")
