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


def _cocitation_graph() -> GraphSnapshot:
    def provision(number: str) -> NodeRecord:
        return NodeRecord(f"provision:india:crpc:{number}:sha256-{number}", NodeType.PROVISION, f"CrPC section {number}",
                          ("d",), available_from=date(1974, 4, 1))

    def case(name: str, decided: date) -> NodeRecord:
        return NodeRecord(f"case:{name}", NodeType.CASE, f"{name} v State", ("d",), decision_date=decided)

    p482, p438, p439, p156 = (provision(number) for number in ("482", "438", "439", "156"))
    cases = {
        "c1": (date(2015, 1, 1), Split.TRAIN, (p482, p438, p439)),
        "c2": (date(2015, 2, 1), Split.TRAIN, (p482, p438)),
        "c3": (date(2015, 3, 1), Split.TRAIN, (p439, p156)),
        "test": (date(2015, 4, 1), Split.TEST, (p482, p156)),
        "later": (date(2017, 1, 1), Split.TRAIN, (p482, p156)),
        "query": (date(2016, 6, 1), Split.VALIDATION, (p482, p156)),
    }
    nodes = [p482, p438, p439, p156] + [case(name, decided) for name, (decided, _, _) in cases.items()]
    edges = tuple(
        EdgeRecord(f"e-{name}-{target.node_id}", RelationType.REFERS_TO_PROVISION, f"case:{name}", target.node_id, ("x",),
                   split, source_decision_date=decided)
        for name, (decided, split, targets) in cases.items() for target in targets
    )
    return GraphSnapshot("g", "1.0.0", tuple(nodes), edges)


class CocitationTests(unittest.TestCase):
    P482, P438, P439, P156 = (f"provision:india:crpc:{n}:sha256-{n}" for n in ("482", "438", "439", "156"))

    def _build(self, budget: int = 10, exclude: frozenset[str] = frozenset({"case:query"})):
        return build_query_subgraph(
            _cocitation_graph(),
            query_id="q",
            document_id="doc:query",
            view_text="The petition under Section 482 of the CrPC was filed.",
            as_of_date=date(2016, 6, 1),
            exclude_node_ids=exclude,
            config=RetrievalConfig(node_budget=budget, expansion_hops=2),
            selector="cocitation",
        )

    def test_statute_nodes_only_ranked_by_eligible_co_citation(self) -> None:
        sub = self._build()
        self.assertEqual(sub.status, "ok")
        # 438 shares two training cases with the seed, 439 one; 156 is co-cited with the seed only by the
        # test, later and query cases, none of which is eligible.
        self.assertEqual(sub.node_ids, (self.P482, self.P438, self.P439))
        self.assertTrue(all(node_id.startswith("provision:") for node_id in sub.node_ids))

    def test_links_join_provisions_cited_by_one_eligible_case(self) -> None:
        sub = self._build()
        relation = RELATIONS.index(RelationType.CO_OCCURS_WITH)
        pairs = {(sub.node_ids[s], sub.node_ids[t]) for s, t, r in sub.edges if r == relation}
        self.assertEqual({r for _, _, r in sub.edges}, {relation})
        expected = {(self.P482, self.P438), (self.P482, self.P439), (self.P438, self.P439)}
        self.assertEqual(pairs, expected | {(b, a) for a, b in expected})

    def test_budget_keeps_seeds_first(self) -> None:
        self.assertEqual(self._build(budget=2).node_ids, (self.P482, self.P438))

    def test_query_case_never_contributes(self) -> None:
        # Even if the query were a training case, its own node is excluded before any link is counted.
        sub = self._build(exclude=frozenset({"case:c2"}))
        self.assertEqual(sub.node_ids, (self.P482, self.P438, self.P439))
        relation = RELATIONS.index(RelationType.CO_OCCURS_WITH)
        self.assertIn((1, 2, relation), sub.edges)  # 438-439 still joined through c1

    def test_no_statute_seed_gives_empty_subgraph(self) -> None:
        sub = build_query_subgraph(
            _cocitation_graph(), query_id="q", document_id="doc:q", view_text="Nothing to resolve.",
            as_of_date=date(2016, 6, 1), exclude_node_ids=frozenset(), config=RetrievalConfig(node_budget=10, expansion_hops=2),
            selector="cocitation",
        )
        self.assertEqual((sub.status, sub.node_ids, sub.edges), ("empty", (), ()))
