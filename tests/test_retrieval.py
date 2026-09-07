from __future__ import annotations

from datetime import date
import unittest

from legal_graph.fixtures import fixture_graph
from legal_graph.models import RetrievalConfig, RetrievalRequest, Split
from legal_graph.retrieval import retrieve_subgraph


class RetrievalTests(unittest.TestCase):
    def test_retrieval_is_temporal_split_safe_connected_and_bounded(self) -> None:
        graph = fixture_graph()
        request = RetrievalRequest(
            query_id="fixture:test:2023",
            seed_node_ids=("case:india:sc:beta-2022",),
            node_prizes={"provision:india:ipc:302:v1": 1.0},
            as_of_date=date(2023, 1, 1),
            allowed_origin_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
        )
        bundle = retrieve_subgraph(graph, request, RetrievalConfig(node_budget=4, expansion_hops=2))

        self.assertEqual(bundle.status, "ok")
        self.assertEqual(bundle.selected_node_ids[0], "case:india:sc:beta-2022")
        self.assertLessEqual(len(bundle.selected_node_ids), 4)
        self.assertIn("provision:india:ipc:302:v1", bundle.selected_node_ids)
        self.assertNotIn("provision:india:bns:103:v1", bundle.selected_node_ids)
        self.assertNotIn("edge:cites:query-beta-test", bundle.selected_edge_ids)

    def test_retrieval_returns_explicit_no_result_when_seeds_are_future(self) -> None:
        bundle = retrieve_subgraph(
            fixture_graph(),
            RetrievalRequest(
                query_id="fixture:test:no-result",
                seed_node_ids=("case:india:sc:query-2025",),
                as_of_date=date(2023, 1, 1),
            ),
            RetrievalConfig(node_budget=5, expansion_hops=2),
        )

        self.assertEqual(bundle.status, "no_result")
        self.assertEqual(bundle.diagnostics["reason"], "no_eligible_seed")

    def test_cache_key_changes_when_split_policy_changes(self) -> None:
        graph = fixture_graph()
        common = dict(
            query_id="fixture:test:cache",
            seed_node_ids=("case:india:sc:beta-2022",),
            as_of_date=date(2023, 1, 1),
        )
        train_safe = retrieve_subgraph(
            graph,
            RetrievalRequest(**common, allowed_origin_splits=frozenset({Split.TRAIN})),
            RetrievalConfig(),
        )
        test_inclusive = retrieve_subgraph(
            graph,
            RetrievalRequest(**common, allowed_origin_splits=frozenset({Split.TRAIN, Split.TEST})),
            RetrievalConfig(),
        )

        self.assertNotEqual(train_safe.cache_key, test_inclusive.cache_key)
