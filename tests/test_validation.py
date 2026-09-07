from __future__ import annotations

from dataclasses import replace
import unittest

from legal_graph.fixtures import fixture_graph
from legal_graph.models import EdgeRecord, GraphSnapshot, NodeType, RelationType, Split
from legal_graph.validation import GraphValidationError, validate_graph


class GraphValidationTests(unittest.TestCase):
    def test_fixture_graph_is_valid(self) -> None:
        report = validate_graph(fixture_graph())

        self.assertEqual(report.node_count, 7)
        self.assertEqual(report.edge_count, 8)
        self.assertEqual(report.relation_counts["cites"], 2)

    def test_illegal_relation_domain_is_rejected(self) -> None:
        graph = fixture_graph()
        invalid_edge = EdgeRecord(
            edge_id="edge:invalid:provision-cites-case",
            relation_type=RelationType.CITES,
            source_id="provision:india:ipc:302:v1",
            target_id="case:india:sc:alpha-2020",
            evidence_extraction_ids=("extract:invalid:1",),
            origin_split=Split.TRAIN,
        )
        invalid_graph = replace(graph, edges=(*graph.edges, invalid_edge))

        with self.assertRaises(GraphValidationError) as context:
            validate_graph(invalid_graph)

        self.assertIn("cannot start at 'provision'", str(context.exception))

    def test_missing_evidence_is_rejected(self) -> None:
        graph = fixture_graph()
        invalid_edge = replace(graph.edges[0], edge_id="edge:missing-evidence", evidence_extraction_ids=())
        invalid_graph = GraphSnapshot(graph.version, graph.schema_version, graph.nodes, (*graph.edges, invalid_edge))

        with self.assertRaises(GraphValidationError) as context:
            validate_graph(invalid_graph)

        self.assertIn("has no evidence_extraction_ids", str(context.exception))
