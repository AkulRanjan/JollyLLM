from __future__ import annotations

from datetime import date
import unittest

from legal_graph.builder import build_reference_edges
from legal_graph.extraction import extract_indian_references, resolve_candidates
from legal_graph.models import RelationType, Split


class GraphBuilderTests(unittest.TestCase):
    def test_builds_grouped_edges_and_retains_unresolved_records(self) -> None:
        candidates = extract_indian_references(
            "Section 302 IPC; s. 302 IPC; (2020) 4 SCC 123; Section 103 BNS.",
            "doc:fixture:builder",
        )
        resolutions = resolve_candidates(
            candidates,
            (
                "provision:india:ipc:302:v1",
                "case:india:reporter:scc:2020:4:123",
            ),
        )

        result = build_reference_edges(
            "case:india:sc:builder-2022",
            resolutions,
            origin_split=Split.TRAIN,
            source_decision_date=date(2022, 1, 1),
        )

        self.assertEqual(len(result.edges), 2)
        provision_edge = next(edge for edge in result.edges if edge.relation_type is RelationType.REFERS_TO_PROVISION)
        case_edge = next(edge for edge in result.edges if edge.relation_type is RelationType.CITES)
        self.assertEqual(provision_edge.target_id, "provision:india:ipc:302:v1")
        self.assertEqual(len(provision_edge.evidence_extraction_ids), 2)
        self.assertEqual(case_edge.target_id, "case:india:reporter:scc:2020:4:123")
        self.assertEqual(len(result.unresolved), 1)
        self.assertEqual(result.unresolved[0].unresolved_reason, "no_canonical_target")

    def test_identical_resolutions_produce_identical_edge_ids(self) -> None:
        candidates = extract_indian_references("(2020) 4 SCC 123", "doc:fixture:deterministic")
        resolutions = resolve_candidates(candidates, ("case:india:reporter:scc:2020:4:123",))

        first = build_reference_edges("case:india:sc:source", resolutions, origin_split=Split.TRAIN)
        second = build_reference_edges("case:india:sc:source", resolutions, origin_split=Split.TRAIN)

        self.assertEqual(first.edges[0].edge_id, second.edges[0].edge_id)
