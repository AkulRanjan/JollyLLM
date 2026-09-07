from __future__ import annotations

import unittest

from legal_graph.extraction import (
    ReferenceKind,
    extract_indian_references,
    resolution_summary,
    resolve_candidates,
)


class IndianReferenceExtractionTests(unittest.TestCase):
    def test_extracts_statute_and_reporter_references_in_text_order(self) -> None:
        text = (
            "The Court considered Section 302 of the Indian Penal Code, followed "
            "(2020) 4 SCC 123, and distinguished 2022 SCC OnLine SC 7."
        )

        candidates = extract_indian_references(text, "doc:fixture:references")

        self.assertEqual([candidate.kind for candidate in candidates], [
            ReferenceKind.PROVISION,
            ReferenceKind.CASE,
            ReferenceKind.CASE,
        ])
        self.assertEqual(candidates[0].canonical_hint, "provision:india:ipc:302:")
        self.assertEqual(candidates[1].canonical_hint, "case:india:reporter:scc:2020:4:123")
        self.assertEqual(candidates[2].canonical_hint, "case:india:reporter:scc-online-sc:2022:7")
        self.assertEqual([candidate.raw_text for candidate in candidates], [
            "Section 302 of the Indian Penal Code",
            "(2020) 4 SCC 123",
            "2022 SCC OnLine SC 7",
        ])

    def test_resolver_retains_ambiguous_and_unresolved_references(self) -> None:
        candidates = extract_indian_references(
            "s. 302 IPC; Section 103 BNS; (2020) 4 SCC 123.",
            "doc:fixture:resolve",
        )
        results = resolve_candidates(
            candidates,
            (
                "provision:india:ipc:302:v1",
                "provision:india:ipc:302:v2",
                "case:india:reporter:scc:2020:4:123",
            ),
        )

        self.assertEqual(results[0].unresolved_reason, "ambiguous_canonical_target")
        self.assertEqual(results[1].unresolved_reason, "no_canonical_target")
        self.assertEqual(results[2].resolved_target_id, "case:india:reporter:scc:2020:4:123")
        self.assertEqual(resolution_summary(results), {"resolved": 1, "unresolved": 1, "ambiguous": 1})

    def test_extraction_id_is_stable_for_same_input(self) -> None:
        first = extract_indian_references("Section 302 IPC", "doc:fixture:stable")
        second = extract_indian_references("Section 302 IPC", "doc:fixture:stable")

        self.assertEqual(first[0].extraction_id, second[0].extraction_id)
