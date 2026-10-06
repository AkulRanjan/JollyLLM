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


class IndianReferenceExtractionV2Tests(unittest.TestCase):
    def _hints(self, text: str) -> list[str]:
        from legal_graph.extraction import extract_indian_references_v2

        return [candidate.canonical_hint for candidate in extract_indian_references_v2(text, "doc:v2")]

    def test_dotted_abbreviations_and_section_lists(self) -> None:
        self.assertEqual(
            self._hints("convicted under Sections 302 and 34 of the I.P.C. and Section 313 Cr.P.C."),
            ["provision:india:ipc:302:", "provision:india:ipc:34:", "provision:india:crpc:313:"],
        )

    def test_subsections_reduce_to_the_base_provision_and_suffix_is_uppercase(self) -> None:
        self.assertEqual(self._hints("Section 482(1)(a) of the Code of Criminal Procedure, 1973; s. 498a IPC"),
                         ["provision:india:crpc:482:", "provision:india:ipc:498A:"])

    def test_evidence_act_and_constitution_articles(self) -> None:
        self.assertEqual(
            self._hints("Section 27 of the Indian Evidence Act. A petition under Articles 32 and 21 of the Constitution."),
            ["provision:india:iea:27:", "provision:india:constitution:32:", "provision:india:constitution:21:"],
        )

    def test_bare_article_defaults_to_constitution_but_not_other_instruments(self) -> None:
        self.assertEqual(self._hints("Special leave under Article 136."), ["provision:india:constitution:136:"])
        self.assertEqual(self._hints("Article 5 of the Vienna Convention applies."), [])

    def test_case_citations_are_kept(self) -> None:
        self.assertIn("case:india:reporter:scc:2020:4:123", self._hints("See (2020) 4 SCC 123."))

    def test_bnss_is_not_read_as_bns(self) -> None:
        self.assertEqual(self._hints("Section 35 BNSS"), ["provision:india:bnss:35:"])

    def test_v1_is_unchanged(self) -> None:
        self.assertEqual([c.canonical_hint for c in extract_indian_references("Section 302 of the I.P.C.", "d")], [])
