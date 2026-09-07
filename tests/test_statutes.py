from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.statutes import (
    Provision,
    StatuteIngestion,
    StatuteSpec,
    build_statute_graph,
    extract_provisions_from_pages,
    write_provision_corpus,
)


class StatuteExtractionTests(unittest.TestCase):
    def test_extracts_sections_with_page_provenance_and_deduplicates(self) -> None:
        provisions, duplicates = extract_provisions_from_pages(
            (
                (4, "BE it enacted by Parliament.\n1. Short title and commencement.\nThe first provision continues."),
                (5, "Continuation of section one.\n2.(1) Definitions.\n(2) Further definitions.\n"),
                (6, "2. Duplicate section heading from a later appendix.\n"),
            )
        )

        self.assertEqual([provision.section for provision in provisions], ["1", "2"])
        self.assertEqual(provisions[0].source_page_start, 4)
        self.assertEqual(provisions[0].source_page_end, 5)
        self.assertNotIn("<<PAGE:", provisions[0].text)
        self.assertEqual(provisions[1].source_page_start, 5)
        self.assertEqual(duplicates, ("2",))


    def test_excludes_numbering_that_restarts_inside_embedded_material(self) -> None:
        provisions, excluded = extract_provisions_from_pages(
            (
                (1, "1. First provision.\n3. Third provision.\n"),
                (2, "1. State-amendment clause.\n4. Fourth provision.\n"),
            )
        )

        self.assertEqual([provision.section for provision in provisions], ["1", "3", "4"])
        self.assertEqual(excluded, ("1",))


    def test_stops_preceding_provision_before_an_embedded_schedule(self) -> None:
        provisions, excluded = extract_provisions_from_pages(
            (
                (7, "1. Main Act provision.\nFIRST SCHEDULE\n1. Form entry.\n"),
            )
        )

        self.assertEqual([provision.section for provision in provisions], ["1"])
        self.assertEqual(provisions[0].source_page_end, 7)
        self.assertNotIn("SCHEDULE", provisions[0].text)
        self.assertEqual(excluded, ("1",))

    def test_builds_a_valid_contains_graph(self) -> None:
        spec = StatuteSpec("bns", "Bharatiya Nyaya Sanhita, 2023", date(2023, 12, 25), ("bns",))
        ingestion = StatuteIngestion(
            spec=spec,
            document_id="doc:india:statute:bns:testhash",
            source_path=Path("data/raw/india_code/bns.pdf"),
            source_sha256="a" * 64,
            page_count=10,
            body_start_page=1,
            provisions=(
                Provision("1", "1. Short title.", 1, 1),
                Provision("2", "2. Definitions.", 2, 2),
            ),
            duplicate_sections=(),
        )

        graph = build_statute_graph((ingestion,), version="test-statutes-0.1.0")

        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(
            graph.nodes[1].node_id,
            "provision:india:bns:1:sha256-aaaaaaaaaaaaaaaa",
        )
        self.assertEqual(graph.edges[0].source_id, "act:india:bns")

    def test_writes_deterministic_provision_corpus(self) -> None:
        ingestion = StatuteIngestion(
            spec=StatuteSpec("bns", "Bharatiya Nyaya Sanhita, 2023", date(2023, 12, 25), ("bns",)),
            document_id="doc:india:statute:bns:testhash",
            source_path=Path("data/raw/india_code/bns.pdf"),
            source_sha256="b" * 64,
            page_count=1,
            body_start_page=1,
            provisions=(Provision("1", "1. Short title.", 1, 1),),
            duplicate_sections=(),
        )
        with TemporaryDirectory() as directory:
            output_path, checksum, count = write_provision_corpus(
                (ingestion,), Path(directory) / "provisions.jsonl"
            )
            line = output_path.read_text(encoding="utf-8")

        self.assertEqual(count, 1)
        self.assertEqual(len(checksum), 64)
        self.assertIn('"section":"1"', line)
        self.assertIn('"source_page_start":1', line)
