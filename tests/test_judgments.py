from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.judgments import (
    SC2016_COURT_ID,
    build_judgment_graph,
    extraction_report,
    ingest_sc2016_directory,
    write_judgment_corpus,
    write_training_manifest,
)
from legal_graph.models import GraphSnapshot, NodeRecord, NodeType


class JudgmentIngestionTests(unittest.TestCase):
    def test_ingests_pairs_writes_corpus_and_builds_provision_edges(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "sc-2016"
            metadata_root = root / "extracted_jsons"
            markdown_root = root / "extracted_mds"
            metadata_root.mkdir(parents=True)
            markdown_root.mkdir()
            payload = {
                "filename": "2016-1-1-1-en.md",
                "metadata": {"year": 2016, "month": 1, "page_start": 1, "page_end": 1},
                "entities": {
                    "case_title": {"title": "Alpha v. Beta"},
                    "judges": [{"name": "A. Judge"}],
                    "parties": [{"name": "Alpha"}, {"name": "Beta"}],
                    "topics": [{"text": "Criminal law"}],
                },
            }
            (metadata_root / "2016-1-1-1-en.json").write_text(json.dumps(payload), encoding="utf-8")
            (markdown_root / "2016-1-1-1-en.md").write_text(
                "# Alpha v. Beta\nJANUARY 13, 2016\nSection 302 IPC applies. (2015) 1 SCC 10.",
                encoding="utf-8",
            )

            ingestion = ingest_sc2016_directory(root)
            record = ingestion.judgments[0]
            self.assertEqual(record.title, "Alpha v. Beta")
            self.assertEqual(record.decision_date, date(2016, 1, 13))
            self.assertIn(record.origin_split.value, {"train", "validation", "test"})
            self.assertEqual(len(ingestion.source_sha256), 64)

            output, checksum, count = write_judgment_corpus(ingestion, root / "clean" / "judgments.jsonl")
            self.assertTrue(output.is_file())
            self.assertEqual(len(checksum), 64)
            self.assertEqual(count, 1)
            training_path, training_sha256 = write_training_manifest(
                ingestion,
                root / "clean" / "training.manifest.json",
                corpus_sha256=checksum,
                graph_content_sha256="c" * 64,
            )
            training_payload = json.loads(training_path.read_text(encoding="utf-8"))
            self.assertEqual(len(training_sha256), 64)
            self.assertEqual(training_payload["training_splits"], ["train"])
            self.assertEqual(training_payload["held_out_splits"], ["test"])

            provision = NodeRecord(
                node_id="provision:india:ipc:302:sha256-aaaaaaaaaaaaaaaa",
                node_type=NodeType.PROVISION,
                label="Indian Penal Code section 302",
                source_document_ids=("doc:ipc",),
            )
            base_graph = GraphSnapshot(
                version="base-0.1.0",
                schema_version="1.0.0",
                nodes=(provision,),
                edges=(),
            )
            assembly = build_judgment_graph(base_graph, ingestion, version="test-judgments-0.1.0")
            relations = {edge.relation_type.value for edge in assembly.graph.edges}
            self.assertEqual(len(assembly.graph.nodes), 3)
            self.assertIn(SC2016_COURT_ID, assembly.graph.nodes_by_id)
            self.assertIn("decided_by", relations)
            self.assertIn("refers_to_provision", relations)
            self.assertEqual(extraction_report(assembly)["unresolved_count"], 1)

    def test_requires_a_corresponding_markdown_transcript(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_root = root / "extracted_jsons"
            (root / "extracted_mds").mkdir(parents=True)
            metadata_root.mkdir()
            (metadata_root / "missing.json").write_text(
                json.dumps({"filename": "missing.md"}), encoding="utf-8"
            )

            with self.assertRaises(FileNotFoundError):
                ingest_sc2016_directory(root)
