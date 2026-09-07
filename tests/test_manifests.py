from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from legal_graph.fixtures import fixture_graph
from legal_graph.manifests import (
    RegistryValidationError,
    SourceRecord,
    create_graph_manifest,
    graph_from_dict,
    graph_to_dict,
    validate_source_records,
    write_graph_and_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_graph_round_trip_retains_records(self) -> None:
        graph = fixture_graph()

        self.assertEqual(graph_from_dict(graph_to_dict(graph)), graph)

    def test_manifest_is_content_deterministic(self) -> None:
        graph = fixture_graph()
        fixed_time = datetime(2026, 9, 5, tzinfo=timezone.utc)
        first = create_graph_manifest(
            graph,
            code_revision="local-test",
            config_payload={"node_budget": 80},
            parents=("source:synthetic:fixture:v1",),
            created_at=fixed_time,
        )
        second = create_graph_manifest(
            graph,
            code_revision="local-test",
            config_payload={"node_budget": 80},
            parents=("source:synthetic:fixture:v1",),
            created_at=fixed_time,
        )

        self.assertEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.config_sha256, second.config_sha256)

    def test_export_writes_local_graph_and_manifest(self) -> None:
        graph = fixture_graph()
        manifest = create_graph_manifest(
            graph,
            code_revision="local-test",
            config_payload={"fixture": True},
        )
        with TemporaryDirectory() as directory:
            graph_path, manifest_path = write_graph_and_manifest(graph, manifest, Path(directory))
            self.assertTrue(graph_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["content_sha256"], manifest.content_sha256)

    def test_approved_source_requires_permitted_uses(self) -> None:
        record = SourceRecord(
            source_id="source:test:bad",
            display_name="bad test record",
            source_locator="local://test",
            acquired_on=date(2026, 9, 5),
            raw_sha256="a" * 64,
            license_label="test-only",
            status="approved",
            permitted_uses=(),
            redistribution_allowed=False,
        )

        with self.assertRaises(RegistryValidationError):
            validate_source_records((record,))
