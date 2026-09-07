from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.audit import audit_provision_candidates, write_audit_report
from legal_graph.statutes import Provision


class StatuteAuditTests(unittest.TestCase):
    def test_reports_duplicates_resets_and_structural_headings(self) -> None:
        findings = audit_provision_candidates(
            act_id="example",
            candidates=(
                Provision("1", "1. First provision.", 2, 2),
                Provision("3", "3. Third provision.", 3, 3),
                Provision("1", "1. First Schedule\nSchedule content.", 9, 9),
            ),
        )

        self.assertEqual(
            [finding.category for finding in findings],
            ["duplicate_section", "section_number_reset", "structural_heading"],
        )
        duplicate = findings[0]
        self.assertEqual(duplicate.section, "1")
        self.assertIn("pages 2-2, 9-9", duplicate.message)

    def test_writes_canonical_report(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "status": "needs_review",
            "findings": [],
        }
        with TemporaryDirectory() as directory:
            output_path, digest = write_audit_report(
                payload, Path(directory) / "audit.json"
            )
            text = output_path.read_text(encoding="utf-8")

        self.assertEqual(len(digest), 64)
        self.assertEqual(text, '{"findings":[],"schema_version":"1.0.0","status":"needs_review"}\n')


if __name__ == "__main__":
    unittest.main()
