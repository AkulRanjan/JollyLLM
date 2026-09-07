from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.approvals import load_approval_policy, validate_policy_sources


class ApprovalPolicyTests(unittest.TestCase):
    def test_binds_approved_policy_to_exact_source_hashes(self) -> None:
        payload = {
            "schema_version": "1.0.0",
            "status": "approved",
            "policy_id": "test-policy",
            "audit_report_content_sha256": "a" * 64,
            "source_sha256": {"bns": "b" * 64},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            policy = load_approval_policy(path)
            validate_policy_sources(policy, {"bns": "b" * 64})
            with self.assertRaises(ValueError):
                validate_policy_sources(policy, {"bns": "c" * 64})

        self.assertEqual(policy.policy_id, "test-policy")
        self.assertEqual(len(policy.policy_sha256), 64)


if __name__ == "__main__":
    unittest.main()
