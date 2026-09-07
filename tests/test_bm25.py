from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.bm25 import (
    Bm25Document,
    build_bm25_index,
    index_to_dict,
    load_bm25_index,
    search_bm25,
    write_bm25_index,
)


class Bm25Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            Bm25Document(
                node_id="provision:india:bns:103",
                text="103. Punishment for murder and culpable homicide.",
                metadata={"act_id": "bns", "section": "103"},
            ),
            Bm25Document(
                node_id="provision:india:bnss:173",
                text="173. Information in cognizable cases.",
                metadata={"act_id": "bnss", "section": "173"},
            ),
            Bm25Document(
                node_id="provision:india:crpc:154",
                text="154. Information in cognizable cases to police.",
                metadata={"act_id": "crpc", "section": "154"},
            ),
        )

    def test_build_is_deterministic_and_returns_ranked_results(self) -> None:
        digest = "a" * 64
        first = build_bm25_index(self.documents, corpus_digest=digest)
        second = build_bm25_index(tuple(reversed(self.documents)), corpus_digest=digest)

        self.assertEqual(index_to_dict(first), index_to_dict(second))
        results = search_bm25(first, "punishment for murder", limit=2)
        self.assertEqual(results[0].node_id, "provision:india:bns:103")
        self.assertGreater(results[0].score, 0)
        self.assertEqual(search_bm25(first, "!!!"), ())

    def test_round_trip_preserves_search_behavior_and_lineage(self) -> None:
        index = build_bm25_index(self.documents, corpus_digest="b" * 64)
        with TemporaryDirectory() as directory:
            index_path, manifest_path, manifest = write_bm25_index(index, Path(directory))
            restored = load_bm25_index(index_path)

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(manifest.status, "provisional")
            self.assertEqual(manifest.parents, ("sha256:" + "b" * 64,))
            with self.assertRaises(ValueError):
                write_bm25_index(index, Path(directory), status="validated")
            _, _, approved_manifest = write_bm25_index(
                index,
                Path(directory),
                status="validated",
                approval_policy_sha256="c" * 64,
            )
            self.assertEqual(approved_manifest.status, "validated")
            self.assertIn("approval-policy:sha256:" + "c" * 64, approved_manifest.parents)
            self.assertEqual(
                search_bm25(restored, "cognizable cases", limit=2),
                search_bm25(index, "cognizable cases", limit=2),
            )


if __name__ == "__main__":
    unittest.main()
