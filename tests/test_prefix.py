from __future__ import annotations

import unittest

from legal_graph.prefix import assemble_prefix


class PrefixAssemblyTests(unittest.TestCase):
    def test_prefix_preserves_order_and_masks_context_labels(self) -> None:
        assembly = assemble_prefix(
            graph_soft_tokens=((1.0, 2.0), (3.0, 4.0)),
            instruction_embeddings=((5.0, 6.0),),
            document_embeddings=((7.0, 8.0),),
            document_labels=(99,),
        )

        self.assertEqual(assembly.embeddings, ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)))
        self.assertEqual(assembly.attention_mask, (1, 1, 1, 1))
        self.assertEqual(assembly.labels, (-100, -100, -100, 99))

    def test_prefix_rejects_mismatched_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2"):
            assemble_prefix(
                graph_soft_tokens=((1.0, 2.0),),
                instruction_embeddings=((3.0,),),
                document_embeddings=(),
                document_labels=(),
            )

    def test_prefix_requires_explicit_null_or_graph_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one graph soft token"):
            assemble_prefix(
                graph_soft_tokens=(),
                instruction_embeddings=((1.0, 2.0),),
                document_embeddings=(),
                document_labels=(),
            )
