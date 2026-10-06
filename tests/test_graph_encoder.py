from __future__ import annotations

import importlib.util
import unittest


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@unittest.skipUnless(_has("torch"), "torch not installed")
class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from legal_graph.graph_encoder import GraphPrefixEncoder

        torch.manual_seed(0)
        self.torch = torch
        self.typed = GraphPrefixEncoder(node_dim=6, hidden_dim=8, model_dim=10, num_tokens=3, num_relations=4, typed=True)
        self.x = torch.randn(4, 6)
        self.query = torch.randn(10)
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    def test_output_shape(self) -> None:
        out = self.typed(self.x, self.edge_index, self.torch.tensor([0, 1, 2]), self.query)
        self.assertEqual(tuple(out.shape), (3, 10))

    def test_relation_types_change_typed_output(self) -> None:
        first = self.typed(self.x, self.edge_index, self.torch.tensor([0, 0, 0]), self.query)
        second = self.typed(self.x, self.edge_index, self.torch.tensor([0, 1, 2]), self.query)
        self.assertFalse(self.torch.allclose(first, second))

    def test_untyped_ignores_relation_types(self) -> None:
        from legal_graph.graph_encoder import GraphPrefixEncoder

        untyped = GraphPrefixEncoder(node_dim=6, hidden_dim=8, model_dim=10, num_tokens=3, num_relations=4, typed=False)
        first = untyped(self.x, self.edge_index, self.torch.tensor([0, 0, 0]), self.query)
        second = untyped(self.x, self.edge_index, self.torch.tensor([0, 1, 3]), self.query)
        self.assertTrue(self.torch.allclose(first, second))
        self.assertEqual(untyped.layers[0].relation_weight.shape[0], 1)

    def test_empty_graph_returns_learned_null_tokens(self) -> None:
        torch = self.torch
        empty = self.typed(torch.zeros(0, 6), torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long), self.query)
        self.assertTrue(torch.equal(empty, self.typed.null_tokens))

    def test_gradients_reach_relation_weights(self) -> None:
        out = self.typed(self.x, self.edge_index, self.torch.tensor([0, 1, 2]), self.query)
        out.sum().backward()
        self.assertIsNotNone(self.typed.layers[0].relation_weight.grad)

    def test_mean_normalisation_per_relation(self) -> None:
        from legal_graph.graph_encoder import RGCNLayer

        torch = self.torch
        layer = RGCNLayer(1, 1, 1)
        with torch.no_grad():
            layer.self_loop.weight.zero_()
            layer.self_loop.bias.zero_()
            layer.relation_weight.fill_(1.0)
        h = torch.tensor([[2.0], [4.0], [0.0]])
        out = layer(h, torch.tensor([[0, 1], [2, 2]]), torch.tensor([0, 0]))
        self.assertAlmostEqual(float(out[2, 0]), 3.0)

    def test_subgraph_tensors_maps_indices(self) -> None:
        from legal_graph.graph_encoder import subgraph_tensors
        from legal_graph.query_graph import QuerySubgraph

        sub = QuerySubgraph("q", ("a", "b"), ((0, 1, 4), (1, 0, 15)), ("a",), "ok")
        features = {"a": self.torch.ones(6), "b": self.torch.zeros(6)}
        x, index, types = subgraph_tensors(sub, features, "cpu")
        self.assertEqual(tuple(x.shape), (2, 6))
        self.assertEqual(index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(types.tolist(), [4, 15])
        empty = subgraph_tensors(QuerySubgraph("q", (), (), (), "empty"), features, "cpu")
        self.assertEqual(tuple(empty[0].shape), (0, 6))
