from __future__ import annotations

import unittest

from legal_graph.comparison import aggregate_seeds, compare_to_projection, paired_bootstrap


class ComparisonTests(unittest.TestCase):
    def test_identical_predictions_have_zero_delta(self) -> None:
        gold = ["allowed", "dismissed", "disposed", "partly_allowed"] * 5
        result = paired_bootstrap(gold, gold, gold, resamples=200)
        self.assertEqual((result["delta"], result["lower"], result["upper"]), (0.0, 0.0, 0.0))

    def test_better_model_has_positive_interval(self) -> None:
        gold = ["allowed", "dismissed"] * 20
        result = paired_bootstrap(gold, list(gold), ["allowed"] * 40, metric="accuracy", resamples=500)
        self.assertGreater(result["lower"], 0.0)
        self.assertLess(result["share_a_not_better"], 0.05)

    def test_bootstrap_is_seeded(self) -> None:
        gold = ["allowed", "dismissed", "allowed", "disposed"] * 5
        first = paired_bootstrap(gold, ["allowed"] * 20, ["dismissed"] * 20, resamples=100)
        second = paired_bootstrap(gold, ["allowed"] * 20, ["dismissed"] * 20, resamples=100)
        self.assertEqual(first, second)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap(["allowed"], ["allowed", "allowed"], ["allowed"])

    def test_aggregate_seeds(self) -> None:
        result = aggregate_seeds([0.5, 0.7])
        self.assertEqual((result["mean"], result["n"]), (0.6, 2))
        self.assertAlmostEqual(result["std"], 0.1414213562373095)
        self.assertEqual(aggregate_seeds([0.4])["std"], 0.0)

    def test_compare_to_projection_flags_interval(self) -> None:
        projection = {"variants": [{"variant_id": "v", "splits": {"test": {
            "accuracy": 0.76, "accuracy_interval": {"lower": 0.63, "upper": 0.86}}}}]}
        self.assertFalse(compare_to_projection({"v": 0.60}, projection)[0]["inside_interval"])
        self.assertTrue(compare_to_projection({"v": 0.70}, projection)[0]["inside_interval"])
        self.assertEqual(compare_to_projection({}, projection), [])
