"""Tests for the labelled-as-projected outcome forecast."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_graph.outcomes import LABELS
from legal_graph.projection import (
    ProjectionConfigurationError,
    confusion_to_sequences,
    learning_curve,
    load_config,
    projected_confusion,
    run,
    separation_probability,
    simulate,
    wilson_interval,
)

CONFIG_PATH = Path("configs/evaluation/sc2016-outcome-projection.json")
CONFUSABILITY = {
    "allowed": {"partly_allowed": 0.45, "dismissed": 0.35, "disposed": 0.2},
    "dismissed": {"allowed": 0.4, "disposed": 0.4, "partly_allowed": 0.2},
    "disposed": {"dismissed": 0.45, "allowed": 0.4, "partly_allowed": 0.15},
    "partly_allowed": {"allowed": 0.55, "dismissed": 0.25, "disposed": 0.2},
}


class ProjectedConfusionTests(unittest.TestCase):
    def test_every_row_preserves_its_support_total(self) -> None:
        supports = {"allowed": 25, "dismissed": 11, "disposed": 9, "partly_allowed": 5}
        recall = {"allowed": 0.86, "dismissed": 0.78, "disposed": 0.6, "partly_allowed": 0.38}
        confusion = projected_confusion(supports, recall, CONFUSABILITY)
        for label in LABELS:
            self.assertEqual(sum(confusion[label].values()), supports[label])

    def test_errors_never_land_on_the_diagonal(self) -> None:
        supports = {"allowed": 10, "dismissed": 10, "disposed": 10, "partly_allowed": 10}
        recall = {label: 0.5 for label in LABELS}
        confusion = projected_confusion(supports, recall, CONFUSABILITY)
        for label in LABELS:
            self.assertEqual(confusion[label][label], 5)
            self.assertEqual(sum(count for other, count in confusion[label].items() if other != label), 5)

    def test_perfect_and_zero_recall_are_representable(self) -> None:
        supports = {label: 4 for label in LABELS}
        perfect = projected_confusion(supports, {label: 1.0 for label in LABELS}, CONFUSABILITY)
        self.assertEqual(sum(perfect[label][label] for label in LABELS), 16)
        empty = projected_confusion(supports, {label: 0.0 for label in LABELS}, CONFUSABILITY)
        self.assertEqual(sum(empty[label][label] for label in LABELS), 0)

    def test_empty_support_yields_an_empty_row(self) -> None:
        supports = {"allowed": 0, "dismissed": 3, "disposed": 0, "partly_allowed": 0}
        confusion = projected_confusion(supports, {label: 0.7 for label in LABELS}, CONFUSABILITY)
        self.assertEqual(sum(confusion["allowed"].values()), 0)
        self.assertEqual(sum(confusion["dismissed"].values()), 3)

    def test_sequences_round_trip_the_matrix(self) -> None:
        supports = {"allowed": 7, "dismissed": 5, "disposed": 3, "partly_allowed": 2}
        recall = {"allowed": 0.8, "dismissed": 0.6, "disposed": 0.4, "partly_allowed": 0.5}
        confusion = projected_confusion(supports, recall, CONFUSABILITY)
        gold, predicted = confusion_to_sequences(confusion)
        self.assertEqual(len(gold), 17)
        self.assertEqual(len(predicted), 17)
        rebuilt = {actual: {forecast: 0 for forecast in LABELS} for actual in LABELS}
        for actual, forecast in zip(gold, predicted, strict=True):
            rebuilt[actual][forecast] += 1
        self.assertEqual(rebuilt, confusion)


class IntervalTests(unittest.TestCase):
    def test_wilson_interval_brackets_the_point_estimate(self) -> None:
        interval = wilson_interval(38, 50)
        self.assertAlmostEqual(interval["point"], 0.76)
        self.assertLess(interval["lower"], 0.76)
        self.assertGreater(interval["upper"], 0.76)
        self.assertGreater(interval["half_width"], 0.1)

    def test_small_samples_stay_inside_the_unit_range(self) -> None:
        for successes, trials in ((0, 5), (5, 5), (1, 3)):
            interval = wilson_interval(successes, trials)
            self.assertGreaterEqual(interval["lower"], 0.0)
            self.assertLessEqual(interval["upper"], 1.0)

    def test_larger_samples_tighten_the_interval(self) -> None:
        self.assertLess(wilson_interval(760, 1000)["half_width"], wilson_interval(38, 50)["half_width"])

    def test_zero_trials_is_rejected(self) -> None:
        with self.assertRaises(ProjectionConfigurationError):
            wilson_interval(0, 0)


class CurveAndSimulationTests(unittest.TestCase):
    def test_learning_curve_passes_through_its_anchor(self) -> None:
        settings = {"ceiling": 0.87, "exponent": 0.62, "anchor_labelled_cases": 359, "labelled_case_points": [50, 359, 1000]}
        curve = learning_curve(settings, 0.76)
        anchor = next(point for point in curve if point["is_current_corpus"])
        self.assertEqual(anchor["labelled_cases"], 359)
        self.assertAlmostEqual(float(anchor["projected_accuracy"]), 0.76, places=5)

    def test_learning_curve_is_monotonic_and_bounded(self) -> None:
        settings = {"ceiling": 0.87, "exponent": 0.62, "anchor_labelled_cases": 359, "labelled_case_points": [50, 100, 200, 359, 1000, 5000]}
        curve = learning_curve(settings, 0.76)
        values = [float(point["projected_accuracy"]) for point in curve]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[-1], 0.87)

    def test_learning_curve_rejects_an_anchor_above_the_ceiling(self) -> None:
        settings = {"ceiling": 0.7, "exponent": 0.62, "anchor_labelled_cases": 359, "labelled_case_points": [359]}
        with self.assertRaises(ProjectionConfigurationError):
            learning_curve(settings, 0.76)

    def test_simulation_is_seeded_and_centred(self) -> None:
        first = simulate(0.76, 2000, 50, 20260910)
        second = simulate(0.76, 2000, 50, 20260910)
        self.assertEqual(first, second)
        self.assertAlmostEqual(float(first["mean"]), 0.76, delta=0.02)
        self.assertLess(float(first["p05"]), 0.76)
        self.assertGreater(float(first["p95"]), 0.76)
        self.assertAlmostEqual(sum(float(bin_["share"]) for bin_ in first["histogram"]), 1.0, places=6)

    def test_separation_probability_grows_with_sample_size(self) -> None:
        small = separation_probability(0.76, 0.68, 50, 1000, 7)
        large = separation_probability(0.76, 0.68, 800, 1000, 7)
        self.assertLess(small, large)
        self.assertGreater(large, 0.9)


class ConfigTests(unittest.TestCase):
    def test_repository_config_loads(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(config.projection_id, "sc2016-outcome-projection-v1")
        self.assertIn(config.primary_variant_id, {variant.variant_id for variant in config.variants})
        for row in config.confusability_prior.values():
            self.assertAlmostEqual(sum(row.values()), 1.0, places=9)

    def test_unknown_method_is_rejected(self) -> None:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        data["method"] = "something_else"
        with self.assertRaises(ProjectionConfigurationError):
            load_config(self._temporary_config(data))

    def test_unknown_primary_variant_is_rejected(self) -> None:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        data["primary_variant_id"] = "not-a-variant"
        with self.assertRaises(ProjectionConfigurationError):
            load_config(self._temporary_config(data))

    def test_out_of_range_recall_is_rejected(self) -> None:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        data["variants"][2]["class_recall"]["allowed"] = 1.4
        with self.assertRaises(ProjectionConfigurationError):
            load_config(self._temporary_config(data))

    def test_incomplete_confusability_prior_is_rejected(self) -> None:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        data["confusability_prior"]["allowed"].pop("dismissed")
        with self.assertRaises(ProjectionConfigurationError):
            load_config(self._temporary_config(data))

    def _temporary_config(self, data: dict[str, object]) -> Path:
        directory = tempfile.mkdtemp()
        path = Path(directory) / "projection.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


class RunTests(unittest.TestCase):
    def test_run_is_labelled_as_a_projection_and_stays_self_consistent(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_into(config, Path(directory))
        self.assertEqual(payload["status"], "projected_not_measured")
        headline = payload["headline"]
        self.assertGreater(headline["primary_projected_test_accuracy"], headline["measured_baseline_accuracy"])
        interval = headline["accuracy_interval"]
        self.assertLess(interval["lower"], headline["primary_projected_test_accuracy"])
        self.assertGreater(interval["upper"], headline["primary_projected_test_accuracy"])

    def test_every_projected_variant_is_flagged_and_ordered(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_into(config, Path(directory))
        projected = [row for row in payload["variants"] if row["evidence"] == "projected"]
        self.assertTrue(projected)
        accuracies = [row["splits"]["test"]["accuracy"] for row in projected]
        self.assertEqual(accuracies, sorted(accuracies))
        for row in payload["variants"]:
            for split in ("validation", "test"):
                projection = row["splits"][split]
                self.assertEqual(
                    sum(projection["confusion_matrix"][label][label] for label in LABELS),
                    projection["correct"],
                )
                self.assertAlmostEqual(
                    projection["accuracy"],
                    projection["correct"] / projection["eligible_case_count"],
                    places=9,
                )

    def test_uplift_decomposition_sums_to_the_headline_gain(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_into(config, Path(directory))
        steps = payload["uplift_decomposition"]
        primary = next(step for step in steps if step["variant_id"] == config.primary_variant_id)
        total = sum(step["delta"] for step in steps[: steps.index(primary) + 1])
        self.assertAlmostEqual(total, payload["headline"]["absolute_gain_over_measured_baseline"], places=9)

    def test_label_supports_match_the_measured_baseline_case_count(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            payload = self._run_into(config, Path(directory))
        validation = payload["label_supports"]["validation"]
        labelled = sum(count for label, count in validation.items() if label in LABELS)
        self.assertEqual(labelled, payload["measured_baseline"]["eligible_case_count"])

    def test_repeated_runs_produce_the_same_content_hash(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = self._run_into(config, Path(first))
            two = self._run_into(config, Path(second))
        self.assertEqual(one["content_sha256"], two["content_sha256"])

    def _run_into(self, config, output_dir: Path) -> dict:
        from dataclasses import replace

        result = run(replace(config, output_dir=output_dir), Path.cwd())
        return json.loads(Path(str(result["report_path"])).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
