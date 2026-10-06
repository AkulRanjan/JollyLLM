from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.results import collect, control_key, discover_runs, markdown_table

LABELS = ("allowed", "dismissed", "disposed", "partly_allowed")


def _metrics(gold, predicted):
    from legal_graph.outcomes import evaluate

    return evaluate(gold, predicted)


def make_run(root: Path, kernel: str, experiment: str, variant: str, predicted: list[str] | None, gold: list[str]) -> Path:
    run = root / "checkpoints" / "kaggle" / kernel / "checkpoints" / "outcome" / experiment
    run.mkdir(parents=True)
    (run / "run.manifest.json").write_text(json.dumps({
        "method": "graph_outcome_qlora_v1", "experiment_id": experiment, "variant": variant, "selector": "expansion",
        "base_model_source": "m", "validation_metrics": {"accuracy": 0.5, "macro_f1": 0.4}, "best_epoch": 1,
        "subgraph_coverage": {"validation": None}, "parameter_report": {"trainable_fraction_with_encoder": 0.01},
    }), encoding="utf-8")
    if predicted is not None:
        (run / "freeze.json").write_text("{}", encoding="utf-8")
        rows = [{"case_id": f"c{i}", "gold": g, "predicted": p} for i, (g, p) in enumerate(zip(gold, predicted))]
        (run / "test-predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (run / "test-evaluation.json").write_text(json.dumps({"metrics": _metrics(gold, predicted)}), encoding="utf-8")
    return run


SETTINGS = {
    "projection_report": "missing.json", "primary_metric": "macro_f1", "bootstrap_resamples": 200,
    "controls": {"qwen3-8b": "qwen3-8b-text-v1"}, "projection_variant_for": {},
}


class ResultsTests(unittest.TestCase):
    def test_control_key(self) -> None:
        self.assertEqual(control_key("qwen3-8b-graph-typed-pcst-g2-v1"), "qwen3-8b")
        self.assertEqual(control_key("qwen3-4b-graph-typed-g2-s2-v1"), "qwen3-4b-s2")
        self.assertEqual(control_key("saul-7b-text-v1"), "saul-7b")

    def test_prefers_the_evaluated_copy_of_a_run(self) -> None:
        gold = list(LABELS) * 3
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, "lg-x-train", "qwen3-8b-text-v1", "text", None, gold)
            evaluated = make_run(root, "lg-x-evaluate", "qwen3-8b-text-v1", "text", gold, gold)
            self.assertEqual(discover_runs(root)["qwen3-8b-text-v1"], evaluated)

    def test_paired_comparison_against_the_text_control(self) -> None:
        gold = list(LABELS) * 5
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, "a", "qwen3-8b-text-v1", "text", ["allowed"] * 20, gold)
            make_run(root, "b", "qwen3-8b-graph-typed-g2-v1", "graph_typed", gold, gold)
            report = collect(root, SETTINGS)
        rows = {row["experiment_id"]: row for row in report["runs"]}
        versus = rows["qwen3-8b-graph-typed-g2-v1"]["versus_text_control"]
        self.assertEqual(versus["control"], "qwen3-8b-text-v1")
        self.assertGreater(versus["delta"], 0)
        self.assertNotIn("versus_text_control", rows["qwen3-8b-text-v1"])
        self.assertIn("qwen3-8b-graph-typed-g2-v1", markdown_table(report))

    def test_different_test_sets_are_rejected(self) -> None:
        gold = list(LABELS) * 2
        with TemporaryDirectory() as directory:
            root = Path(directory)
            make_run(root, "a", "qwen3-8b-text-v1", "text", gold, gold)
            make_run(root, "b", "qwen3-8b-graph-typed-g2-v1", "graph_typed", gold[:-1], gold[:-1])
            with self.assertRaises(ValueError):
                collect(root, SETTINGS)

    def test_seed_groups_are_aggregated(self) -> None:
        gold = list(LABELS) * 2
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1, 2):
                make_run(root, f"k{seed}", f"qwen3-4b-graph-typed-g2-s{seed}-v1", "graph_typed", gold, gold)
            report = collect(root, SETTINGS)
        group = report["seed_groups"]["qwen3-4b-graph-typed-g2"]
        self.assertEqual(group["test_macro_f1"]["n"], 2)


class ExclusionTests(unittest.TestCase):
    def test_runs_on_another_input_view_are_excluded(self) -> None:
        gold = list(LABELS) * 2
        with TemporaryDirectory() as directory:
            root = Path(directory)
            leaky = make_run(root, "a", "qwen3-8b-text-v1", "text", gold, gold)
            (leaky / "config.json").write_text(json.dumps({"input_view": "prefix-with-disposition-masking-v1"}), encoding="utf-8")
            clean = make_run(root, "b", "qwen3-8b-text-v2", "text", gold, gold)
            (clean / "config.json").write_text(json.dumps({"input_view": "body-prefix-masked-v2"}), encoding="utf-8")
            report = collect(root, {**SETTINGS, "required_input_view": "body-prefix-masked-v2"})
        self.assertEqual([row["experiment_id"] for row in report["runs"]], ["qwen3-8b-text-v2"])
        self.assertIn("qwen3-8b-text-v1", report["excluded_runs"])
