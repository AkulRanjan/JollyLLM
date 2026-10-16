from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legal_graph.aws_judgments import AWS_DATASET
from legal_graph.eval_report import (
    progress_line, render_evaluation, render_training_comparison, render_training_curve, running_curve, summary_lines,
    training_curve,
)
from legal_graph.outcomes import evaluate

SC2016 = "Shreyasrao/Indian-law-supreme-court-judgements-2016"
ROWS = [
    {"case_id": "a", "gold": "allowed", "predicted": "allowed", "source": SC2016},
    {"case_id": "b", "gold": "dismissed", "predicted": "allowed", "source": SC2016},
    {"case_id": "c", "gold": "dismissed", "predicted": "dismissed", "source": AWS_DATASET},
    {"case_id": "d", "gold": "disposed", "predicted": "disposed", "source": AWS_DATASET},
]

LOG_LINES = [
    "epoch 1 step 25/50 loss 2.0000 elapsed 100s\n",
    "epoch 1 step 50/50 loss 1.5000 elapsed 200s\n",
    '{"epoch": 1, "train_loss": 1.5, "validation_accuracy": 0.5, "validation_macro_f1": 0.25, "elapsed_seconds": 210.0}\n',
    "epoch 2 step 25/50 loss 1.0000 elapsed 300s\n",
    "epoch 2 step 25/50 loss 1.0000 elapsed 300s\n",  # repeated by a resumed session
    "epoch 2 step 50/50 loss 0.8000 elapsed 400s\n",
    '{"epoch": 2, "train_loss": 0.8, "validation_accuracy": 0.75, "validation_macro_f1": 0.5, "elapsed_seconds": 410.0}\n',
]


def _payload(rows: list[dict], variant: str) -> dict:
    by_source = {
        source: evaluate([r["gold"] for r in rows if r["source"] == source], [r["predicted"] for r in rows if r["source"] == source])
        for source in (SC2016, AWS_DATASET)
    }
    return {"experiment_id": "demo-v1", "variant": variant, "metrics": evaluate([r["gold"] for r in rows], [r["predicted"] for r in rows]),
            "metrics_by_source": by_source}


class ProgressTests(unittest.TestCase):
    def test_progress_line_reports_time_left_and_score_so_far(self) -> None:
        line = progress_line(ROWS[:2], 4, 120.0, "test")
        self.assertIn("test: 2/4 cases (50%)", line)
        self.assertIn("2.0 min elapsed, ~2.0 min left", line)
        self.assertIn("accuracy 0.500", line)

    def test_running_curve_ends_at_the_final_score(self) -> None:
        curve = running_curve(ROWS, every=3)
        self.assertEqual([point["cases"] for point in curve], [3, 4])
        self.assertEqual(curve[-1]["accuracy"], 0.75)

    def test_summary_names_each_dataset(self) -> None:
        text = "\n".join(summary_lines("text", _payload(ROWS, "text")))
        self.assertIn("all test cases", text)
        self.assertIn("SC-2016 test", text)
        self.assertIn("2017 test (AWS bucket)", text)


class TrainingCurveTests(unittest.TestCase):
    def test_interval_loss_is_recovered_from_running_means(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text("".join(LOG_LINES), encoding="utf-8")
            curve = training_curve(log)
        self.assertEqual([s["global_step"] for s in curve["steps"]], [25, 50, 75, 100])
        self.assertAlmostEqual(curve["steps"][1]["interval_loss"], 1.0)  # (1.5 * 50 - 2.0 * 25) / 25
        self.assertAlmostEqual(curve["steps"][3]["interval_loss"], 0.6)
        self.assertEqual([e["validation_macro_f1"] for e in curve["epochs"]], [0.25, 0.5])

    def test_kaggle_json_log_is_read_like_plain_text(self) -> None:
        records = [{"stream_name": "stdout", "time": float(i), "data": line} for i, line in enumerate(LOG_LINES)]
        with tempfile.TemporaryDirectory() as directory:
            plain, kaggle = Path(directory) / "plain.log", Path(directory) / "kaggle.log"
            plain.write_text("".join(LOG_LINES), encoding="utf-8")
            kaggle.write_text(json.dumps(records), encoding="utf-8")
            self.assertEqual(training_curve(plain), training_curve(kaggle))


class FigureTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

    def test_evaluation_figures_cover_both_passes(self) -> None:
        ablated = [dict(row, predicted="allowed") for row in ROWS]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            for suffix, rows in (("", ROWS), ("-graph-ablated", ablated)):
                (run / f"test-evaluation{suffix}.json").write_text(json.dumps(_payload(rows, "graph_typed")), encoding="utf-8")
                (run / f"test-predictions{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            report = render_evaluation(run, Path(directory) / "figures")
            self.assertEqual(len(report["figures"]), 3)
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in report["figures"]))
        self.assertIn("== graph_typed (with graph)", report["lines"])
        self.assertIn("== graph removed", report["lines"])

    def test_training_curve_figure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text("".join(LOG_LINES), encoding="utf-8")
            report = render_training_curve(log, Path(directory) / "figures", "demo")
            self.assertTrue(Path(report["figures"][0]).exists())
        self.assertIn("  epoch 2: validation accuracy 0.750, macro-F1 0.500", report["lines"])

    def test_training_comparison_overlays_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text("".join(LOG_LINES), encoding="utf-8")
            report = render_training_comparison({"text": log, "graph": log}, Path(directory) / "figures")
            self.assertTrue(Path(report["figures"][0]).exists())


if __name__ == "__main__":
    unittest.main()
