from __future__ import annotations

from pathlib import Path
import unittest

from legal_graph.outcome_training import load_outcome_config

CONFIG_DIR = Path("configs/training/outcome")
MATCHED = ("model_revision", "max_prompt_tokens", "epochs", "learning_rate", "seed", "lora_rank",
           "lora_target_modules", "max_train_examples", "input_character_limit", "gradient_accumulation")


class OutcomeConfigTests(unittest.TestCase):
    def test_every_config_loads_into_its_own_output_dir(self) -> None:
        paths = sorted(CONFIG_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 40)
        outputs = set()
        for path in paths:
            with self.subTest(path=path.name):
                config = load_outcome_config(path)
                self.assertTrue(str(config.output_dir).replace("\\", "/").startswith("checkpoints/outcome/"))
                self.assertEqual(config.input_character_limit, 6000)
                outputs.add(config.output_dir)
        self.assertEqual(len(outputs), len(paths))

    def test_headline_variants_match_except_variant_and_selector(self) -> None:
        text = load_outcome_config(CONFIG_DIR / "qwen3-8b-text.json")
        for stem in ("graph-untyped", "graph-typed", "graph-typed-pcst", "graph-random"):
            other = load_outcome_config(CONFIG_DIR / f"qwen3-8b-{stem}.json")
            for field in MATCHED:
                self.assertEqual(getattr(text, field), getattr(other, field), f"{stem}: {field}")

    def test_balanced_recipe_pair_matches_except_variant_and_selector(self) -> None:
        text = load_outcome_config(CONFIG_DIR / "qwen3-8b-text-bal.json")
        graph = load_outcome_config(CONFIG_DIR / "qwen3-8b-graph-typed-pcst-g2-bal.json")
        for field in MATCHED + ("class_weighting", "warmup_ratio", "lr_schedule", "max_grad_norm"):
            self.assertEqual(getattr(text, field), getattr(graph, field), field)
        self.assertEqual((text.class_weighting, text.lr_schedule), ("inverse_frequency", "linear"))

    def test_aws_data_pair_matches_except_variant_and_selector(self) -> None:
        text = load_outcome_config(CONFIG_DIR / "qwen3-8b-text-g3-bal.json")
        graph = load_outcome_config(CONFIG_DIR / "qwen3-8b-graph-statute-g3-bal.json")
        for field in MATCHED + ("class_weighting", "warmup_ratio", "lr_schedule", "max_grad_norm", "corpus_path",
                                "training_data_manifest", "graph_path", "input_view", "checkpoint_every_examples"):
            self.assertEqual(getattr(text, field), getattr(graph, field), field)
        self.assertEqual((graph.variant, graph.selector), ("graph_typed", "cocitation"))
        self.assertEqual(text.variant, "text")
        self.assertGreater(text.max_train_examples, 3000)  # the cap must not cut the added training years

    def test_seed_replicates_differ_only_in_seed(self) -> None:
        first = load_outcome_config(CONFIG_DIR / "qwen3-4b-graph-typed-s1.json")
        third = load_outcome_config(CONFIG_DIR / "qwen3-4b-graph-typed-s3.json")
        self.assertNotEqual(first.seed, third.seed)
        self.assertEqual(first.learning_rate, third.learning_rate)

    def test_primary_variant_uses_typed_pcst(self) -> None:
        primary = load_outcome_config(CONFIG_DIR / "qwen3-8b-graph-typed-pcst.json")
        self.assertEqual((primary.variant, primary.selector), ("graph_typed", "pcst"))
