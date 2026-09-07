from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.model_preflight import preflight_model_experiment


class ModelPreflightTests(unittest.TestCase):
    def test_requires_local_weights_approved_data_and_validated_artifacts(self) -> None:
        registry = {
            "schema_version": "1.0.0",
            "local_files_only": True,
            "experiments": [
                {
                    "experiment_id": "saullm-test",
                    "role": "controlled_variant",
                    "model_family": "saullm",
                    "local_model_path": "models/saullm",
                    "minimum_free_bytes": 0,
                    "required_packages": [],
                    "required_artifact_manifests": ["data/index.manifest.json"],
                    "training_data_manifest": "data/train.manifest.json",
                }
            ],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "models/saullm").mkdir(parents=True)
            (root / "data").mkdir()
            (root / "data/index.manifest.json").write_text(
                json.dumps({"status": "validated"}), encoding="utf-8"
            )
            (root / "data/train.manifest.json").write_text("{}", encoding="utf-8")

            ready = preflight_model_experiment(
                registry, experiment_id="saullm-test", workspace_root=root
            )
            self.assertTrue(ready.ready)
            self.assertIn("Controlled variants", ready.warnings[0])

            (root / "data/index.manifest.json").write_text(
                json.dumps({"status": "provisional"}), encoding="utf-8"
            )
            blocked = preflight_model_experiment(
                registry, experiment_id="saullm-test", workspace_root=root
            )

        self.assertFalse(blocked.ready)
        self.assertIn("provisional", blocked.blockers[0])


if __name__ == "__main__":
    unittest.main()
