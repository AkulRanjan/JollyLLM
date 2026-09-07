"""Local-only readiness checks for a configured QLoRA base-model experiment."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelPreflight:
    experiment_id: str
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    details: Mapping[str, object]


def load_model_registry(path: Path) -> Mapping[str, object]:
    """Load and minimally validate a local model registry without contacting a hub."""

    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load model registry {path}: {error}") from error
    if registry.get("schema_version") != "1.0.0":
        raise ValueError("model registry must use schema_version 1.0.0")
    if registry.get("local_files_only") is not True:
        raise ValueError("model registry must require local_files_only=true")
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("model registry must contain at least one experiment")
    return registry


def preflight_model_experiment(
    registry: Mapping[str, object],
    *,
    experiment_id: str,
    workspace_root: Path,
) -> ModelPreflight:
    """Report whether a named experiment can run with local files only.

    This function never invokes a downloader, package installer, model loader,
    or remote API. A false result is expected until the user stages local
    weights, the optional ML stack, and approved training data.
    """

    experiments = registry["experiments"]
    experiment = next(
        (
            item
            for item in experiments
            if isinstance(item, dict) and item.get("experiment_id") == experiment_id
        ),
        None,
    )
    if experiment is None:
        raise ValueError(f"unknown model experiment {experiment_id!r}")
    required_fields = (
        "role",
        "model_family",
        "local_model_path",
        "minimum_free_bytes",
        "required_packages",
        "required_artifact_manifests",
        "training_data_manifest",
    )
    missing = [field for field in required_fields if field not in experiment]
    if missing:
        raise ValueError(f"model experiment {experiment_id!r} is missing {', '.join(missing)}")

    blockers: list[str] = []
    warnings: list[str] = []
    model_path = workspace_root / str(experiment["local_model_path"])
    if not model_path.is_dir():
        blockers.append(f"Local model weights are not staged at {model_path.as_posix()}.")
    required_packages = tuple(str(value) for value in experiment["required_packages"])
    unavailable_packages = tuple(
        package for package in required_packages if importlib.util.find_spec(package) is None
    )
    if unavailable_packages:
        blockers.append(
            "Required local Python packages are unavailable: " + ", ".join(unavailable_packages) + "."
        )

    free_bytes = shutil.disk_usage(workspace_root).free
    minimum_free_bytes = int(experiment["minimum_free_bytes"])
    if free_bytes < minimum_free_bytes:
        blockers.append(
            f"Free disk is {free_bytes} bytes; this experiment requires at least {minimum_free_bytes} bytes."
        )

    artifact_statuses: dict[str, str] = {}
    for relative_path in experiment["required_artifact_manifests"]:
        artifact_path = workspace_root / str(relative_path)
        if not artifact_path.is_file():
            blockers.append(f"Required artifact manifest is missing: {artifact_path.as_posix()}.")
            continue
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            status = str(artifact.get("status", "unknown"))
        except json.JSONDecodeError:
            blockers.append(f"Required artifact manifest is invalid JSON: {artifact_path.as_posix()}.")
            continue
        artifact_statuses[str(relative_path)] = status
        if status != "validated":
            blockers.append(
                f"Required artifact {relative_path} has status {status!r}; training requires validated inputs."
            )

    training_manifest = workspace_root / str(experiment["training_data_manifest"])
    if not training_manifest.is_file():
        blockers.append(
            f"Approved training-data manifest is missing: {training_manifest.as_posix()}."
        )

    if experiment.get("role") == "controlled_variant":
        warnings.append("Controlled variants must use the same frozen graph/index/task split as the primary model.")
    details: dict[str, object] = {
        "local_files_only": True,
        "model_path": model_path.as_posix(),
        "model_path_exists": model_path.is_dir(),
        "free_bytes": free_bytes,
        "minimum_free_bytes": minimum_free_bytes,
        "required_packages": required_packages,
        "unavailable_packages": unavailable_packages,
        "artifact_statuses": artifact_statuses,
        "training_data_manifest": training_manifest.as_posix(),
    }
    return ModelPreflight(
        experiment_id=experiment_id,
        ready=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        details=details,
    )


def result_to_dict(result: ModelPreflight) -> dict[str, Any]:
    return {
        "experiment_id": result.experiment_id,
        "ready": result.ready,
        "blockers": list(result.blockers),
        "warnings": list(result.warnings),
        "details": dict(result.details),
    }
