"""Freeze a trained run, then evaluate it on the held-out test split exactly once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _hashes(run_dir: Path) -> dict[str, str]:
    files = sorted(path for path in (run_dir / "adapter").rglob("*") if path.is_file()) if (run_dir / "adapter").exists() else []
    files += [path for path in (run_dir / "encoder.pt", run_dir / "config.json") if path.exists()]
    return {str(path.relative_to(run_dir)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def freeze_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    target = run_dir / "freeze.json"
    if target.exists():
        raise FileExistsError(f"{run_dir} is already frozen")
    files = _hashes(run_dir)
    if "config.json" not in files:
        raise ValueError("run directory has no config.json to freeze")
    payload = {"schema_version": "1.0.0", "status": "frozen", "files": files}
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_freeze(run_dir: Path) -> None:
    target = Path(run_dir) / "freeze.json"
    if not target.exists():
        raise ValueError("run is not frozen")
    if json.loads(target.read_text(encoding="utf-8"))["files"] != _hashes(Path(run_dir)):
        raise ValueError("run changed after freeze")


def evaluate_frozen(
    run_dir: Path,
    workspace_root: Path,
    *,
    ablate_graph: bool = False,
    base_model=None,
    tokenizer=None,
) -> dict:
    """Score the test split once with the frozen adapter (and encoder); refuse any repeat."""
    import torch
    from peft import PeftModel

    from .manifests import graph_from_dict
    from .outcome_examples import load_outcome_examples
    from .outcome_training import (
        FROZEN_VARIANTS, OutcomeModel, SubgraphBuilder, build_encoder, build_evidence, is_graph_variant, load_base_model,
        load_outcome_config, node_features, score_examples,
    )
    from .outcomes import evaluate

    run_dir, root = Path(run_dir), Path(workspace_root)
    verify_freeze(run_dir)
    suffix = "-graph-ablated" if ablate_graph else ""
    result_path = run_dir / f"test-evaluation{suffix}.json"
    if result_path.exists():
        raise FileExistsError(f"{result_path.name} already exists; the test split is evaluated once")
    config = load_outcome_config(run_dir / "config.json")
    if ablate_graph and not is_graph_variant(config.variant):
        raise ValueError("graph-token ablation applies to graph variants only")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if base_model is None:
        base_model, tokenizer = load_base_model(config, root, device)
    else:
        base_model = base_model.to(device)
    model = base_model if config.variant in FROZEN_VARIANTS else PeftModel.from_pretrained(base_model, run_dir / "adapter")
    model.eval()
    graph = graph_from_dict(json.loads((root / config.graph_path).read_text(encoding="utf-8")))
    features = node_features(config, root, graph, tokenizer, model, device)
    encoder = build_encoder(config, model.get_input_embeddings().weight.shape[1], 1.0)
    if encoder is not None:
        encoder.load_state_dict(torch.load(run_dir / "encoder.pt", map_location="cpu"))
        encoder = encoder.to(device).eval()
    wrapper = OutcomeModel(model, tokenizer, encoder, config.max_prompt_tokens, device, features,
                           build_evidence(config, root), config.max_evidence_tokens)
    examples = load_outcome_examples(root / config.corpus_path, root / config.training_data_manifest, "test",
                                     config.input_character_limit, allow_test_labels=True)
    if not examples:
        raise ValueError("no labelled test examples")
    rows = score_examples(wrapper, examples, SubgraphBuilder(graph, config), ablate_graph=ablate_graph)
    metrics = evaluate([row["gold"] for row in rows], [row["predicted"] for row in rows])
    (run_dir / f"test-predictions{suffix}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "schema_version": "1.0.0",
        "status": "measured_on_test",
        "experiment_id": config.experiment_id,
        "variant": config.variant,
        "ablate_graph": ablate_graph,
        "freeze_verified": True,
        "metrics": metrics,
    }
    result_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload
