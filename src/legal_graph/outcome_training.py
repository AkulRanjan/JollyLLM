"""Graph-augmented (or text-only) QLoRA outcome classifier: training and validation.

Variants share one code path. ``text`` adds no prefix; ``graph_untyped`` and
``graph_typed`` prepend K soft tokens from the R-GCN encoder over a split- and
time-safe query subgraph; ``graph_random`` feeds the same encoder a random subgraph
of matched size. Only the verbalised label tokens are supervised, and the label is
chosen by scoring the four candidate continuations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .manifests import graph_from_dict
from .models import GraphSnapshot, RetrievalConfig
from .outcome_examples import OutcomeExample, load_outcome_examples, node_feature_texts
from .outcome_prompt import label_token_ids, pick_label, prompt_token_ids
from .outcomes import evaluate
from .query_graph import NUM_RELATION_TYPES, SELECTORS, QuerySubgraph, build_query_subgraph
from .training import _parameter_report, _pin_base_model, _sha256_file

METHOD = "graph_outcome_qlora_v1"
VARIANTS = ("text", "graph_untyped", "graph_typed", "graph_random", "zero_shot", "bm25_prompt")
FROZEN_VARIANTS = ("zero_shot", "bm25_prompt")  # no adapter, no training: the base model is scored as is


def is_graph_variant(variant: str) -> bool:
    return variant.startswith("graph_")
PARAMETER_CAP = 0.02


@dataclass(frozen=True, slots=True)
class OutcomeTrainConfig:
    experiment_id: str
    variant: str
    selector: str
    model_source: str
    model_revision: str
    local_model_path: Path
    corpus_path: Path
    training_data_manifest: Path
    graph_path: Path
    provisions_path: Path
    output_dir: Path
    input_character_limit: int
    max_prompt_tokens: int
    epochs: int
    learning_rate: float
    encoder_learning_rate: float
    gradient_accumulation: int
    seed: int
    quantize_4bit: bool
    min_cuda_memory_bytes: int
    max_train_examples: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    num_graph_tokens: int
    graph_hidden_dim: int
    graph_layers: int
    node_budget: int
    expansion_hops: int
    node_feature_tokens: int
    extractor: str = "v1"
    bm25_top_k: int = 0
    max_evidence_tokens: int = 0
    checkpoint_every_examples: int = 0
    input_view: str = "prefix-with-disposition-masking-v1"


def load_outcome_config(path: Path) -> OutcomeTrainConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def config_from_dict(data: Mapping[str, Any]) -> OutcomeTrainConfig:
    if data.get("schema_version") != "1.0.0" or data.get("method") != METHOD:
        raise ValueError(f"config must use schema 1.0.0 and {METHOD}")
    variant = str(data["variant"])
    selector = "random" if variant == "graph_random" else str(data.get("selector", "expansion"))
    if variant not in VARIANTS or selector not in SELECTORS:
        raise ValueError(f"unknown variant {variant!r} or selector {selector!r}")
    if variant != "graph_random" and selector == "random":
        raise ValueError("the random selector belongs to the graph_random variant only")
    lora, graph = data["lora"], data["graph"]
    config = OutcomeTrainConfig(
        experiment_id=str(data["experiment_id"]),
        variant=variant,
        selector=selector,
        model_source=str(data["model_source"]),
        model_revision=str(data["model_revision"]),
        local_model_path=Path(data["local_model_path"]),
        corpus_path=Path(data["corpus_path"]),
        training_data_manifest=Path(data["training_data_manifest"]),
        graph_path=Path(data["graph_path"]),
        provisions_path=Path(data["provisions_path"]),
        output_dir=Path(data["output_dir"]),
        input_character_limit=int(data["input_character_limit"]),
        max_prompt_tokens=int(data["max_prompt_tokens"]),
        epochs=int(data["epochs"]),
        learning_rate=float(data["learning_rate"]),
        encoder_learning_rate=float(data["encoder_learning_rate"]),
        gradient_accumulation=int(data["gradient_accumulation"]),
        seed=int(data["seed"]),
        quantize_4bit=bool(data["quantize_4bit"]),
        min_cuda_memory_bytes=int(data["min_cuda_memory_bytes"]),
        max_train_examples=int(data["max_train_examples"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        lora_target_modules=tuple(str(value) for value in lora["target_modules"]),
        num_graph_tokens=int(graph["num_tokens"]),
        graph_hidden_dim=int(graph["hidden_dim"]),
        graph_layers=int(graph["layers"]),
        node_budget=int(graph["node_budget"]),
        expansion_hops=int(graph["expansion_hops"]),
        node_feature_tokens=int(graph["node_feature_tokens"]),
        extractor=str(data.get("extractor", "v1")),
        bm25_top_k=int(data.get("evidence", {}).get("top_k", 0)),
        max_evidence_tokens=int(data.get("evidence", {}).get("max_tokens", 0)),
        checkpoint_every_examples=int(data.get("checkpoint_every_examples", 0)),
        input_view=str(data.get("input_view", "prefix-with-disposition-masking-v1")),
    )
    from .outcomes import INPUT_VIEWS

    if config.input_view not in INPUT_VIEWS:
        raise ValueError(f"unknown input view {config.input_view!r}")
    if config.checkpoint_every_examples < 0 or config.checkpoint_every_examples % config.gradient_accumulation:
        raise ValueError("checkpoint_every_examples must be a non-negative multiple of gradient_accumulation")
    if config.extractor not in ("v1", "v2"):
        raise ValueError(f"unknown extractor {config.extractor!r}")
    uses_evidence = config.bm25_top_k > 0 or config.max_evidence_tokens > 0
    if (variant == "bm25_prompt") != uses_evidence or (variant == "bm25_prompt" and min(config.bm25_top_k, config.max_evidence_tokens) <= 0):
        raise ValueError("evidence.top_k and evidence.max_tokens are required by, and only by, bm25_prompt")
    if min(config.epochs, config.gradient_accumulation, config.max_train_examples, config.max_prompt_tokens,
           config.input_character_limit, config.num_graph_tokens, config.node_budget, config.lora_rank) <= 0:
        raise ValueError("count configuration values must be positive")
    return config


def config_to_dict(config: OutcomeTrainConfig) -> dict[str, Any]:
    """Inverse of config_from_dict, so a run directory can be reloaded for evaluation."""
    flat = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    moved = {"lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules", "num_graph_tokens", "graph_hidden_dim",
             "graph_layers", "node_budget", "expansion_hops", "node_feature_tokens", "bm25_top_k", "max_evidence_tokens"}
    nested: dict[str, Any] = {"schema_version": "1.0.0", "method": METHOD}
    nested.update({key: value for key, value in flat.items() if key not in moved})
    nested["lora"] = {"rank": config.lora_rank, "alpha": config.lora_alpha, "dropout": config.lora_dropout,
                      "target_modules": list(config.lora_target_modules)}
    nested["graph"] = {"num_tokens": config.num_graph_tokens, "hidden_dim": config.graph_hidden_dim,
                       "layers": config.graph_layers, "node_budget": config.node_budget,
                       "expansion_hops": config.expansion_hops, "node_feature_tokens": config.node_feature_tokens}
    if config.variant == "bm25_prompt":
        nested["evidence"] = {"top_k": config.bm25_top_k, "max_tokens": config.max_evidence_tokens}
    return nested


class OutcomeModel:
    """The (PEFT) causal LM, the optional graph encoder, and the node feature table."""

    def __init__(self, lm, tokenizer, encoder, max_prompt_tokens: int, device, features: Mapping | None = None,
                 evidence=None, max_evidence_tokens: int = 0) -> None:
        import torch

        self.torch = torch
        self.lm = lm
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_prompt_tokens = max_prompt_tokens
        self.device = device
        self.features = dict(features or {})
        self.label_ids = label_token_ids(tokenizer)
        self.eos = tokenizer.eos_token_id
        self.evidence = evidence
        self.max_evidence_tokens = max_evidence_tokens

    def _embed(self, ids: list[int]):
        embedding = self.lm.get_input_embeddings()
        return embedding(self.torch.tensor(ids, device=self.device)).unsqueeze(0)

    def prefix_for(self, sub: QuerySubgraph | None, query=None, *, ablate: bool = False):
        """K soft tokens [1, K, d], the learned null tokens, or None for the text variant."""
        if self.encoder is None or sub is None:
            return None
        if ablate or not sub.node_ids:
            return self.encoder.null_tokens.unsqueeze(0)
        from .graph_encoder import subgraph_tensors

        x, index, types = subgraph_tensors(sub, self.features, self.device)
        weight = self.encoder.input_proj.weight
        return self.encoder(x.to(weight.dtype), index, types, query.to(weight.dtype)).unsqueeze(0)

    def _prompt(self, example: OutcomeExample, sub: QuerySubgraph | None, *, ablate: bool = False):
        evidence = self.evidence(example) if self.evidence is not None else ""
        prompt = prompt_token_ids(self.tokenizer, example.view_text, self.max_prompt_tokens,
                                  evidence=evidence, max_evidence_tokens=self.max_evidence_tokens)
        embeds = self._embed(prompt)
        prefix = self.prefix_for(sub, embeds[0].detach().float().mean(dim=0), ablate=ablate)
        if prefix is not None:
            embeds = self.torch.cat([prefix.to(embeds.dtype), embeds], dim=1)
        return embeds

    def loss(self, example: OutcomeExample, sub: QuerySubgraph | None):
        if example.label is None:
            raise ValueError("training requires labelled examples")
        prompt = self._prompt(example, sub)
        answer = list(self.label_ids[example.label]) + [self.eos]
        embeds = self.torch.cat([prompt, self._embed(answer).to(prompt.dtype)], dim=1)
        labels = self.torch.tensor([[-100] * prompt.shape[1] + answer], device=self.device)
        return self.lm(inputs_embeds=embeds, labels=labels, use_cache=False).loss

    def score(self, example: OutcomeExample, sub: QuerySubgraph | None, *, ablate_graph: bool = False) -> dict[str, float]:
        """Total log-probability of each label continuation, sharing one prompt pass."""
        torch = self.torch
        with torch.no_grad():
            prompt = self._prompt(example, sub, ablate=ablate_graph)
            out = self.lm(inputs_embeds=prompt, use_cache=True)
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            scores: dict[str, float] = {}
            for label, ids in self.label_ids.items():
                total = float(first[ids[0]])
                if len(ids) > 1:
                    cache = copy.deepcopy(out.past_key_values)
                    follow = self.lm(input_ids=torch.tensor([list(ids[:-1])], device=self.device),
                                     past_key_values=cache, use_cache=True)
                    log_probs = torch.log_softmax(follow.logits[0].float(), dim=-1)
                    total += float(sum(log_probs[i, ids[i + 1]] for i in range(len(ids) - 1)))
                scores[label] = total
        return scores


class SubgraphBuilder:
    """Builds (and caches) the query subgraph for each example under one run's policy."""

    def __init__(self, graph: GraphSnapshot, config: OutcomeTrainConfig) -> None:
        self.graph = graph
        self.config = config
        self.retrieval = RetrievalConfig(node_budget=config.node_budget, expansion_hops=config.expansion_hops)
        self._cache: dict[str, QuerySubgraph] = {}

    def __call__(self, example: OutcomeExample) -> QuerySubgraph | None:
        if not is_graph_variant(self.config.variant):
            return None
        if example.case_id not in self._cache:
            self._cache[example.case_id] = build_query_subgraph(
                self.graph,
                query_id=example.case_id,
                document_id=example.document_id,
                view_text=example.view_text,
                as_of_date=example.as_of,
                exclude_node_ids=frozenset({example.case_id}),
                config=self.retrieval,
                selector=self.config.selector,
                random_seed=self.config.seed,
                extractor=self.config.extractor,
            )
        return self._cache[example.case_id]

    def coverage(self, examples: Iterable[OutcomeExample]) -> dict[str, float] | None:
        subs = [self(example) for example in examples]
        if not subs or subs[0] is None:
            return None
        sizes = [len(sub.node_ids) for sub in subs]
        return {
            "nonempty_share": sum(sub.status == "ok" for sub in subs) / len(subs),
            "mean_nodes": sum(sizes) / len(sizes),
            "mean_edges": sum(len(sub.edges) for sub in subs) / len(subs),
        }


def build_encoder(config: OutcomeTrainConfig, model_dim: int, output_rms: float):
    if not is_graph_variant(config.variant):
        return None
    from .graph_encoder import GraphPrefixEncoder

    return GraphPrefixEncoder(
        node_dim=model_dim,
        hidden_dim=config.graph_hidden_dim,
        model_dim=model_dim,
        num_tokens=config.num_graph_tokens,
        num_relations=NUM_RELATION_TYPES,
        typed=config.variant == "graph_typed",
        num_layers=config.graph_layers,
        output_rms=output_rms,
    )


def node_features(config: OutcomeTrainConfig, root: Path, graph: GraphSnapshot, tokenizer, lm, device) -> dict:
    if not is_graph_variant(config.variant):
        return {}
    from .graph_encoder import embed_node_texts

    texts = node_feature_texts(graph, root / config.provisions_path)
    return embed_node_texts(texts, tokenizer, lm.get_input_embeddings(), config.node_feature_tokens, device)


def build_evidence(config: OutcomeTrainConfig, root: Path):
    if config.variant != "bm25_prompt":
        return None
    from .outcome_evidence import Bm25Evidence

    return Bm25Evidence(root / config.provisions_path, config.bm25_top_k)


def score_examples(wrapper: OutcomeModel, examples, subgraphs: SubgraphBuilder, *, ablate_graph: bool = False) -> list[dict]:
    rows = []
    for example in examples:
        scores = wrapper.score(example, subgraphs(example), ablate_graph=ablate_graph)
        rows.append({"case_id": example.case_id, "gold": example.label, "predicted": pick_label(scores), "scores": scores})
    return rows


CHECKPOINT_DIR = "latest-checkpoint"


def train_outcome(
    config: OutcomeTrainConfig, workspace_root: Path, *, model=None, tokenizer=None, resume: bool = False,
) -> dict[str, Any]:
    import torch

    root = Path(workspace_root)
    output = root / config.output_dir
    resume_state = _resume_state(output, config) if resume else None
    if resume_state is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output} (pass resume=True to continue an interrupted run)")
    corpus, manifest = root / config.corpus_path, root / config.training_data_manifest
    train = load_outcome_examples(corpus, manifest, "train", config.input_character_limit, input_view_id=config.input_view)
    validation = load_outcome_examples(corpus, manifest, "validation", config.input_character_limit, input_view_id=config.input_view)
    if not train or not validation:
        raise ValueError("train and validation splits must both contain labelled cases")
    if any(example.label is None for example in (*train, *validation)):
        raise ValueError("training and validation examples must be labelled")
    ordered = sorted(train, key=lambda e: hashlib.sha256(f"{config.seed}|{e.case_id}".encode("utf-8")).hexdigest())
    train = tuple(ordered[: config.max_train_examples])

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        lm, tokenizer = load_base_model(config, root, device)
    else:
        lm = model.to(device)
    if config.variant in FROZEN_VARIANTS:
        return _score_frozen(config, root, output, lm, tokenizer, device, validation, corpus, manifest)
    lm = attach_lora(lm, config)
    graph = graph_from_dict(json.loads((root / config.graph_path).read_text(encoding="utf-8")))
    features = node_features(config, root, graph, tokenizer, lm, device)
    weight = lm.get_input_embeddings().weight
    encoder = build_encoder(config, weight.shape[1], float(weight.detach().float().pow(2).mean().sqrt()))
    if encoder is not None:
        encoder = encoder.to(device)

    report = _parameter_report(lm)
    encoder_parameters = sum(p.numel() for p in encoder.parameters()) if encoder is not None else 0
    trainable = report["trainable_parameters"] + encoder_parameters
    fraction = trainable / (report["total_parameters"] + encoder_parameters)
    if fraction > PARAMETER_CAP and os.environ.get("LEGAL_GRAPH_SKIP_PARAM_CAP") != "1":
        raise RuntimeError(f"trainable fraction {fraction:.4f} exceeds the {PARAMETER_CAP:.0%} cap")

    wrapper = OutcomeModel(lm, tokenizer, encoder, config.max_prompt_tokens, device, features)
    subgraphs = SubgraphBuilder(graph, config)
    coverage = {"train": subgraphs.coverage(train), "validation": subgraphs.coverage(validation)}
    groups = [{"params": [p for p in lm.parameters() if p.requires_grad], "lr": config.learning_rate}]
    if encoder is not None:
        groups.append({"params": list(encoder.parameters()), "lr": config.encoder_learning_rate})
    optimiser = torch.optim.AdamW(groups)

    order_rng = random.Random(config.seed)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    first_epoch, skip, running_carry, resumed_order = 1, 0, 0.0, None
    checkpoints_written = 0
    if resume_state is not None:
        state = resume_state["state"]
        _restore(resume_state["path"], lm, encoder, optimiser, device)
        order_rng.setstate(_tuple_state(state["order_rng"]))
        history, best = state["history"], state["best"]
        checkpoints_written = state["checkpoints_written"]
        by_id = {example.case_id: example for example in train}
        if state["step"] >= len(train):  # the saved point is an epoch boundary
            first_epoch = state["epoch"] + 1
        else:
            first_epoch, skip, running_carry = state["epoch"], state["step"], state["running"]
            resumed_order = [by_id[case_id] for case_id in state["order"]]
        print(f"resuming {config.experiment_id} at epoch {first_epoch} step {skip}", flush=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "config.json", config_to_dict(config))
    for epoch in range(first_epoch, config.epochs + 1):
        lm.train()
        if encoder is not None:
            encoder.train()
        if resumed_order is not None and epoch == first_epoch:
            order, running = resumed_order, running_carry
        else:
            order = list(train)
            order_rng.shuffle(order)
            running, skip = 0.0, 0
        for step, example in enumerate(order, start=1):
            if step <= skip:
                continue
            loss = wrapper.loss(example, subgraphs(example))
            (loss / config.gradient_accumulation).backward()
            running += float(loss.detach())
            if step % config.gradient_accumulation == 0 or step == len(order):
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                every = config.checkpoint_every_examples
                if every and step % every == 0 and step < len(order):
                    checkpoints_written += 1
                    _write_training_checkpoint(output, lm, encoder, optimiser, {
                        "epoch": epoch, "step": step, "running": running, "order": [e.case_id for e in order],
                        "order_rng": order_rng.getstate(), "history": history, "best": best,
                        "checkpoints_written": checkpoints_written,
                    })
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{len(order)} loss {running / step:.4f} "
                      f"elapsed {time.time() - started:.0f}s", flush=True)
        lm.eval()
        if encoder is not None:
            encoder.eval()
        predictions = score_examples(wrapper, validation, subgraphs)
        metrics = evaluate([row["gold"] for row in predictions], [row["predicted"] for row in predictions])
        history.append({
            "epoch": epoch,
            "train_loss": running / len(order),
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "elapsed_seconds": time.time() - started,
        })
        print(json.dumps(history[-1]), flush=True)
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = {"epoch": epoch, "metrics": metrics, "predictions": predictions}
            _save_checkpoint(output, lm, tokenizer, encoder)
        if config.checkpoint_every_examples and epoch < config.epochs:
            checkpoints_written += 1
            _write_training_checkpoint(output, lm, encoder, optimiser, {
                "epoch": epoch, "step": len(order), "running": running, "order": [e.case_id for e in order],
                "order_rng": order_rng.getstate(), "history": history, "best": best,
                "checkpoints_written": checkpoints_written,
            })

    assert best is not None
    _write_jsonl(output / "validation-predictions.jsonl", best["predictions"])
    _write_json(output / "validation-metrics.json", best["metrics"])
    _write_json(output / "subgraph-stats.json", coverage)
    payload = {
        "schema_version": "1.0.0",
        "status": "trained_not_frozen",
        "experiment_id": config.experiment_id,
        "method": METHOD,
        "variant": config.variant,
        "selector": config.selector,
        "base_model_source": config.model_source,
        "base_model_revision": config.model_revision,
        "config_sha256": _sha256_file(output / "config.json"),
        "corpus_sha256": _sha256_file(corpus),
        "training_manifest_sha256": _sha256_file(manifest),
        "graph_sha256": _sha256_file(root / config.graph_path),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "best_epoch": best["epoch"],
        "history": history,
        "validation_metrics": {"accuracy": best["metrics"]["accuracy"], "macro_f1": best["metrics"]["macro_f1"]},
        "subgraph_coverage": coverage,
        "parameter_report": {**report, "encoder_parameters": encoder_parameters, "trainable_fraction_with_encoder": fraction},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "test_labels_accessed": False,
        "resumed": resume_state is not None,
        "training_checkpoints_written": checkpoints_written,
    }
    _write_json(output / "run.manifest.json", payload)
    _remove_tree(output / CHECKPOINT_DIR)  # the finished run keeps only its best-validation adapter
    return payload


def _write_training_checkpoint(output: Path, lm, encoder, optimiser, state: dict[str, Any]) -> None:
    """Atomically replace output/latest-checkpoint with LoRA, encoder, optimiser, RNG and loop state."""
    import torch
    from peft import get_peft_model_state_dict

    staging = output / f"{CHECKPOINT_DIR}.tmp"
    _remove_tree(staging)
    staging.mkdir(parents=True)
    torch.save(get_peft_model_state_dict(lm), staging / "lora.pt")
    if encoder is not None:
        torch.save(encoder.state_dict(), staging / "encoder.pt")
    torch.save(optimiser.state_dict(), staging / "optimizer.pt")
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, staging / "rng.pt")
    _write_json(staging / "state.json", state)
    previous = output / f"{CHECKPOINT_DIR}.old"
    _remove_tree(previous)
    current = output / CHECKPOINT_DIR
    if current.exists():
        current.rename(previous)
    staging.rename(current)
    _remove_tree(previous)
    print(f"checkpoint: epoch {state['epoch']} step {state['step']} -> {current}", flush=True)


def _resume_state(output: Path, config: OutcomeTrainConfig) -> dict[str, Any] | None:
    checkpoint = output / CHECKPOINT_DIR
    if not (checkpoint / "state.json").exists():
        return None
    if (output / "run.manifest.json").exists():
        raise FileExistsError(f"{output} already finished; nothing to resume")
    saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if saved != json.loads(json.dumps(config_to_dict(config))):
        raise ValueError("refusing to resume: the run directory was trained with a different config")
    return {"path": checkpoint, "state": json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))}


def _restore(checkpoint: Path, lm, encoder, optimiser, device) -> None:
    import torch
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(lm, torch.load(checkpoint / "lora.pt", map_location=device))
    if encoder is not None:
        encoder.load_state_dict(torch.load(checkpoint / "encoder.pt", map_location=device))
    optimiser.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device))
    rng = torch.load(checkpoint / "rng.pt", weights_only=False)
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def _tuple_state(value):
    """JSON turns random.getstate() tuples into lists; turn them back."""
    return tuple(_tuple_state(item) for item in value) if isinstance(value, list) else value


def _remove_tree(path: Path) -> None:
    import shutil

    if path.exists():
        shutil.rmtree(path)


def _score_frozen(config, root, output, lm, tokenizer, device, validation, corpus, manifest) -> dict[str, Any]:
    """Zero-shot and BM25-prompt rungs: no adapter and no training, the base model is scored as is."""
    import torch

    lm.eval()
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    evidence = build_evidence(config, root)
    wrapper = OutcomeModel(lm, tokenizer, None, config.max_prompt_tokens, device, None, evidence, config.max_evidence_tokens)
    graph = graph_from_dict(json.loads((root / config.graph_path).read_text(encoding="utf-8")))
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config_to_dict(config))
    predictions = score_examples(wrapper, validation, SubgraphBuilder(graph, config))
    metrics = evaluate([row["gold"] for row in predictions], [row["predicted"] for row in predictions])
    _write_jsonl(output / "validation-predictions.jsonl", predictions)
    _write_json(output / "validation-metrics.json", metrics)
    payload = {
        "schema_version": "1.0.0",
        "status": "trained_not_frozen",
        "experiment_id": config.experiment_id,
        "method": METHOD,
        "variant": config.variant,
        "selector": config.selector,
        "frozen_base_model": True,
        "base_model_source": config.model_source,
        "base_model_revision": config.model_revision,
        "config_sha256": _sha256_file(output / "config.json"),
        "corpus_sha256": _sha256_file(corpus),
        "training_manifest_sha256": _sha256_file(manifest),
        "graph_sha256": _sha256_file(root / config.graph_path),
        "evidence_corpus_sha256": evidence.corpus_sha256 if evidence is not None else None,
        "train_examples": 0,
        "validation_examples": len(validation),
        "best_epoch": 0,
        "history": [{"epoch": 0, "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
                     "elapsed_seconds": time.time() - started}],
        "validation_metrics": {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]},
        "subgraph_coverage": {"train": None, "validation": None},
        "parameter_report": {**_parameter_report(lm), "encoder_parameters": 0, "trainable_fraction_with_encoder": 0.0},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "test_labels_accessed": False,
    }
    _write_json(output / "run.manifest.json", payload)
    return payload


def load_base_model(config: OutcomeTrainConfig, root: Path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if torch.cuda.is_available():
        memory = torch.cuda.get_device_properties(0).total_memory
        if memory < config.min_cuda_memory_bytes:
            raise RuntimeError(f"GPU memory {memory} is below configured minimum {config.min_cuda_memory_bytes}")
    path = root / config.local_model_path
    if not path.is_dir():
        raise FileNotFoundError(f"local model weights are not staged at {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    import transformers

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"  # renamed in transformers 4.56
    kwargs: dict[str, Any] = {"local_files_only": True, dtype_key: dtype}
    if config.quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["device_map"] = {"": 0}
    lm = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not config.quantize_4bit:
        lm = lm.to(device)
    lm.config.use_cache = False
    return lm, tokenizer


def attach_lora(lm, config: OutcomeTrainConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.quantize_4bit:
        lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True,
                                             gradient_checkpointing_kwargs={"use_reentrant": False})
    lm = get_peft_model(lm, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=list(config.lora_target_modules),
    ))
    _pin_base_model(lm.peft_config, config.model_source, config.model_revision)
    return lm


def _save_checkpoint(output: Path, lm, tokenizer, encoder) -> None:
    import torch

    lm.save_pretrained(output / "adapter")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output / "adapter")
    if encoder is not None:
        torch.save(encoder.state_dict(), output / "encoder.pt")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
