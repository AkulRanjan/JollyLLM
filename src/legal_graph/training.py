"""Bounded local QLoRA DAPT baseline for the split-safe SC-2016 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from itertools import cycle
from pathlib import Path
from typing import Any, Mapping, Sequence


class TrainingConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DaptConfig:
    experiment_id: str
    model_source: str
    model_revision: str
    local_model_path: Path
    corpus_path: Path
    training_data_manifest: Path
    output_dir: Path
    max_seq_length: int
    batch_size: int
    max_steps: int
    checkpoint_every_steps: int
    learning_rate: float
    seed: int
    max_train_documents: int
    max_validation_documents: int
    max_train_blocks: int
    max_validation_blocks: int
    min_cuda_memory_bytes: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]


def load_config(path: Path) -> DaptConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingConfigurationError(f"cannot load {path}: {error}") from error
    if data.get("schema_version") != "1.0.0" or data.get("local_files_only") is not True:
        raise TrainingConfigurationError("config must use schema 1.0.0 and local_files_only=true")
    if data.get("method") != "qlora_dapt_text_only_v1":
        raise TrainingConfigurationError("only qlora_dapt_text_only_v1 is supported")
    lora = data.get("lora")
    if not isinstance(lora, Mapping) or not isinstance(lora.get("target_modules"), list):
        raise TrainingConfigurationError("config requires lora.target_modules")
    try:
        result = DaptConfig(
            experiment_id=str(data["experiment_id"]),
            model_source=str(data["model_source"]),
            model_revision=str(data["model_revision"]),
            local_model_path=Path(str(data["local_model_path"])),
            corpus_path=Path(str(data["corpus_path"])),
            training_data_manifest=Path(str(data["training_data_manifest"])),
            output_dir=Path(str(data["output_dir"])),
            max_seq_length=int(data["max_seq_length"]),
            batch_size=int(data["per_device_batch_size"]),
            max_steps=int(data["max_steps"]),
            checkpoint_every_steps=int(data["checkpoint_every_steps"]),
            learning_rate=float(data["learning_rate"]),
            seed=int(data["seed"]),
            max_train_documents=int(data["max_train_documents"]),
            max_validation_documents=int(data["max_validation_documents"]),
            max_train_blocks=int(data["max_train_blocks"]),
            max_validation_blocks=int(data["max_validation_blocks"]),
            min_cuda_memory_bytes=int(data["min_cuda_memory_bytes"]),
            lora_rank=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            lora_target_modules=tuple(str(value) for value in lora["target_modules"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingConfigurationError(f"invalid DAPT config: {error}") from error
    if min(result.max_seq_length, result.batch_size, result.max_steps, result.checkpoint_every_steps, result.max_train_documents, result.max_validation_documents, result.max_train_blocks, result.max_validation_blocks, result.lora_rank) <= 0:
        raise TrainingConfigurationError("count configuration values must be positive")
    return result


def load_documents(corpus_path: Path, manifest_path: Path, split: str, maximum: int, seed: int) -> tuple[dict[str, str], ...]:
    manifest = _manifest(manifest_path)
    split_field = {"train": "training_splits", "validation": "validation_splits"}.get(split)
    if not split_field:
        raise TrainingConfigurationError(f"unsupported split {split!r}")
    if manifest.get("corpus_sha256") != _sha256_file(corpus_path):
        raise TrainingConfigurationError("corpus checksum does not match training manifest")
    allowed = manifest.get(split_field)
    if not isinstance(allowed, list) or not all(isinstance(value, str) for value in allowed):
        raise TrainingConfigurationError(f"invalid manifest field {split_field}")
    results: list[dict[str, str]] = []
    with corpus_path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            item = json.loads(line)
            if item.get("origin_split") not in allowed:
                continue
            if not isinstance(item.get("case_id"), str) or not isinstance(item.get("text"), str) or not item["text"].strip():
                raise TrainingConfigurationError(f"invalid corpus record at line {number}")
            results.append({"case_id": item["case_id"], "text": item["text"]})
    return tuple(sorted(results, key=lambda item: _stable_order(seed, item["case_id"]))[:maximum])


def run(config: DaptConfig, workspace_root: Path) -> dict[str, object]:
    torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training = _dependencies()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; local QLoRA cannot run.")
    available_memory = torch.cuda.get_device_properties(0).total_memory
    if available_memory < config.min_cuda_memory_bytes:
        raise RuntimeError(f"GPU memory {available_memory} is below configured minimum {config.min_cuda_memory_bytes}.")
    model_path = workspace_root / config.local_model_path
    corpus_path = workspace_root / config.corpus_path
    manifest_path = workspace_root / config.training_data_manifest
    output_dir = workspace_root / config.output_dir
    if not model_path.is_dir():
        raise FileNotFoundError(f"local model weights are not staged at {model_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_docs = load_documents(corpus_path, manifest_path, "train", config.max_train_documents, config.seed)
    validation_docs = load_documents(corpus_path, manifest_path, "validation", config.max_validation_documents, config.seed)
    if not train_docs or not validation_docs:
        raise TrainingConfigurationError("train and validation samples must both be non-empty")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_blocks = _blocks(tokenizer, train_docs, config.max_seq_length, config.max_train_blocks)
    validation_blocks = _blocks(tokenizer, validation_docs, config.max_seq_length, config.max_validation_blocks)
    if not train_blocks or not validation_blocks:
        raise TrainingConfigurationError("tokenisation produced no usable blocks")
    quantisation = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=quantisation, device_map={"": 0}, torch_dtype=torch.float16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=list(config.lora_target_modules)))
    _pin_base_model(model.peft_config, config.model_source, config.model_revision)
    parameters = _parameter_report(model)
    if parameters["trainable_fraction"] > 0.02:
        raise RuntimeError("trainable parameter fraction exceeds 2% cap")
    collate = _collator(torch, tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(_dataset(torch, train_blocks), batch_size=config.batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.seed), collate_fn=collate)
    validation_loader = torch.utils.data.DataLoader(_dataset(torch, validation_blocks), batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate)
    device = torch.device("cuda:0")
    model.train()
    losses: list[float] = []
    checkpoint_steps: list[int] = []
    iterator = cycle(train_loader)
    for step in range(1, config.max_steps + 1):
        batch = {key: value.to(device) for key, value in next(iterator).items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        if step % config.checkpoint_every_steps == 0 or step == config.max_steps:
            checkpoint_dir = output_dir / f"step-{step:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_steps.append(step)
    validation_loss = _evaluate(torch, model, validation_loader, device)
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    run_manifest = {
        "schema_version": "1.0.0",
        "status": "completed",
        "experiment_id": config.experiment_id,
        "method": "qlora_dapt_text_only_v1",
        "local_files_only": True,
        "base_model_source": config.model_source,
        "base_model_revision": config.model_revision,
        "corpus_sha256": _sha256_file(corpus_path),
        "training_manifest_sha256": _sha256_file(manifest_path),
        "train_document_count": len(train_docs),
        "validation_document_count": len(validation_docs),
        "train_block_count": len(train_blocks),
        "validation_block_count": len(validation_blocks),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "validation_loss": validation_loss,
        "checkpoint_every_steps": config.checkpoint_every_steps,
        "checkpoint_steps": checkpoint_steps,
        "parameter_report": parameters,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_total_memory_bytes": available_memory,
        "torch_version": torch.__version__,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    (output_dir / "run.manifest.json").write_text(json.dumps(run_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return run_manifest


def _manifest(path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingConfigurationError(f"cannot read training manifest: {error}") from error
    if not isinstance(manifest, Mapping) or manifest.get("status") != "validated" or manifest.get("artifact_type") != "judgment_training_corpus":
        raise TrainingConfigurationError("training manifest is not a validated judgment corpus")
    held_out = manifest.get("held_out_splits")
    if not isinstance(held_out, list) or "test" not in held_out:
        raise TrainingConfigurationError("training manifest must hold out the test split")
    return manifest


def _blocks(tokenizer: Any, documents: Sequence[Mapping[str, str]], length: int, maximum: int) -> tuple[tuple[int, ...], ...]:
    blocks: list[tuple[int, ...]] = []
    for document in documents:
        tokens = tokenizer(document["text"], add_special_tokens=True, truncation=False)["input_ids"]
        if tokenizer.eos_token_id is not None and (not tokens or tokens[-1] != tokenizer.eos_token_id):
            tokens.append(tokenizer.eos_token_id)
        for start in range(0, len(tokens), length):
            block = tuple(tokens[start : start + length])
            if len(block) >= 16:
                blocks.append(block)
            if len(blocks) >= maximum:
                return tuple(blocks)
    return tuple(blocks)


def _dataset(torch: Any, blocks: Sequence[Sequence[int]]) -> Any:
    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(blocks)

        def __getitem__(self, index: int) -> Any:
            return torch.tensor(blocks[index], dtype=torch.long)

    return Dataset()


def _collator(torch: Any, pad_id: int) -> Any:
    def collate(items: Sequence[Any]) -> dict[str, Any]:
        # Mask by block length, not by token identity: tokenizers without a dedicated pad
        # token fall back to pad_token = eos_token, and comparing against pad_id would then
        # drop every real end-of-document token from both the mask and the loss.
        lengths = torch.tensor([len(item) for item in items], dtype=torch.long)
        input_ids = torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=pad_id)
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        attention_mask = positions.lt(lengths.unsqueeze(1)).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.masked_fill(attention_mask.eq(0), -100)}

    return collate


def _evaluate(torch: Any, model: Any, loader: Any, device: Any) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            values.append(float(model(**{key: value.to(device) for key, value in batch.items()}).loss.detach().cpu()))
    model.train()
    return sum(values) / len(values)


def _pin_base_model(peft_config: Mapping[str, Any], source: str, revision: str) -> None:
    """Name the pinned upstream repo in every adapter config.

    PEFT records whatever path the base model was loaded from, which on Kaggle is a
    scratch directory that exists nowhere else. Saved adapters would then fail to
    resolve their base model anywhere but the machine that trained them.
    """
    for adapter_config in peft_config.values():
        adapter_config.base_model_name_or_path = source
        adapter_config.revision = revision


def _parameter_report(model: Any) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_fraction": trainable / total if total else 0.0}


def _dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("install local torch, transformers, peft, bitsandbytes, and accelerate before training") from error
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _stable_order(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}|{case_id}".encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded local QLoRA DAPT baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/training/qwen25-1.5b-sc2016-dapt-smoke.json"))
    arguments = parser.parse_args()
    result = run(load_config(arguments.config), Path.cwd())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
