"""Controlled CPU prefill timing for the prompt variants of benchmark_efficiency.py.

The first version of this measurement ran each variant once in sequence, and two rows with identical length and no prefix differed
by a factor of 2.2, so it was discarded. This version interleaves the variants over several rounds, includes a duplicate of the
text-only row as a noise control, and reports the minimum and the median over rounds.

    python scripts/benchmark_prefill.py --hf <dir with qwen3-0.6b> --results <results.json> --threads 8 --rounds 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_graph.graph_encoder import GraphPrefixEncoder  # noqa: E402
from legal_graph.query_graph import NUM_RELATION_TYPES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    tokens = results["prompt_cost"]["tokens"]
    base = 1536
    labels = int(tokens["subgraph_as_labels_text"]["added_tokens_mean_non_empty"])
    node_texts = int(tokens["subgraph_as_node_texts_48_tokens"]["added_tokens_mean_non_empty"])

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.hf / "qwen3-0.6b", dtype=torch.float32).eval()
    embed = model.get_input_embeddings()
    width = model.config.hidden_size
    generator = torch.Generator().manual_seed(1)
    prefix = torch.randn(8, width, generator=generator) * 0.02

    def make(length: int, with_prefix: bool) -> torch.Tensor:
        ids = torch.randint(1000, 50000, (1, length - (8 if with_prefix else 0)), generator=generator)
        with torch.inference_mode():
            vectors = embed(ids)
            return torch.cat([prefix.unsqueeze(0), vectors], dim=1) if with_prefix else vectors

    scenarios = {
        "text_only": make(base, False),
        "text_only_repeat_noise_control": make(base, False),
        "graph_soft_prefix_8_tokens": make(base + 8, True),
        "subgraph_as_labels_text": make(base + labels, False),
        "subgraph_as_node_texts_48_tokens": make(base + node_texts, False),
    }
    names = list(scenarios)
    samples = {name: [] for name in names}
    with torch.inference_mode():
        for name in names:  # warm-up, not recorded
            model(inputs_embeds=scenarios[name][:, :256], use_cache=True)
        for round_index in range(args.rounds):
            order = names[round_index % len(names):] + names[:round_index % len(names)]
            for name in order:
                start = time.perf_counter()
                model(inputs_embeds=scenarios[name], use_cache=True)
                samples[name].append((time.perf_counter() - start) * 1e3)
            print("round", round_index, {n: round(samples[n][-1]) for n in names}, flush=True)
    rows = {}
    for name in names:
        rows[name] = {"sequence_tokens": scenarios[name].shape[1], "min_ms": round(min(samples[name]), 1),
                      "median_ms": round(statistics.median(samples[name]), 1),
                      "spread_max_over_min": round(max(samples[name]) / min(samples[name]), 3)}
    floor = rows["text_only"]["min_ms"]
    for row in rows.values():
        row["min_relative_to_text_only"] = round(row["min_ms"] / floor, 3)
    results["prefill_proxy_qwen3_0_6b_cpu_controlled"] = {
        "rows": rows, "rounds": args.rounds, "threads": args.threads,
        "note": "Qwen3-0.6B float32 on CPU stands in for Qwen3-8B on a T4 and shows relative cost only; "
                "the noise-control row repeats the text-only prompt, so its distance from 1.0 bounds the measurement error"}
    results.pop("prefill_proxy_qwen3_0_6b_cpu", None)
    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["prefill_proxy_qwen3_0_6b_cpu_controlled"], indent=1))


if __name__ == "__main__":
    main()
