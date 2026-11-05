"""Efficiency benchmark for the statute-graph prefix against text-based augmentation (CPU only).

    python scripts/benchmark_efficiency.py --records <dir with records-2013/2014/2015.jsonl.gz> \
        --statutes <dir with ipc.json iea.json constitution.json> --hf <dir with qwen3-0.6b and qwen3-8b/config.json> \
        --out <results.json>

Measures, with the project's own code where it exists:
  1. graph index build time, memory and disk (time-safe training window 2013-2014);
  2. per-query latency of the co-citation selector against BM25 statute retrieval, on the 2015 development views;
  3. the graph encoder at its real size: parameters, bytes on disk, per-query latency;
  4. prompt cost of one augmentation (8 soft tokens, BM25 evidence text, or the subgraph written out as text):
     token counts, analytic prefill FLOPs and KV-cache bytes for Qwen3-8B, and measured CPU prefill on Qwen3-0.6B.

Only years 2013-2015 are read. No model is scored, and 2017, 2018 and 2019 are not opened.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import pickle
import statistics
import sys
import time
import tracemalloc
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_graph.aws_judgments import official_label, read_aws_records  # noqa: E402
from legal_graph.bm25 import Bm25Document  # noqa: E402
from legal_graph.builder import build_reference_edges  # noqa: E402
from legal_graph.extraction import extract_indian_references_v2, resolve_candidates  # noqa: E402
from legal_graph.graph_encoder import GraphPrefixEncoder  # noqa: E402
from legal_graph.models import GraphSnapshot, NodeRecord, NodeType, RetrievalConfig, Split  # noqa: E402
from legal_graph.outcome_evidence import Bm25Evidence  # noqa: E402
from legal_graph.outcomes import build_input_view_v2, infer_disposition  # noqa: E402
from legal_graph.query_graph import NUM_RELATION_TYPES, build_query_subgraph  # noqa: E402
from legal_graph.statutes import ingest_statute_json, spec_for, write_provision_corpus  # noqa: E402

TRAIN_YEARS, DEV_YEAR, BASE_CHARS, NODE_BUDGET = (2013, 2014), 2015, 6000, 32
VIEW_TOKENS, GRAPH_TOKENS, BM25_TOKENS, NODE_TEXT_TOKENS = 1536, 8, 384, 48
NODE_DIM, HIDDEN, MODEL_DIM = 4096, 256, 4096


def summary(values_ms: list[float]) -> dict[str, float]:
    ordered = sorted(values_ms)
    return {
        "n": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
        "max_ms": round(ordered[-1], 3),
    }


def timed(function, *args, **kwargs):
    start = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - start) / 1e6


def load_cases(records_dir: Path):
    seen, cases = set(), []
    for year in (*TRAIN_YEARS, DEV_YEAR):
        for record in read_aws_records(records_dir / f"records-{year}.jsonl.gz"):
            if str(record.get("decision_date") or "")[:4] != str(year) or record["path"].lower() in seen:
                continue
            seen.add(record["path"].lower())
            text, label = str(record.get("text") or ""), official_label(str(record.get("disposal_nature", "")))
            if not text.strip() or label is None:
                continue
            inferred = infer_disposition(text[int(len(text) * 0.8):])
            view = build_input_view_v2(text, BASE_CHARS)
            if inferred != label or view is None:
                continue
            try:
                decided = date.fromisoformat(str(record["decision_date"])[:10])
            except ValueError:
                decided = date(year, 1, 1)
            cases.append({"path": record["path"], "year": year, "text": text, "view": view, "date": decided})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--statutes", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    args = parser.parse_args()
    results: dict[str, object] = {"environment": {
        "torch": torch.__version__, "threads": torch.get_num_threads(), "cpu_count": __import__("os").cpu_count(),
        "note": "CPU only, no GPU; timings are single-process wall clock after warm-up."}}

    # ---- statute corpus (IPC, IEA, Constitution only: BNS, BNSS and CrPC need PDFs that are not stored locally)
    registry = json.loads((ROOT / "configs/sources/statutes-v2-sources.json").read_text(encoding="utf-8"))
    ingestions = []
    for act_id, source in registry["json_sources"].items():
        ingestions.append(ingest_statute_json(
            args.statutes / f"{act_id}.json", spec_for(act_id), number_field=source["number_field"],
            title_field=source["title_field"], text_field=source["text_field"], expected_sha256=source["sha256"]))
    corpus_path = args.out.parent / "provisions-3acts.jsonl"
    _, corpus_hash, corpus_count = write_provision_corpus(ingestions, corpus_path)
    corpus_docs = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results["statute_corpus"] = {"acts": list(registry["json_sources"]), "provisions": corpus_count, "sha256": corpus_hash,
                                 "note": "the paper's layer has 2,619 provisions over six acts; this subset is smaller"}

    # ---- cases
    cases = load_cases(args.records)
    train = [c for c in cases if c["year"] in TRAIN_YEARS]
    dev = [c for c in cases if c["year"] == DEV_YEAR]
    print(f"train {len(train)} dev {len(dev)}", flush=True)
    results["cases"] = {"train_window": list(TRAIN_YEARS), "train_cases": len(train), "query_year": DEV_YEAR, "queries": len(dev)}

    # ---- 1. graph build ------------------------------------------------------------------------------
    corpus_ids = [doc["node_id"] for doc in corpus_docs]
    node_text = {doc["node_id"]: doc["text"] for doc in corpus_docs}

    def provision_hints(text: str, document_id: str):
        return [c for c in extract_indian_references_v2(text, document_id) if c.canonical_hint.startswith("provision:")]

    start = time.perf_counter()
    tracemalloc.start()
    train_candidates = {c["path"]: provision_hints(c["text"], c["path"]) for c in train}
    dev_seed_candidates = {c["path"]: provision_hints(c["view"], c["path"]) for c in dev}
    hints = {cand.canonical_hint for group in (*train_candidates.values(), *dev_seed_candidates.values()) for cand in group}
    provision_ids = list(corpus_ids)
    for hint in sorted(hints):
        if not any(node_id.startswith(hint) for node_id in corpus_ids):
            provision_ids.append(hint + "base")  # an act whose text is not stored locally: node without text
    nodes = [NodeRecord(node_id, NodeType.PROVISION, node_id.rsplit(":", 2)[-2] if ":" in node_id else node_id, ())
             for node_id in provision_ids]
    nodes += [NodeRecord("case:" + c["path"], NodeType.CASE, c["path"], (c["path"],), decision_date=c["date"]) for c in train]
    edges = []
    provision_set = tuple(provision_ids)
    for c in train:
        resolutions = resolve_candidates(train_candidates[c["path"]], provision_set)
        edges.extend(build_reference_edges("case:" + c["path"], resolutions, origin_split=Split.TRAIN,
                                           source_decision_date=c["date"]).edges)
    graph = GraphSnapshot("bench", "1", tuple(nodes), tuple(edges))
    build_seconds = time.perf_counter() - start
    _, build_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    pickled = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
    edge_rows = "\n".join(f"{e.source_id}\t{e.target_id}\t{e.source_decision_date}" for e in edges).encode()
    edge_gz = gzip.compress(edge_rows)
    from scipy.sparse import csr_matrix, save_npz
    col = {pid: i for i, pid in enumerate(provision_ids)}
    case_row = {c["path"]: i for i, c in enumerate(train)}
    rows = [case_row[e.source_id[5:]] for e in edges if e.target_id in col]
    cols = [col[e.target_id] for e in edges if e.target_id in col]
    incidence = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(train), len(provision_ids)))
    buffer = io.BytesIO()
    save_npz(buffer, incidence, compressed=True)
    results["graph_index"] = {
        "nodes": len(nodes), "provision_nodes": len(provision_ids), "case_nodes": len(train), "edges": len(edges),
        "build_seconds_including_extraction": round(build_seconds, 2), "python_peak_mb": round(build_peak / 1e6, 1),
        "pickle_mb": round(len(pickled) / 1e6, 3), "edge_list_gzip_kb": round(len(edge_gz) / 1e3, 1),
        "csr_incidence_npz_kb": round(len(buffer.getvalue()) / 1e3, 1),
        "note": "built from 2013-2014 training judgments only; the paper's graph also holds 2015 and SC-2016 judgments"}
    print("graph", results["graph_index"], flush=True)

    # ---- 2. per-query retrieval latency ----------------------------------------------------------------
    as_of = date(DEV_YEAR, 1, 1)
    config = RetrievalConfig(node_budget=NODE_BUDGET)

    def graph_query(case):
        return build_query_subgraph(graph, query_id=case["path"], document_id=case["path"], view_text=case["view"],
                                    as_of_date=as_of, exclude_node_ids=frozenset(), config=config,
                                    selector="cocitation", extractor="v2")

    for case in dev[:3]:
        graph_query(case)
    graph_ms, subgraphs = [], []
    for case in dev:
        sub, ms = timed(graph_query, case)
        graph_ms.append(ms)
        subgraphs.append(sub)
    non_empty = [s for s in subgraphs if s.node_ids]
    non_empty_ms = [ms for ms, s in zip(graph_ms, subgraphs) if s.node_ids]
    results["graph_retrieval"] = {
        **summary(graph_ms), "non_empty_only": summary(non_empty_ms), "non_empty_subgraphs": len(non_empty),
        "mean_nodes_non_empty": round(statistics.fmean(len(s.node_ids) for s in non_empty), 1) if non_empty else 0,
        "mean_directed_edges_non_empty": round(statistics.fmean(len(s.edges) for s in non_empty), 1) if non_empty else 0}
    print("graph retrieval", results["graph_retrieval"], flush=True)

    tracemalloc.start()
    bm25, bm25_build_ms = timed(Bm25Evidence, corpus_path, 4)
    _, bm25_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    index_pickle = pickle.dumps(bm25.index, protocol=pickle.HIGHEST_PROTOCOL)
    example = lambda case: SimpleNamespace(view_text=case["view"], as_of=as_of)  # noqa: E731
    for case in dev[:3]:
        bm25(example(case))
    bm25_ms, evidence_texts = [], []
    for case in dev:
        text, ms = timed(bm25, example(case))
        bm25_ms.append(ms)
        evidence_texts.append(text)
    results["bm25_retrieval"] = {
        **summary(bm25_ms), "index_build_ms": round(bm25_build_ms, 1), "index_python_peak_mb": round(bm25_peak / 1e6, 2),
        "index_pickle_mb": round(len(index_pickle) / 1e6, 3), "indexed_provisions": corpus_count,
        "note": "corpus of three acts; per-query BM25 time grows with corpus size, so this is a lower bound"}
    print("bm25", results["bm25_retrieval"], flush=True)

    # ---- 3. encoder at its real size ---------------------------------------------------------------------
    torch.manual_seed(0)
    encoder = GraphPrefixEncoder(NODE_DIM, HIDDEN, MODEL_DIM, GRAPH_TOKENS, NUM_RELATION_TYPES, typed=True).eval()
    parameters = sum(p.numel() for p in encoder.parameters())
    state = io.BytesIO()
    torch.save(encoder.state_dict(), state)
    query_vector = torch.randn(MODEL_DIM)
    latencies = {}
    for threads in (1, torch.get_num_threads()):
        torch.set_num_threads(threads)
        samples = []
        with torch.inference_mode():
            for sub in non_empty[:150]:
                x = torch.randn(len(sub.node_ids), NODE_DIM)
                index = torch.tensor([[s for s, _, _ in sub.edges], [t for _, t, _ in sub.edges]], dtype=torch.long) \
                    if sub.edges else torch.zeros(2, 0, dtype=torch.long)
                types = torch.tensor([r for _, _, r in sub.edges], dtype=torch.long)
                encoder(x, index, types, query_vector)
                _, ms = timed(encoder, x, index, types, query_vector)
                samples.append(ms)
            empty = [timed(encoder, torch.zeros(0, NODE_DIM), torch.zeros(2, 0, dtype=torch.long),
                           torch.zeros(0, dtype=torch.long), query_vector)[1] for _ in range(50)]
        latencies[f"threads_{threads}"] = {"non_empty": summary(samples), "null_tokens": summary(empty)}
    torch.set_num_threads(int(results["environment"]["threads"]))
    feature_table_nodes = 2619
    results["encoder"] = {
        "parameters": parameters, "paper_reported_parameters": 6340097,
        "state_dict_fp32_mb": round(len(state.getvalue()) / 1e6, 2), "state_dict_fp16_mb": round(parameters * 2 / 1e6, 2),
        "node_feature_table_fp32_mb_for_2619_nodes": round(feature_table_nodes * NODE_DIM * 4 / 1e6, 1),
        "node_feature_table_fp16_mb_for_2619_nodes": round(feature_table_nodes * NODE_DIM * 2 / 1e6, 1),
        "latency": latencies}
    print("encoder", {k: v for k, v in results["encoder"].items() if k != "latency"}, flush=True)

    # ---- 4. prompt cost -----------------------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf / "qwen3-0.6b")

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def label(node_id: str) -> str:
        parts = node_id.split(":")
        return f"{parts[2].upper()} {parts[3]}" if len(parts) > 3 else node_id

    bm25_tokens = [min(count(t), BM25_TOKENS) if t else 0 for t in evidence_texts]
    labels_only, labels_edges, labels_text = [], [], []
    for sub in non_empty:
        names = [label(n) for n in sub.node_ids]
        labels_only.append(count("Relevant provisions: " + "; ".join(names)))
        pairs = sorted({(min(a, b), max(a, b)) for a, b, _ in sub.edges})
        labels_edges.append(labels_only[-1] + count("\n".join(f"{names[a]} - {names[b]}" for a, b in pairs)))
        labels_text.append(sum(min(NODE_TEXT_TOKENS, count(node_text[n])) if n in node_text else NODE_TEXT_TOKENS
                               for n in sub.node_ids))
    mean = lambda values: round(statistics.fmean(values), 1) if values else 0  # noqa: E731
    token_rows = {
        "text_only": {"added_tokens": 0},
        "graph_soft_prefix": {"added_tokens": GRAPH_TOKENS},
        "bm25_evidence_text": {"added_tokens_mean": mean(bm25_tokens), "added_tokens_cap": BM25_TOKENS},
        "subgraph_as_labels_text": {"added_tokens_mean_non_empty": mean(labels_only)},
        "subgraph_as_labels_and_edges_text": {"added_tokens_mean_non_empty": mean(labels_edges)},
        "subgraph_as_node_texts_48_tokens": {"added_tokens_mean_non_empty": mean(labels_text)}}

    config8 = json.loads((args.hf / "qwen3-8b" / "config.json").read_text())
    layers, hidden, heads, kv_heads = (config8["num_hidden_layers"], config8["hidden_size"],
                                       config8["num_attention_heads"], config8["num_key_value_heads"])
    head_dim, inter, vocab = config8.get("head_dim", hidden // heads), config8["intermediate_size"], config8["vocab_size"]
    per_layer = hidden * head_dim * (heads + 2 * kv_heads) + heads * head_dim * hidden + 3 * hidden * inter
    body_params = layers * per_layer
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 2

    def prefill_flops(length: int) -> float:
        return 2 * body_params * length + 4 * layers * heads * head_dim * length * length / 2

    base_length = VIEW_TOKENS
    analytic = {}
    for name, row in token_rows.items():
        added = row.get("added_tokens", row.get("added_tokens_mean", row.get("added_tokens_mean_non_empty", 0)))
        length = base_length + added
        analytic[name] = {"sequence_tokens": round(length, 1), "prefill_tflops": round(prefill_flops(length) / 1e12, 3),
                          "relative_flops": round(prefill_flops(length) / prefill_flops(base_length), 4),
                          "kv_cache_mb_fp16": round(length * kv_bytes_per_token / 1e6, 1)}
    # the BM25 prompt keeps the same token budget: evidence displaces view text instead of extending the sequence
    analytic["bm25_evidence_same_budget"] = {
        "sequence_tokens": base_length, "prefill_tflops": round(prefill_flops(base_length) / 1e12, 3),
        "relative_flops": 1.0, "kv_cache_mb_fp16": round(base_length * kv_bytes_per_token / 1e6, 1),
        "note": "evidence replaces up to 384 view tokens; the view the model reads is shorter by that amount"}
    results["prompt_cost"] = {
        "tokens": token_rows, "qwen3_8b": {"layers": layers, "hidden": hidden, "kv_heads": kv_heads, "head_dim": head_dim,
                                           "kv_bytes_per_token_fp16": kv_bytes_per_token,
                                           "non_embedding_params_billions": round(body_params / 1e9, 2)},
        "analytic_qwen3_8b": analytic}
    print("tokens", token_rows, flush=True)

    # measured prefill on the small model of the same family (CPU, float32)
    model = AutoModelForCausalLM.from_pretrained(args.hf / "qwen3-0.6b", dtype=torch.float32).eval()
    embed = model.get_input_embeddings()
    proxy_encoder = GraphPrefixEncoder(NODE_DIM, HIDDEN, model.config.hidden_size, GRAPH_TOKENS, NUM_RELATION_TYPES, True).eval()
    generator = torch.Generator().manual_seed(1)

    def prefill(length: int, prefix: torch.Tensor | None = None) -> float:
        ids = torch.randint(1000, 50000, (1, length - (0 if prefix is None else prefix.shape[0])), generator=generator)
        with torch.inference_mode():
            embeddings = embed(ids)
            if prefix is not None:
                embeddings = torch.cat([prefix.unsqueeze(0), embeddings], dim=1)
            model(inputs_embeds=embeddings, use_cache=True)  # warm-up
            samples = []
            for _ in range(args.prefill_repeats):
                _, ms = timed(model, inputs_embeds=embeddings, use_cache=True)
                samples.append(ms)
        return statistics.median(samples)

    prefix = torch.randn(GRAPH_TOKENS, model.config.hidden_size) * 0.02
    scenarios = {
        "text_only": (VIEW_TOKENS, None),
        "graph_soft_prefix": (VIEW_TOKENS + GRAPH_TOKENS, prefix),
        "bm25_evidence_same_budget": (VIEW_TOKENS, None),
        "subgraph_as_labels_text": (VIEW_TOKENS + int(mean(labels_only)), None),
        "subgraph_as_node_texts_48_tokens": (VIEW_TOKENS + int(mean(labels_text)), None)}
    measured = {}
    for name, (length, pre) in scenarios.items():
        measured[name] = {"sequence_tokens": length, "median_ms": round(prefill(length, pre), 1)}
        print(name, measured[name], flush=True)
    base_ms = measured["text_only"]["median_ms"]
    for row in measured.values():
        row["relative_to_text_only"] = round(row["median_ms"] / base_ms, 3)
    results["prefill_proxy_qwen3_0_6b_cpu"] = {
        "rows": measured, "repeats": args.prefill_repeats,
        "note": "a 0.6B model on CPU stands in for Qwen3-8B on a T4: it shows relative cost, not absolute latency"}
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
