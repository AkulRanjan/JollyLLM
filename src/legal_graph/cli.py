"""Local CLI for fixture checks and statute-corpus retrieval components."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .approvals import load_approval_policy
from .audit import audit_statute_ingestions, write_audit_report
from .bm25 import (
    build_bm25_index,
    corpus_sha256,
    load_bm25_index,
    read_provision_corpus,
    search_bm25,
    write_bm25_index,
)
from .fixtures import fixture_graph
from .model_preflight import (
    load_model_registry,
    preflight_model_experiment,
    result_to_dict,
)
from .models import RetrievalConfig, RetrievalRequest, Split
from .prefix import assemble_prefix
from .retrieval import retrieve_subgraph
from .statutes import ingest_statute_pdf
from .validation import validate_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Local graph-augmented legal LLM commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate-fixture", help="validate the synthetic legal graph")
    subcommands.add_parser("retrieve-fixture", help="run split-safe temporal fixture retrieval")
    subcommands.add_parser("prefix-fixture", help="validate a graph soft-token prefix assembly")

    audit_parser = subcommands.add_parser(
        "audit-statutes",
        help="write a non-destructive quality audit for local statute PDFs",
    )
    audit_parser.add_argument("pdf_paths", nargs="+", type=Path, help="local BNS, BNSS, or CrPC PDFs")
    audit_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/india_statutes/ingestion-audit.json"),
        help="local report path",
    )

    build_parser = subcommands.add_parser(
        "build-bm25",
        help="build a local BM25 index over a provenance-backed provision corpus",
    )
    build_parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/clean/india_statutes/provisions.jsonl"),
        help="local provision JSONL created by statute ingestion",
    )
    build_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes/india_statutes/bm25"),
        help="local index and manifest directory",
    )
    build_parser.add_argument("--index-version", default="bm25-india-statutes-0.1.0")
    build_parser.add_argument("--code-revision", default="local-uncommitted")
    build_parser.add_argument(
        "--approval-policy",
        type=Path,
        help="approved local policy required to mark the generated index validated",
    )

    search_parser = subcommands.add_parser(
        "search-bm25",
        help="search a previously built local statute BM25 index",
    )
    search_parser.add_argument("query", help="plain-text query")
    search_parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/indexes/india_statutes/bm25/bm25.index.json"),
        help="local BM25 index file",
    )
    search_parser.add_argument("--limit", type=int, default=5)

    model_parser = subcommands.add_parser(
        "model-preflight",
        help="check a configured local-only QLoRA experiment without downloads",
    )
    model_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/models/local-base-models.json"),
        help="local model registry JSON",
    )
    model_parser.add_argument(
        "--experiment",
        default="qwen25-7b-graphprefix-qlora-v1",
        help="configured experiment ID",
    )

    arguments = parser.parse_args()

    if arguments.command == "validate-fixture":
        report = validate_graph(fixture_graph())
        _print(
            {
                "graph_version": report.graph_version,
                "node_count": report.node_count,
                "edge_count": report.edge_count,
                "relation_counts": report.relation_counts,
                "warnings": report.warnings,
            }
        )
    elif arguments.command == "retrieve-fixture":
        bundle = retrieve_subgraph(
            fixture_graph(),
            RetrievalRequest(
                query_id="fixture:retrieve:2023",
                seed_node_ids=("case:india:sc:beta-2022",),
                node_prizes={
                    "case:india:sc:alpha-2020": 0.8,
                    "provision:india:ipc:302:v1": 1.0,
                    "court:india:supreme-court": 0.4,
                },
                as_of_date=date(2023, 1, 1),
                allowed_origin_splits=frozenset({Split.TRAIN, Split.VALIDATION}),
            ),
            RetrievalConfig(node_budget=5, expansion_hops=2),
        )
        _print(
            {
                "status": bundle.status,
                "cache_key": bundle.cache_key,
                "selected_node_ids": bundle.selected_node_ids,
                "selected_edge_ids": bundle.selected_edge_ids,
                "diagnostics": bundle.diagnostics,
            }
        )
    elif arguments.command == "prefix-fixture":
        assembly = assemble_prefix(
            graph_soft_tokens=((0.1, 0.2, 0.3), (0.3, 0.2, 0.1)),
            instruction_embeddings=((0.5, 0.5, 0.5),),
            document_embeddings=((0.9, 0.8, 0.7), (0.6, 0.5, 0.4)),
            document_labels=(42, 43),
        )
        _print(
            {
                "sequence_length": len(assembly.embeddings),
                "attention_mask": assembly.attention_mask,
                "labels": assembly.labels,
                "graph_token_count": assembly.graph_token_count,
            }
        )
    elif arguments.command == "audit-statutes":
        ingestions = tuple(ingest_statute_pdf(path) for path in arguments.pdf_paths)
        report = audit_statute_ingestions(ingestions)
        output_path, output_sha256 = write_audit_report(report, arguments.output)
        _print(
            {
                "status": report["status"],
                "finding_count": report["finding_count"],
                "report_path": str(output_path),
                "report_sha256": output_sha256,
                "report_content_sha256": report["content_sha256"],
            }
        )
    elif arguments.command == "build-bm25":
        documents = read_provision_corpus(arguments.corpus)
        index = build_bm25_index(
            documents,
            corpus_digest=corpus_sha256(arguments.corpus),
            index_version=arguments.index_version,
        )
        approval_policy = (
            load_approval_policy(arguments.approval_policy) if arguments.approval_policy else None
        )
        index_path, manifest_path, manifest = write_bm25_index(
            index,
            arguments.output_dir,
            code_revision=arguments.code_revision,
            status="validated" if approval_policy else "provisional",
            approval_policy_sha256=approval_policy.policy_sha256 if approval_policy else None,
        )
        _print(
            {
                "index_path": str(index_path),
                "manifest_path": str(manifest_path),
                "status": manifest.status,
                "approval_policy_id": approval_policy.policy_id if approval_policy else None,
                "document_count": len(index.documents),
                "term_count": len(index.postings),
                "corpus_sha256": index.corpus_sha256,
                "index_content_sha256": manifest.content_sha256,
            }
        )
    elif arguments.command == "model-preflight":
        registry = load_model_registry(arguments.registry)
        result = preflight_model_experiment(
            registry,
            experiment_id=arguments.experiment,
            workspace_root=Path.cwd(),
        )
        _print(result_to_dict(result))
    else:
        index = load_bm25_index(arguments.index)
        results = search_bm25(index, arguments.query, limit=arguments.limit)
        _print(
            {
                "index_version": index.index_version,
                "corpus_sha256": index.corpus_sha256,
                "query": arguments.query,
                "result_count": len(results),
                "results": [
                    {
                        "node_id": result.node_id,
                        "score": result.score,
                        "metadata": dict(sorted(result.metadata.items())),
                    }
                    for result in results
                ],
            }
        )


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
