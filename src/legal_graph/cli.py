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
from .judgments import (
    SC2016_SOURCE_ID,
    build_judgment_graph,
    ingest_sc2016_directory,
    load_source_record,
    write_extraction_report,
    write_judgment_corpus,
    write_training_manifest,
)
from .manifests import create_graph_manifest, graph_from_dict, sha256_payload, write_graph_and_manifest
from .outcomes import load_config as load_outcome_config, run as run_outcome_baseline
from .model_preflight import (
    load_model_registry,
    preflight_model_experiment,
    result_to_dict,
)
from .models import RetrievalConfig, RetrievalRequest, Split
from .prefix import assemble_prefix
from .projection import load_config as load_projection_config, run as run_projection
from .retrieval import retrieve_subgraph
from .statutes import ingest_statute_pdf
from .training import load_config as load_dapt_config, run as run_dapt
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

    judgment_parser = subcommands.add_parser(
        "ingest-sc2016",
        help="ingest the approved local Supreme Court 2016 judgment corpus",
    )
    judgment_parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw/judgments/sc-2016"),
        help="local SC-2016 corpus root containing extracted_jsons and extracted_mds",
    )
    judgment_parser.add_argument(
        "--source-registry",
        type=Path,
        default=Path("configs/sources/sc2016-source.json"),
        help="approved local provenance record for this source",
    )
    judgment_parser.add_argument("--source-id", default=SC2016_SOURCE_ID)
    judgment_parser.add_argument(
        "--base-graph",
        type=Path,
        default=Path("data/graphs/india_statutes/graph.json"),
        help="validated local statute graph to extend",
    )
    judgment_parser.add_argument(
        "--corpus-output",
        type=Path,
        default=Path("data/clean/judgments/sc-2016/judgments.jsonl"),
        help="local judgment JSONL output",
    )
    judgment_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/graphs/judgments/sc-2016"),
        help="local graph and manifest output directory",
    )
    judgment_parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/reports/judgments/sc-2016/extraction-report.json"),
        help="local unresolved-reference report output",
    )
    judgment_parser.add_argument("--graph-version", default="india-statutes-sc2016-0.1.0")
    judgment_parser.add_argument("--extractor", choices=("v1", "v2"), default="v1",
                                 help="citation extractor; v2 builds graph version 0.2 and later")
    judgment_parser.add_argument("--code-revision", default="local-uncommitted")
    judgment_parser.add_argument(
        "--training-manifest",
        type=Path,
        default=Path("data/clean/judgments/training.manifest.json"),
        help="local split-safe training supervision manifest",
    )

    dapt_parser = subcommands.add_parser(
        "train-dapt",
        help="run the bounded local QLoRA text-only DAPT baseline",
    )
    dapt_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/qwen25-1.5b-sc2016-dapt-smoke.json"),
        help="local QLoRA DAPT experiment configuration",
    )

    outcome_parser = subcommands.add_parser(
        "predict-outcome-baseline",
        help="create train-only held-out SC-2016 disposition predictions",
    )
    outcome_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/sc2016-outcome-baseline.json"),
        help="local pre-training outcome baseline configuration",
    )

    projection_parser = subcommands.add_parser(
        "project-outcomes",
        help="write the labelled-as-projected accuracy forecast for the planned variant ladder",
    )
    projection_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/sc2016-outcome-projection.json"),
        help="local outcome projection configuration",
    )

    train_outcome_parser = subcommands.add_parser("train-outcome", help="train a text or graph-augmented outcome model")
    train_outcome_parser.add_argument("--config", type=Path, required=True)
    freeze_parser = subcommands.add_parser("freeze-outcome", help="freeze a trained outcome run before test evaluation")
    freeze_parser.add_argument("--run", type=Path, required=True)
    evaluate_parser = subcommands.add_parser("evaluate-outcome", help="evaluate a frozen run once on the test split")
    evaluate_parser.add_argument("--run", type=Path, required=True)
    evaluate_parser.add_argument("--ablate-graph", action="store_true")
    compare_parser = subcommands.add_parser("compare-outcomes", help="paired bootstrap between two evaluated runs")
    compare_parser.add_argument("--run-a", type=Path, required=True)
    compare_parser.add_argument("--run-b", type=Path, required=True)
    compare_parser.add_argument("--metric", default="macro_f1", choices=("macro_f1", "accuracy"))
    compare_parser.add_argument("--resamples", type=int, default=10000)
    collect_parser = subcommands.add_parser("collect-outcomes", help="aggregate measured runs against controls and the forecast")
    collect_parser.add_argument("--config", type=Path, default=Path("configs/evaluation/sc2016-measured-vs-projected.json"))

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
    elif arguments.command == "ingest-sc2016":
        source = load_source_record(arguments.source_registry, arguments.source_id)
        ingestion = ingest_sc2016_directory(
            arguments.source_root,
            source_id=source.source_id,
            expected_source_sha256=source.raw_sha256,
        )
        base_payload = json.loads(arguments.base_graph.read_text(encoding="utf-8"))
        base_graph = graph_from_dict(base_payload)
        assembly = build_judgment_graph(
            base_graph,
            ingestion,
            version=arguments.graph_version,
            extractor=arguments.extractor,
        )
        corpus_path, corpus_sha256, corpus_count = write_judgment_corpus(
            ingestion,
            arguments.corpus_output,
        )
        report_path, report_sha256 = write_extraction_report(assembly, arguments.report_output)
        base_graph_sha256 = sha256_payload(base_payload)
        manifest = create_graph_manifest(
            assembly.graph,
            code_revision=arguments.code_revision,
            config_payload={
                "ingester": "sc2016-json-markdown-v1",
                "source_id": source.source_id,
                "source_sha256": ingestion.source_sha256,
                "base_graph_sha256": base_graph_sha256,
                "split_strategy": "sha256-prefix-modulo-10-v1",
                **({"extractor": "indian_regex_v2"} if arguments.extractor == "v2" else {}),
            },
            parents=(
                source.source_id,
                f"source:sha256:{ingestion.source_sha256}",
                f"graph:sha256:{base_graph_sha256}",
            ),
            storage_uri=f"local://{arguments.output_dir.as_posix()}",
            status="validated",
        )
        graph_path, manifest_path = write_graph_and_manifest(assembly.graph, manifest, arguments.output_dir)
        training_manifest_path, training_manifest_sha256 = write_training_manifest(
            ingestion,
            arguments.training_manifest,
            corpus_sha256=corpus_sha256,
            graph_content_sha256=manifest.content_sha256,
        )
        decision_dates = sum(record.decision_date is not None for record in ingestion.judgments)
        split_counts = {
            split.value: sum(record.origin_split is split for record in ingestion.judgments)
            for split in Split
            if any(record.origin_split is split for record in ingestion.judgments)
        }
        _print(
            {
                "status": manifest.status,
                "source_id": source.source_id,
                "source_sha256": ingestion.source_sha256,
                "judgment_count": corpus_count,
                "decision_date_count": decision_dates,
                "split_counts": split_counts,
                "corpus_path": str(corpus_path),
                "corpus_sha256": corpus_sha256,
                "graph_path": str(graph_path),
                "graph_manifest_path": str(manifest_path),
                "graph_content_sha256": manifest.content_sha256,
                "report_path": str(report_path),
                "report_sha256": report_sha256,
                "training_manifest_path": str(training_manifest_path),
                "training_manifest_sha256": training_manifest_sha256,
                "resolution_summary": assembly.resolution_counts,
            }
        )
    elif arguments.command == "train-dapt":
        result = run_dapt(load_dapt_config(arguments.config), Path.cwd())
        _print(result)
    elif arguments.command == "predict-outcome-baseline":
        result = run_outcome_baseline(load_outcome_config(arguments.config), Path.cwd())
        _print(result)
    elif arguments.command == "project-outcomes":
        result = run_projection(load_projection_config(arguments.config), Path.cwd())
        _print(result)
    elif arguments.command == "train-outcome":
        from .outcome_training import load_outcome_config as load_model_config, train_outcome

        _print(train_outcome(load_model_config(arguments.config), Path.cwd()))
    elif arguments.command == "freeze-outcome":
        from .outcome_eval import freeze_run

        _print(freeze_run(arguments.run))
    elif arguments.command == "evaluate-outcome":
        from .outcome_eval import evaluate_frozen

        _print(evaluate_frozen(arguments.run, Path.cwd(), ablate_graph=arguments.ablate_graph))
    elif arguments.command == "collect-outcomes":
        from .results import write_report

        _print(write_report(Path.cwd(), arguments.config))
    elif arguments.command == "compare-outcomes":
        _print(_compare_runs(arguments.run_a, arguments.run_b, arguments.metric, arguments.resamples))
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


def _compare_runs(run_a: Path, run_b: Path, metric: str, resamples: int) -> dict[str, object]:
    from .comparison import paired_bootstrap

    def rows(run: Path) -> dict[str, dict]:
        lines = (run / "test-predictions.jsonl").read_text(encoding="utf-8").splitlines()
        return {row["case_id"]: row for row in map(json.loads, lines) if row}

    first, second = rows(run_a), rows(run_b)
    if sorted(first) != sorted(second):
        raise SystemExit("runs were evaluated on different case sets")
    ids = sorted(first)
    if any(first[case]["gold"] != second[case]["gold"] for case in ids):
        raise SystemExit("runs disagree on gold labels")
    result = paired_bootstrap(
        [first[case]["gold"] for case in ids],
        [first[case]["predicted"] for case in ids],
        [second[case]["predicted"] for case in ids],
        metric=metric,
        resamples=resamples,
    )
    return {"run_a": str(run_a), "run_b": str(run_b), **result}


def _print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
