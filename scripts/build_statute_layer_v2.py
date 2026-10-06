"""Build statute layer 0.2: BNS, BNSS and CrPC from their India Code PDFs, plus the IPC,
Indian Evidence Act and Constitution from commit-pinned JSON transcriptions.

    py -3 scripts/build_statute_layer_v2.py

Writes data/graphs/india_statutes-v2/{graph.json, graph.manifest.json} and
data/clean/india_statutes-v2/provisions.jsonl. Layer 0.1 is left untouched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_graph.manifests import create_graph_manifest, write_graph_and_manifest  # noqa: E402
from legal_graph.statutes import (  # noqa: E402
    build_statute_graph,
    ingest_statute_json,
    ingest_statute_pdf,
    spec_for,
    write_provision_corpus,
)

REGISTRY = ROOT / "configs/sources/statutes-v2-sources.json"
GRAPH_DIR = ROOT / "data/graphs/india_statutes-v2"
CORPUS = ROOT / "data/clean/india_statutes-v2/provisions.jsonl"


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    ingestions = [ingest_statute_pdf(ROOT / path) for path in registry["pdf_sources"].values()]
    for act_id, source in registry["json_sources"].items():
        ingestions.append(ingest_statute_json(
            ROOT / source["path"], spec_for(act_id),
            number_field=source["number_field"], title_field=source["title_field"], text_field=source["text_field"],
            expected_sha256=source["sha256"],
        ))
    corpus_path, corpus_sha256, corpus_count = write_provision_corpus(ingestions, CORPUS)
    graph = build_statute_graph(ingestions, version=registry["statute_layer_version"])
    registry_sha256 = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    manifest = create_graph_manifest(
        graph,
        code_revision="local",
        config_payload={
            "ingester": "statute-pdf-v1+statute-json-v1",
            "acts": sorted(ingestion.spec.act_id for ingestion in ingestions),
            "source_hashes": sorted(ingestion.source_sha256 for ingestion in ingestions),
            "source_registry_sha256": registry_sha256,
        },
        parents=tuple(sorted(ingestion.document_id for ingestion in ingestions)) + (f"registry:sha256:{registry_sha256}",),
        storage_uri="local://data/graphs/india_statutes-v2",
        status="validated",
    )
    write_graph_and_manifest(graph, manifest, GRAPH_DIR)
    print(json.dumps({
        "graph_version": graph.version,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "provision_corpus": str(corpus_path.relative_to(ROOT)),
        "provision_corpus_sha256": corpus_sha256,
        "provisions": corpus_count,
        "per_act": {ingestion.spec.act_id: len(ingestion.provisions) for ingestion in ingestions},
        "graph_content_sha256": manifest.content_sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
