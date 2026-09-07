# LLM Graph - local implementation scaffold

This repository implements a local-first graph-augmented legal LLM research pipeline. There is no GitHub configuration, remote origin, or automatic upload of data, logs, artifacts, or model weights.

The implemented local path is:

1. ingest supplied statute PDFs into provenance-backed provision records;
2. audit duplicate, reset, and structural parser candidates without altering source PDFs;
3. materialize a validated act/provision graph and JSONL corpus;
4. build and search a deterministic BM25 baseline with a lineage manifest; and
5. check whether the planned Qwen or SaulLM QLoRA experiment is locally ready, without downloading anything.

## Local quick start

Requires Python 3.11 or later and the pypdf dependency declared in pyproject.toml.

~~~powershell
$env:PYTHONPATH = (Join-Path $PWD "src")

py -3 -m unittest discover -s tests -v
py -3 -m legal_graph.cli validate-fixture
py -3 -m legal_graph.cli retrieve-fixture
py -3 -m legal_graph.cli prefix-fixture

py -3 -m legal_graph.cli audit-statutes data/raw/india_code/bharatiya-nyaya-sanhita-bns.pdf data/raw/india_code/Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf data/raw/india_code/the_code_of_criminal_procedure,_1973.pdf

py -3 -m legal_graph.cli build-bm25 --approval-policy configs/quality/india-statutes-extraction-policy.json
py -3 -m legal_graph.cli search-bm25 "punishment for murder" --limit 5
py -3 -m legal_graph.cli model-preflight --experiment saullm-7b-graphprefix-qlora-v1
~~~

All outputs remain on this computer. Generated corpus, graph, report, index, checkpoint, and model-weight directories are intentionally ignored by local Git.

## Current safeguards

- Graph records retain node type, evidence IDs, source splits, and temporal provenance.
- Retrieval excludes forbidden splits and temporally ineligible records before selecting a connected subgraph.
- Statute extraction keeps the earliest monotonic provision sequence and excludes embedded state-amendment, schedule, appendix, and form numbering from the searchable Act corpus.
- The audit report is non-destructive and records every remaining duplicate or number-reset candidate with PDF page references.
- BM25 uses the exact corpus checksum and approved extraction-policy hash in its manifest. A rebuild without that policy is intentionally provisional.
- Model preflight requires local model weights, optional ML packages, validated input manifests, and an approved judgment-supervision manifest. It never installs packages or downloads a model.

## Qwen and SaulLM

[local-base-models.json](configs/models/local-base-models.json) tracks both planned 7B QLoRA experiment families:

- Qwen 2.5 is the primary initial model.
- SaulLM is a controlled legal-model variant that must use the same frozen graph, index, and task split.

Neither model has been downloaded or run. That is deliberate: the current statute-only corpus has no approved judgment supervision, and the readiness gate prevents an unreproducible training run.

## Layout

~~~text
src/legal_graph/     ingestion, audit, retrieval, BM25, and model-preflight modules
tests/               deterministic unit and vertical-slice tests
configs/             versioned, non-secret runtime settings
schemas/             portable record contracts
data/fixtures/       small, synthetic, redistributable provenance inputs
execution.md         roadmap and quality gates
architecture.md      architecture and data contracts
versionup.md         local versioning and release controls
~~~

See [architecture.md](architecture.md) for the planned R-GCN, PCST, QLoRA, and evaluation integrations.
