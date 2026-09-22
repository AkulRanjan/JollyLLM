# LLM Graph - local implementation scaffold

This repository implements a local-first graph-augmented legal LLM research pipeline. There is no GitHub configuration, remote origin, or automatic upload of data, logs, artifacts, or model weights.

The implemented local path is:

1. ingest supplied statute PDFs into provenance-backed provision records;
2. audit duplicate, reset, and structural parser candidates without altering source PDFs;
3. materialize a validated act/provision graph and JSONL corpus;
4. build and search a deterministic BM25 baseline with a lineage manifest; and
5. check whether the planned Qwen or SaulLM QLoRA experiment is locally ready, without downloading anything.
6. ingest the approved local 2016 Supreme Court corpus into case, court, and statute-reference graph records; and
7. forecast, before any training run, where outcome accuracy is likely to land for each planned variant, with the uncertainty that the corpus size actually supports.

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
py -3 -m legal_graph.cli ingest-sc2016
py -3 -m legal_graph.cli predict-outcome-baseline --config configs/evaluation/sc2016-outcome-baseline.json
py -3 -m legal_graph.cli project-outcomes --config configs/evaluation/sc2016-outcome-projection.json
py -3 -m legal_graph.cli train-dapt --config configs/training/qwen25-1.5b-sc2016-dapt-smoke.json
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
- The outcome projection is stamped `projected_not_measured`. It reads real label supports and the one measured baseline, derives every other figure from configured per-class recall assumptions, and reports Wilson intervals, a data-scaling curve, simulated single-run spread, and the test-set size needed to rank two variants apart. It never touches held-out labels and is not evidence about a trained system.
- Judgment ingestion verifies the approved SC-2016 source checksum, uses the full local Markdown transcript, assigns a deterministic split, retains unresolved citation evidence in a local report, and writes a split-safe training-manifest lineage file for model preflight.

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

## Figures

`py -3 scripts/render_figures.py` renders 17 analysis PNGs under `figures/` from the local artifacts only.
It writes nothing else and touches no report.

~~~text
figures/corpus/       7 measured corpus figures: splits and label supports, class imbalance, graph
                      structure, subject and bench composition, length and time, citation resolution,
                      statute retrieval pool
figures/projection/   9 forecast figures: variant ladder with Wilson intervals, accuracy against
                      macro-F1, stage attribution, data-scaling curve, per-class F1, confusion
                      matrices, single-run spread, ranking power, and a combined summary sheet
figures/diagrams/     1 pipeline schematic marking which stages are done, measured, or planned
~~~

Orange marks a measurement, blue and green mark forecasts, and every forecast figure says so in its
footer. Regenerate the projection report first if the assumptions change.

## Kaggle GPU run

`py -3 scripts/build_kaggle_notebook.py` writes `notebooks/kaggle_sc2016_dapt.ipynb`, a single notebook that runs
`train-dapt` on a Kaggle T4. It embeds the code, configs, and statute graph. At run time it fetches SC-2016 from Hugging
Face with a sparse git fetch at the registry's pinned commit and downloads the model at the config's pinned revision.
It re-runs `ingest-sc2016` and stops unless the corpus and training-manifest hashes match the local build. Nothing is
uploaded as a Kaggle dataset.

In Kaggle, create a notebook, use File → Import Notebook, and set Accelerator to GPU T4 x2 and Internet to On. Pick
`RUN` in the first cell: `qwen3-8b` is the main model ([qwen3-8b-sc2016-dapt-v1.json](configs/training/qwen3-8b-sc2016-dapt-v1.json)),
`saul-7b` is the legal-domain comparison with identical data and settings ([saul-7b-sc2016-dapt-v1.json](configs/training/saul-7b-sc2016-dapt-v1.json)),
and `smoke` is a short pipeline check. Then use Save Version → Save & Run All. Checkpoints are written straight to
`/kaggle/working`, so a run stopped by the 12-hour session limit keeps what it saved. Judgments and model weights stay on the session's temporary disk. Only adapters and text-free manifests reach
`/kaggle/working`. The generated notebook is ignored by Git because it contains a generated graph. Rebuild it after
changing code or configs.

## Accuracy outlook

`project-outcomes` writes `data/reports/evaluation/sc2016-outcome-projection-v1/outcome-projection.json`, and
`accuracy-outlook.html` beside it renders that report as a dashboard. The primary planned configuration is forecast
at roughly three quarters accuracy on the evaluable held-out cases, against a measured 46.8 percent Naive Bayes
baseline; the 95 percent interval on that forecast spans about 23 points, so the corpus cannot yet distinguish
variants that sit a few points apart. Change the assumptions in the config, never the emitted report.

See [architecture.md](architecture.md) for the planned R-GCN, PCST, QLoRA, and evaluation integrations.
