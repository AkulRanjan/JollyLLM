# Project execution plan

## 1. Purpose and completion criteria

This plan executes the graph-augmented legal LLM project described in the project document and its overview and architecture diagrams. The primary claim to test is narrow and falsifiable:

> At a matched trainable-parameter and compute budget, do query-conditioned legal-graph soft tokens improve legal-document understanding over text-only PEFT?

The project is complete when the team has released, or can reproducibly regenerate:

- a provenance-recorded heterogeneous legal graph covering a primary jurisdiction and one transfer jurisdiction;
- a validated extraction set of at least 500 manually adjudicated references, with precision and recall by jurisdiction and reference type;
- a working retrieval, R-GCN, projector, and QLoRA training path that keeps the total trainable parameters below 2% of the selected base model;
- matched baselines, ablations, three-seed result tables, paired-bootstrap tests, and an error analysis;
- a citation-faithfulness evaluation that reports validity and expert-adjudicated relevance separately; and
- a reproducible code, data-manifest, configuration, model-adapter, and documentation release.

No stage should be called complete merely because it runs. Each exit criterion below is a required decision gate.

## 2. Operating principles

1. **Start with one tractable primary slice.** Use Indian case law and statutes as the implementation spine because it supports the target transfer experiment and exposes citation, statute, and temporal relations. Add US or EU sources only after the primary graph passes validation. Contracts remain a later task-family expansion, not a dependency of the first end-to-end result.
2. **Freeze comparisons before results are visible.** Fix task splits, baselines, model budget, decoding settings, metrics, and test protocol in versioned experiment specifications. Do not alter them to improve a headline number.
3. **Treat graph quality as a measured upstream dependency.** Preserve raw citations and unresolved references. Never silently discard records to improve precision.
4. **Separate offline and online work.** Corpus ingestion, canonicalisation, graph building, node embeddings, FAISS indices, and split-safe graph snapshots are cached, versioned offline artifacts. Retrieval, encoding, projection, and inference are online paths.
5. **Make leakage impossible by construction.** Test-document edges, labels, future authorities, and target answers must be excluded from each training graph snapshot. Temporal filtering is applied before retrieval, not after generation.
6. **Use a fail-fast vertical slice.** Before scaling corpora or running a full model grid, prove the complete path on a small licensed corpus: ingest -> extract -> graph -> retrieve -> encode -> prefix -> train -> evaluate.

## 3. Initial scope and decisions to lock in week 1

| Decision | Default | Why it is the default | Change control |
| --- | --- | --- | --- |
| Primary graph domain | Indian Supreme Court cases plus Indian statutory provisions | Supports ILDC/IL-TUR and the US/EU-to-India transfer objective | Requires a design-record update and a new graph version |
| First task | Statute/precedent retrieval, then citation-grounded classification | Directly exercises graph edges before harder generation | Add tasks only after retrieval criteria pass |
| First base model | Qwen 2.5 7B in 4-bit QLoRA | Practical 7B baseline; keep Llama, Mistral, and SaulLM as controlled variants | Model swaps require a separate experiment family |
| Graph encoder | Two-layer R-GCN | Lowest-risk relation-aware implementation | HGT is a planned ablation/variant, not an untracked replacement |
| Retrieval | Dense seed retrieval + 2-hop expansion + PCST pruning | Implements the prescribed connected, bounded subgraph | Record candidate, scorer, and PCST version in every run |
| Defaults | `N=80`, `K=5`, LoRA `r=32`, alpha `64`, dropout `0.05` | Document defaults for a first reproducible run | Sweep values only through named configs |
| Experiment tracker | MLflow or Weights & Biases, selected before first run | Every seed/config/data/graph combination must be queryable | Export a tracker-independent JSON/CSV summary |

### Week-1 definition of done

- Repository layout, Python environment lock, formatter/linter, tests, secret policy, and CI are present.
- A source registry states licence, acquisition date, source URL/identifier, permitted use, checksums, and redistribution decision for every candidate corpus.
- A `data-use` owner signs off on the first corpus set; sources with unclear terms are excluded.
- A short architecture decision record (ADR) locks the first vertical slice and the exact benchmark split.
- The team can run a synthetic end-to-end example locally without downloading restricted data.

## 4. Workstreams and ownership model

| Workstream | Primary responsibility | Key outputs | Depends on |
| --- | --- | --- | --- |
| W1: governance and data operations | Data steward | source registry, licences, manifests, redaction policy | none |
| W2: document processing and graph | Graph/data engineer | canonical documents, extraction records, graph snapshot, validation set | W1 |
| W3: retrieval and indexing | Retrieval engineer | node embeddings, FAISS index, PCST selector, latency and recall report | W2 |
| W4: model fusion and training | ML engineer | GNN/projector/QLoRA modules, checkpoints, training configs | W2, W3 |
| W5: evaluation and annotation | Evaluation lead | baseline harness, metrics, adjudication guide, results tables | W1, W2, W4 |
| W6: reproducibility and release | Release owner | CI, reproducibility audit, model/data cards, release bundle | all |

One person may fill multiple roles in a small team, but W1 approval and W5 evaluation review should remain independent from model-training decisions where possible.

## 5. Delivery roadmap

The 26-week proposal is retained, with explicit evidence and stop/go decisions. Weeks indicate sequencing, not permission to skip exit criteria.

### Phase 0 - foundations and vertical slice (weeks 1-3)

**Activities**

- Create the repository skeleton and configuration conventions described in `architecture.md`.
- Register candidate data sources and choose the primary licensed slice.
- Define canonical identifiers for documents, provisions, courts, paragraphs, source records, and graph edges.
- Create a tiny fixture corpus with known citations, one amendment, one overruling/distinguishing relation, and a held-out example.
- Implement a single command that builds a fixture graph, retrieves a connected subgraph, emits `K` projected vectors, and performs one QLoRA training/evaluation step.
- Write annotation guidance and double-annotate a pilot of 25 references to uncover ambiguous resolution rules.

**Exit evidence**

- Fixture tests pass, including temporal and split filtering.
- Every generated artifact has a source version and deterministic content hash.
- Memory estimate confirms the selected base-model sequence length and effective batch size are feasible on the target hardware.

**Decision**: continue only if the full interface between graph and LLM works on the fixture. Do not begin full crawling or hyperparameter sweeps before this path exists.

### Phase 1 - corpus ingestion, preprocessing, and provenance (weeks 1-5)

**Activities**

- Acquire approved source snapshots and store immutable raw files outside Git.
- Normalise encoding, document IDs, headings, paragraph boundaries, dates, reporter/provision notation, and language metadata.
- Deduplicate at document and near-duplicate paragraph level while retaining provenance links to duplicates.
- Split source documents by the official benchmark split where available. For generated splits, use document-level and temporal boundaries; record the split algorithm and seed.
- Produce per-source manifests and data-quality reports: missing dates, malformed documents, licence status, duplication rate, and text-length distributions.

**Exit evidence**

- `manifest.jsonl` validates against its schema and contains a checksum for each raw and cleaned shard.
- A leakage audit shows no held-out labels or prohibited test-edge origins in the training snapshot.
- The release owner can recreate a cleaned shard from a pinned raw input and configuration.

### Phase 2 - extraction, linking, graph assembly, and validation (weeks 4-8)

**Activities**

- Implement jurisdiction-specific regex extractors for citations, provisions, sections, acts, amendments, and cross-references.
- Add a trainable sequence labeller only for residual citation formats; retain which method produced each candidate.
- Normalise references to canonical IDs, record confidence, and send ambiguous/unresolved candidates to a review queue rather than dropping them.
- Segment documents into rhetorical units. Record the segmentation model/version and map every unit to source character offsets.
- Add entity resolution only when structurally necessary; minimise personal entity storage.
- Materialise typed nodes and typed directed edges: `cites`, `is_cited_by`, `overrules`, `distinguishes`, `refers_to_provision`, `contains`, `defines`, `amends`, `decided_by`, `co_occurs_with`, and `temporally_precedes`.
- Attach validity windows and decision dates. Add inverse edges only where the schema explicitly calls for them.
- Draw a stratified validation sample of at least 500 references across jurisdictions and relation/reference types. Blind-adjudicate it, measure agreement, then compute precision/recall and unresolved rate.

**Exit evidence**

- Graph schema validation passes: IDs resolve, required provenance exists, relation domains/ranges are valid, and dates are coherent.
- Extraction precision and recall are reported per slice, not just in aggregate.
- The primary jurisdiction meets the team-approved quality threshold. If it does not, fix extraction or restrict the main experiment; do not train on an unmeasured noisy graph.

**Decision gate - week 8**: drop any jurisdiction from the main table that fails the extraction-quality threshold. It may remain as a clearly labelled transfer-only or engineering dataset.

### Phase 3 - retrieval and split-safe indexing (weeks 8-11)

**Activities**

- Encode node text with LegalBERT/InLegalBERT or the locked equivalent. Persist embedding model, pooling, normalisation, and dimension.
- Build an approximate nearest-neighbour index over eligible nodes only.
- Implement query-to-seed retrieval, 2-hop expansion, temporal/split filters, edge costs, node prizes, and PCST pruning.
- Cache retrieved node IDs and subgraphs by `(graph_version, index_version, query_hash, retrieval_config)`.
- Implement BM25 and dense-passage retrieval baselines on the same document universe.
- Measure recall/nDCG, connectedness, node-budget adherence, cold/warm latency, cache-hit rate, and temporal/split-filter violations.

**Exit evidence**

- At each `N in {20,40,80,200}`, the selector returns a connected graph or a declared, diagnosable no-result state.
- Retrieval evaluation is performed on a held-out query set and compares PCST with dense passage and BM25.
- Test queries cannot retrieve edges that originate from their own held-out documents or from future authorities.

### Phase 4 - graph encoder and projector alignment (weeks 12-15)

**Activities**

- Build two-layer R-GCN with relation-specific weights; add relation/type-aware HGT only as a separately configured alternative.
- Implement query-conditioned pooling: mean, attention, and top-prize variants.
- Implement the two-layer GELU projector from `d_g` to the selected LLM `d_model`.
- Implement `L_align` (InfoNCE between graph-token representations and linked text spans) and `L_link` (citation-link prediction) with masks that prevent test leakage.
- Train the graph-side modules first on the graph snapshot and perform qualitative nearest-neighbour and link-prediction sanity checks.
- Run early ablations: remove graph tokens at inference, replace the GNN with mean-pooled node features, and replace retrieved nodes with random nodes.

**Exit evidence**

- Tensor contracts and shapes are tested for empty, small, maximum-budget, and mixed relation subgraphs.
- Alignment and link loss behave stably; graph-token-removal and random-retrieval ablations show whether the structural channel carries signal.
- Checkpoints contain only trainable graph/projector state and reference immutable graph/index versions.

**Decision gate - week 15**: if alignment fails to converge or removal shows no signal, pause full multitask training. Investigate retrieval/feature quality first; if it is sound, implement the documented intermediate-layer cross-attention variant as a controlled pivot rather than blending architectures mid-run.

### Phase 5 - joint PEFT training (weeks 16-19)

**Activities**

- Load the base LLM in 4-bit NF4, freeze its native weights, and attach LoRA/QLoRA adapters to `q,k,v,o,gate,up,down` projections.
- Assemble inputs exactly as `[K graph soft tokens] || [instruction tokens] || [document tokens]` with correct attention masks, position handling, padding, and generation labels.
- Train a documented task mixture. Weight datasets explicitly rather than accidentally by corpus size.
- Run the pre-registered sweep: LoRA rank `16/32/64`, learning rate `1e-4..3e-4`, `N=20/40/80/200`, `K=1/5/10/20`, GNN depth `1/2/3`, and specified alignment/link weights. Use staged screening on validation data before committing three test seeds.
- Record GPU type, CUDA/package environment, config hash, data/graph/index versions, random seeds, wall time, peak memory, and token counts for every run.

**Exit evidence**

- The parameter counter proves LoRA + GNN + projector are below the 2% cap and is emitted with every run.
- A resume-from-checkpoint test preserves loss trajectory within a predefined tolerance.
- A smoke inference confirms graph tokens are prepended in both train and generation paths.

**Decision gate - week 19**: freeze the baseline and ablation grid. Any late-discovered defect is fixed in a new versioned run family; it is not silently mixed with previous results.

### Phase 6 - controlled evaluation, ablation, and analysis (weeks 20-23)

**Activities**

- Run B1-B7 under matched data, budget, context policy, compute ceiling, and decoding configuration.
- Run the registered ablations: no graph tokens, mean-pooled nodes, random retrieval, untyped edges, `K`, GNN depth, no alignment loss, no link loss, and randomised node features.
- Evaluate classification/judgment tasks with macro-F1; retrieval with nDCG@k, recall@k, MRR; extraction with exact match/F1/AUPR; and summarisation with ROUGE-L/BERTScore plus specified human checks.
- Extract every generated citation, resolve it against the graph, report validity automatically, and sample valid citations stratified by task/model/result for blinded legal relevance adjudication.
- Compute mean, standard deviation, confidence intervals, and paired-bootstrap significance against B2 over exactly the same test instances.
- Produce error slices by court level, temporal period, document length, relation type, rare labels, jurisdiction, and extraction confidence.
- Execute US/EU-to-India transfer at 0, 100, 1,000, and 10,000 target examples only after the primary model is stable.

**Exit evidence**

- Each reported number links to immutable predictions, config, checkpoint, dataset, graph, and evaluator versions.
- Relevance sampling and annotation agreement are reported alongside automatic validity.
- Negative and non-significant results remain in the tables and paper draft.

### Phase 7 - reproducibility, release, and paper package (weeks 24-26)

**Activities**

- Reproduce the final primary result from a clean environment and pinned artifacts.
- Publish code, configuration examples, schemas, manifests, evaluation harness, annotation guide, adapter weights, and a release manifest. Release raw data/graph content only where licences permit.
- Prepare dataset cards, model cards, limitation statements, contamination caveats, privacy/redaction notes, and source-attribution notices.
- Finalise figures/tables, paper methods, and appendices describing rejected alternatives, graph quality, and all ablations.

**Exit evidence**

- A separate reviewer can run the documented minimal reproduction without access to developer machines.
- Release checks in `versionup.md` pass and licences/provenance are complete.
- The final artifact manifest verifies checksums and cross-links versions.

## 6. Experiment matrix and progression

Avoid the combinatorial explosion of running every model, task, and ablation at once. Progress through four tiers:

| Tier | Purpose | Data/model scope | Required outcome |
| --- | --- | --- | --- |
| T0 | interface proof | fixtures + one small public split | deterministic end-to-end smoke test |
| T1 | structural signal | one retrieval/classification task, one 7B model | graph-token removal and random-retrieval ablations establish non-trivial signal |
| T2 | primary claim | primary task suite, B1-B7, 3 seeds | significance-tested graph-vs-B2 result |
| T3 | generality | remaining task families and transfer | limitations and transfer curves, not a replacement for T2 |

For every tier, use a fixed evaluation manifest. Validation data drives selection; the held-out test set is read only once per frozen candidate and seed.

## 7. Quality, safety, and governance controls

### Data and licensing

- Store a provenance record at node, edge, document, and graph-snapshot level.
- Exclude sources with unclear rights. A source may be used internally only when its terms explicitly permit the intended use; do not assume research use is sufficient.
- Keep source text and redistributable metadata in separate storage classes. Make release permissions explicit.
- Apply source redaction conventions and minimise party/entity retention. Do not invent redaction rules that damage citation resolution without documenting the effect.

### Leakage and contamination

- Version graph snapshots per split. Training retrieval must query the train-safe snapshot; evaluation queries the evaluation-safe snapshot with no direct test-answer leakage.
- Filter nodes and edges by decision date or statutory validity date where the task makes future authority invalid.
- Maintain a contamination report for benchmarks likely included in web-scale pretraining. Caveat, partition, or exclude affected results.

### Legal and research safety

- Label all outputs as retrospective document-understanding results, never legal advice.
- Do not use the system for real-party decisions or legal-service delivery.
- Provide citation links/IDs, resolver status, and an abstention/unknown condition when citations cannot be verified.
- Review performance by court, jurisdiction, outcome class, and rare label to reveal base-rate shortcuts.

## 8. Operational risk register

| Risk | Early warning | Mitigation | Owner |
| --- | --- | --- | --- |
| Low citation resolution quality | high unresolved rate or weak validation precision | improve normalisation/rules; restrict scope; retain error logs | W2 |
| Retrieval too slow | p95 latency exceeds evaluation or inference budget | smaller candidate set, precomputed embeddings, cache, PCST profiling | W3 |
| Graph tokens ignored | removal ablation does not change predictions | inspect attention/gradient paths, test projector alignment, validate retrieval relevance | W4 |
| Apparent gains caused by extra capacity | trainable count or context differs from B2 | enforce budget/context assertion in runner | W4/W5 |
| Graph/test leakage | too-good validation, edge provenance violations | split-safe snapshots; audit query paths | W2/W5 |
| Annotation inconsistency | low agreement on relevance | calibrate guide, adjudicate disagreements, stratify reporting | W5 |
| Compute overrun | sweep queue exceeds allocation | tiered screening, fixed grid at week 19, early-stop nonviable runs | W4 |
| Licence/privacy issue | source terms unclear or PII is unnecessarily retained | exclude source or quarantine; release only permitted derivatives | W1 |

## 9. Minimum reproducible run

The repository should expose one documented command sequence, with no manual notebook steps, that:

1. Fetches or verifies a permitted small fixture dataset.
2. Builds a split-safe graph snapshot and validates its schema.
3. Creates node embeddings and a retrieval index.
4. Trains a small graph-prefix QLoRA run from a versioned config.
5. Runs text-only and graph-augmented evaluation.
6. Writes predictions, citation-resolution results, metrics, environment capture, and an artifact manifest.

This is the release gate and the first thing a new contributor should run.

## 10. Status cadence

- **Daily during active implementation:** build/test health, data-pipeline failures, blocked licences, and GPU queue status.
- **Weekly:** workstream demo against an exit criterion; publish a short evidence log with artifact versions.
- **At each phase gate:** review the required evidence with the data, ML, and evaluation owners; record continue/pivot/restrict decisions in an ADR.
- **Before any external claim:** reproduce the number from immutable artifacts and check the relevant baseline, seed, significance, licence, and caveat fields.

## 11. Source alignment

This execution plan implements the supplied project specification's five stages, 26-week sequence, relation and node schemas, matched-budget baselines, validation commitments, and decision points. The SVG diagrams are treated as the logical architecture; this plan adds the implementation sequencing, test gates, reproducibility contracts, and release controls needed to execute it.
