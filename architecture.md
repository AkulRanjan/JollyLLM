# Technical architecture plan

## 1. Architecture intent

The system injects retrieved legal structure into a frozen open-source LLM through a small prefix of continuous graph soft tokens. Its key property is that graph size is bounded before the LLM sees it: the model receives `K` projected vectors rather than a linearised edge list whose token cost grows with the subgraph.

The architecture has five stages:

```text
licensed legal corpora
  -> offline parsing, linking, provenance, graph snapshots
  -> online query-conditioned subgraph retrieval
  -> relation-aware GNN and graph-to-LLM projection
  -> frozen 4-bit LLM + trainable QLoRA, with a graph-token prefix
  -> task predictions, verifiable citations, and evaluation
```

Offline artifacts are immutable and content-addressed. Online components must state exactly which immutable graph/index/model versions they used. This division prevents accidental graph rebuilds, gives reproducible retrieval, and makes leakage audits feasible.

## 2. Context, boundaries, and non-goals

### In scope

- A heterogeneous graph over legal cases, rhetorical units, statutory provisions, contract clauses, concepts, courts, parties where necessary, and date anchors.
- Typed, directed legal relations including citation authority, statutory references, containment, amendment, temporal ordering, and legally asymmetric treatment of precedent.
- Query-conditioned connected-subgraph retrieval, graph encoding, soft-token injection, parameter-efficient adaptation, and benchmark evaluation.
- Citation validity checked against the graph and citation relevance checked through a separately versioned annotation workflow.

### Explicitly out of scope

- Legal advice, real-party decision-making, production case management, or claims of deployment-ready calibration.
- A general-purpose web crawler that ignores licences or a graph released without source provenance.
- Full fine-tuning of the selected LLM or changing its core transformer architecture in the primary experiment.
- Prompt-only graph serialisation as the production architecture; it is a required baseline.

## 3. Logical component architecture

| Component | Responsibility | Input | Output | State/contract |
| --- | --- | --- | --- | --- |
| Source registry | records permitted sources and rights | source metadata | source record | immutable source ID, licence and terms evidence |
| Ingestion/normalisation | creates canonical legal documents | raw files | normalised documents | source checksum, offsets, date, language, split eligibility |
| Extract/link | finds and resolves legal references | documents | extraction candidates and resolutions | method, confidence, raw span, canonical target or unresolved reason |
| Graph builder | materialises the heterograph | canonical records | graph snapshot | schema version, provenance, split/temporal policy |
| Feature builder | computes node features | graph snapshot + text | feature snapshot | embedding model/revision, dimensions, pooling |
| Index builder | enables seed lookup | feature snapshot | ANN index | index params, eligible-node policy |
| Retriever | selects connected bounded `G_q` | query + graph/index | subgraph bundle | retrieval config, temporal/split filters, cache key |
| Graph encoder | builds contextual node representations | subgraph bundle | `H in R^(N x d_g)` | GNN/config/checkpoint version |
| Pooler/projector | makes graph soft tokens | `H`, query state | `Z in R^(K x d_model)` | pooling and projector parameters |
| PEFT LLM | predicts/generates with graph prefix | `Z`, instruction, document | logits/text/hidden states | base-model revision and adapter checkpoint |
| Citation verifier | resolves output citations | generated output + graph snapshot | validity results | parser/version, matched target, failure reason |
| Evaluator | computes comparable task metrics | predictions + gold data | metrics/report | evaluator config and annotation-set version |

## 4. Data architecture

### 4.1 Canonical identifiers

Stable identifiers must never be derived from a file path or database row number. Use namespace-qualified IDs such as:

```text
source:{publisher}:{collection}:{source_key}
doc:{jurisdiction}:{court_or_source}:{canonical_key}
unit:{doc_id}:para:{ordinal}:{text_hash_prefix}
provision:{jurisdiction}:{act_id}:{version_id}:{section_path}
court:{jurisdiction}:{canonical_name}
edge:{relation}:{source_node_id}:{target_node_id}:{evidence_hash}
```

The exact canonicalisation policy belongs in a versioned schema document. Human-readable labels can change; canonical IDs do not. A replacement source record points to its predecessor rather than reusing an ID.

### 4.2 Core records

`DocumentRecord`

| Field | Requirement |
| --- | --- |
| `document_id`, `source_id`, `source_locator`, `raw_checksum` | required provenance and reproducibility fields |
| `jurisdiction`, `court`, `language`, `document_type` | required for filtering and stratified evaluation |
| `decision_date`, `effective_from`, `effective_to` | nullable only when source lacks the value; never silently imputed |
| `text`, `normalisation_version`, `paragraph_map` | preserve a mapping to original offsets |
| `split_membership` | training/validation/test or benchmark-native split, with assignment method |
| `licence_status`, `redistribution_status` | prevents release mistakes |

`ExtractionRecord`

| Field | Requirement |
| --- | --- |
| `extraction_id`, `document_id`, `source_unit_id`, `span_start`, `span_end`, `raw_text` | evidence for every extraction |
| `candidate_relation`, `candidate_target_type`, `method`, `method_version`, `confidence` | distinguish rules, models, and manual review |
| `resolved_target_id` or `unresolved_reason` | one must be present |
| `review_status`, `reviewer_ids`, `adjudication_id` | required for validation records |

`NodeRecord`

| Field | Requirement |
| --- | --- |
| `node_id`, `node_type`, `label`, `source_document_ids` | identity and provenance |
| `text_ref` / `text_checksum`, `feature_ref` | pointers instead of duplicate text where practical |
| `valid_from`, `valid_to`, `decision_date` | applies to temporal filtering |
| `attributes` | schema-validated type-specific data, e.g. act ID or rhetorical role |

`EdgeRecord`

| Field | Requirement |
| --- | --- |
| `edge_id`, `relation_type`, `source_id`, `target_id`, `directed` | identity and schema validation |
| `evidence_extraction_ids`, `confidence`, `method_version` | provenance and quality reporting |
| `valid_from`, `valid_to`, `source_decision_date` | temporal checks |
| `origin_split`, `eligible_for_train_graph` | leakage control |

### 4.3 Graph schema

Nodes:

| Node type | Principal features | Notes |
| --- | --- | --- |
| `case` | headnote/ratio embedding, court, date | text may be represented by linked rhetorical units |
| `rhetorical_unit` | sentence embedding, role label, offsets | roles include facts, argument, precedent analysis, ratio, etc. |
| `provision` | text embedding, act ID, version/effective window | amendments create linked versions, not overwrites |
| `contract_clause` | clause text embedding, clause taxonomy | later expansion |
| `legal_concept` | controlled vocabulary or derived concept embedding | record vocabulary/source |
| `court` | learned embedding + hierarchy attributes | supports authority analysis |
| `party` | minimal linked representation | retain only when structurally necessary |
| `date_anchor` | temporal encoding | may represent decision/effective dates |

Relations are typed and directed:

| Relation | Domain -> range | Required controls |
| --- | --- | --- |
| `cites`, `is_cited_by` | case -> case | materialise inverse intentionally; preserve evidence span |
| `overrules`, `distinguishes` | case -> case | asymmetric; validate date ordering where possible |
| `refers_to_provision` | case/clause -> provision | version must be temporally eligible |
| `contains` | act -> provision, document -> unit | acyclic within a container; preserve ordinal |
| `defines` | provision/clause -> concept | preserve source text |
| `amends` | provision -> provision | requires validity windows |
| `decided_by` | case -> court | court ID must resolve |
| `co_occurs_with` | provision <-> provision | weighted; calculations versioned |
| `temporally_precedes` | case -> case | cannot violate source dates |

Schema validation must reject an illegal domain/range, an unknown target, an unprovenanced edge, or an edge violating a strict temporal rule. Records failing validation go to a quarantine report, never into a silent `dropna` path.

### 4.4 Split and temporal isolation

The graph is a potential side channel. Build at least these immutable variants for every data release:

```text
graph/full                 all permitted source material; never used blindly for reporting
graph/train_safe/{split}   excludes held-out document-origin edges and forbidden labels
graph/eval_safe/{split}    applies the task's test and temporal eligibility policy
graph/release              only nodes/edges/text permitted for redistribution
```

`graph_version` alone is insufficient: every retrieval request also records `split_policy_id`, `cutoff_date`, and `allowed_origin_splits`. The evaluator asserts that these fields match the run manifest.

## 5. Offline pipeline

```text
raw source snapshot
  -> source registry + checksum
  -> normalised DocumentRecords
  -> segmenter and extraction candidates
  -> canonical linking / unresolved queue
  -> NodeRecords + EdgeRecords
  -> schema, licence, split, and temporal validation
  -> immutable graph snapshot
  -> node feature snapshot
  -> ANN index and retrieval cache seed
```

### Required pipeline properties

- **Idempotency:** same input snapshot + config must yield the same logical records and content hashes.
- **Incrementality:** new source data creates a new snapshot; it never mutates an old graph in place.
- **Traceability:** graph edge -> extraction -> document span -> raw source can be traversed in reverse.
- **Observability:** emit counts by node/edge/relation/source/split, resolution rate, confidence distribution, invalid records, duplication rate, and runtime.
- **Privacy:** separate personally identifying raw text from redistributable graph metadata; limit entity extraction.

## 6. Online retrieval architecture

### 6.1 Query contract

The task adapter produces a `QueryRequest`:

```json
{
  "query_id": "sha256:...",
  "task_id": "statute_retrieval",
  "instruction": "...",
  "document_ref": "doc:...",
  "query_text_hash": "sha256:...",
  "as_of_date": "2025-01-01",
  "graph_version": "graph-1.2.0",
  "index_version": "index-1.2.0-legalbert-r3",
  "split_policy_id": "eval-safe-v1",
  "retrieval_config_id": "pcst-n80-k2-v1"
}
```

Text may be carried in memory or a secured store, but logging should retain hashes/identifiers unless approved source rules allow text retention.

### 6.2 Retrieval algorithm

1. Embed the query using the pinned query/node embedding policy.
2. Search the eligible ANN index for seed nodes. Eligibility enforces graph version, jurisdiction/task scope, split policy, source rights, and temporal cutoff.
3. Expand seeds by up to two hops over allowed relation types.
4. Score candidate nodes with query relevance prizes and edges with configurable costs; costs may encode relation reliability and direction but must be defined before evaluation.
5. Solve PCST, or use a deterministic documented approximation, under the node budget `N`.
6. Validate connectedness, maximum node count, and eligibility. If no valid connected result exists, return an explicit empty/partial status with diagnostics.
7. Cache the resulting ordered node IDs and edge IDs under the full retrieval cache key.

### 6.3 Subgraph bundle

```text
SubgraphBundle
  graph_version, feature_version, index_version, retrieval_config_id
  query_id, cache_status, retrieval_ms
  node_ids[N'], node_types[N'], feature_matrix[N', d_in]
  edge_index[2, E'], edge_types[E'], edge_weights[E']
  node_prizes[N'], source_evidence_refs
  temporal/split eligibility summary
```

The bundle preserves a deterministic node order. A re-run with a cache miss must yield the same order or report a changed dependency version. `N'` may be zero; all downstream modules must handle that state without producing malformed LLM input.

## 7. Graph encoding and projection

### 7.1 Primary encoder

Use a two-layer R-GCN as the primary controlled implementation. Each layer consumes typed edges, per-type normalisation, and optional node-type parameters:

```text
h_v^(l+1) = sigma(W_0^(l) h_v^(l)
                    + sum_r sum_{u in N_r(v)} 1/c_(v,r) * W_r^(l) h_u^(l))
```

The primary model must expose relation masks so that the untyped-edge ablation can use exactly the same node set, depth, feature width, and parameter-budget accounting. HGT is a named alternative configuration with its own baseline and checkpoint lineage; do not substitute it after selection without a new experiment family.

### 7.2 Pooling and soft-token projection

1. Produce `H` with shape `[N', d_g]`.
2. Pool to exactly `K` structural vectors using a configured mechanism: query-conditioned attention (default), mean, or top-prize selection. For `N' < K`, use a documented learned/null vector and an explicit mask rather than duplicate random nodes.
3. Apply a two-layer MLP with GELU: `d_g -> d_hidden -> d_model`.
4. Apply normalisation compatible with the selected LLM embedding distribution; validate scale/variance against ordinary input embeddings.
5. Return `Z` with shape `[K, d_model]` and a graph-token mask.

`L_align` is an InfoNCE objective between pooled/projected graph state and the source text spans represented by the retrieved nodes. `L_link` is relation-aware link prediction performed only over edges permitted by the split policy. The loss configuration is explicit:

```text
L = L_task + alpha * L_align + beta * L_link
```

Default `alpha=0.1` and `beta=0.05`; both are sweeped and ablated.

## 8. LLM fusion and training architecture

### 8.1 Prefix assembly

For each batch example:

```text
inputs_embeds = concat(
  graph_soft_tokens[K, d_model],
  instruction_token_embeddings[I, d_model],
  document_token_embeddings[T, d_model]
)
attention_mask = concat(graph_mask[K], instruction_mask[I], document_mask[T])
labels = ignore_for_graph_and_instruction + task_labels
```

Position IDs and generation cache setup must make the graph prefix visible during both teacher-forced training and autoregressive inference. The implementation must test left/right padding, an empty subgraph, `K=1`, `K>1`, maximum sequence length, and mixed batches.

### 8.2 Trainable boundary

| Module | Status | Control |
| --- | --- | --- |
| base LLM weights and embedding table | frozen, 4-bit NF4 | assert `requires_grad=False` |
| LoRA adapter weights | trainable | targets and rank in config |
| GNN encoder | trainable | checkpointed separately and jointly |
| pooler/projector | trainable | checkpointed separately and jointly |
| task formatting/generation heads | task-dependent | declare whether part of PEFT budget |

Before the first optimizer step, emit a machine-readable parameter report. CI or the runner must fail if the trainable count exceeds the 2% experiment cap or if a frozen base parameter receives a gradient.

### 8.3 Configurable task adapters

Tasks should share retrieval and fusion but not pretend they share labels:

| Task family | Adapter/output | Primary metric | Graph evidence expected |
| --- | --- | --- | --- |
| judgment/classification | constrained label generation or classifier head | macro-F1 | cited authorities, provisions, court context |
| precedent/statute retrieval | ranked IDs | nDCG@k | citation/provision path |
| clause extraction | span or structured JSON under schema validation | exact match/token F1 | contains/defines/cross-reference edges |
| grounded summarisation | generation with citation IDs | ROUGE-L + citation validity/relevance | cited provisions/cases and temporal validity |

Prompts, label normalisation, parser versions, decoding parameters, and maximum context must be identical for the graph and comparison variants unless the ablation intentionally measures one of them.

## 9. Evaluation and observability architecture

### 9.1 Run manifest

Each run writes an immutable manifest before training and completes it after evaluation:

```text
run_id, git_commit, code_dirty_state, environment_lock_hash
task/dataset/split versions, source/data manifest hashes
graph/feature/index/retrieval versions and temporal policy
base-model revision, quantisation config, adapter/GNN/projector config
parameter count, seed, hardware, precision, duration, peak memory
prompt/decoder version, prediction artifact hash, evaluator version
metrics, confidence intervals, significance-comparison IDs
```

### 9.2 Citation verifier

The verifier pipeline is deliberately independent from generator training code:

```text
generated output -> citation span parser -> normaliser -> canonical graph resolver
                 -> validity result -> relevance sampling/adjudication link
```

Validity means a cited reference resolves to a real, eligible graph node. Relevance means a qualified annotator judges the resolved citation pertinent to the proposition it is used to support. Reports must never collapse the two measures.

### 9.3 Telemetry and failure modes

Record: extraction confidence, retrieval hit/empty rate, node/edge counts, PCST solve time, cache hit rate, token-prefix norm, loss components, gradient norms, GPU memory, decode latency, resolver failures, and abstentions. Sample and retain enough query/subgraph provenance to debug failures while meeting source privacy constraints.

## 10. Repository and storage layout

```text
.
├── configs/             # schema-validated data, graph, retrieval, model, eval configs
├── docs/adr/            # decision records and approved scope changes
├── schemas/             # JSON/Arrow/Pydantic contracts and graph relation registry
├── src/
│   ├── ingest/          # source registry, normalisation, segmentation
│   ├── extract/         # rules, taggers, resolution, review queue
│   ├── graph/           # graph builder, validators, split/temporal snapshots
│   ├── features/        # node embeddings and feature snapshots
│   ├── retrieval/       # ANN, expansion, PCST, cache
│   ├── models/          # R-GCN/HGT, pooling, projector, PEFT integration
│   ├── tasks/           # dataset adapters, prompts, parsers
│   ├── evaluate/        # metrics, citation verifier, bootstrap tests
│   └── common/          # IDs, provenance, manifests, logging
├── tests/               # unit, integration, contract, leakage, and reproducibility tests
├── scripts/             # thin deterministic CLI entry points
├── data/                # manifests and small fixtures only; no large/raw restricted data
├── artifacts/           # ignored pointers/cache; remote store is authoritative
├── execution.md
├── architecture.md
└── versionup.md
```

Large raw data, graph binaries, embedding matrices, FAISS indices, checkpoints, and predictions belong in a versioned artifact store rather than Git. Git stores manifests, configuration, metadata, small fixtures, and content identifiers.

## 11. Required test suite

| Test class | Examples | Gate |
| --- | --- | --- |
| schema/unit | relation domain/range, canonical IDs, validity-window checks | every change |
| extraction | known citation fixtures, unresolved logging, span offsets | every extractor change |
| graph integration | graph build is deterministic, provenance path resolves | graph-release candidate |
| leakage | test-origin edges/labels and future authorities are unreachable | every split/evaluator change |
| retrieval | budget, connectedness, determinism, cache-key completeness | every retrieval change |
| model contract | tensor shapes, empty graph, masks, prefix/label alignment | every model change |
| PEFT boundary | frozen base has no gradients; budget cap holds | every training run |
| evaluation | metric fixtures, citation validity/relevance separation, bootstrap determinism | every evaluator change |
| reproduction | clean environment repeats the minimal run | release candidate |

## 12. Architecture decision log

The following decisions should be made explicit as ADRs before implementation starts:

1. Primary legal corpus/jurisdiction and exact permitted sources.
2. Canonical document/provision/citation identifier policy.
3. Graph storage format and feature/index artifact format.
4. Split-safe and time-safe graph snapshot rules for each benchmark.
5. Embedding model and vector dimension used for node features and seed retrieval.
6. PCST library/approximation and deterministic tie-breaking policy.
7. R-GCN primary configuration, pooling strategy, and null-graph handling.
8. First base model, prompt/label interface, and LoRA target modules.
9. Citation parser/resolver policy and relevance annotation protocol.
10. Artifact store, tracker, and the release-reproduction environment.

Any decision that changes the meaning of a graph, retrieval result, model input, metric, or reported baseline must create a new version according to `versionup.md`; do not overwrite an old artifact or result.
