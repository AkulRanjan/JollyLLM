# Version-control, data, model, and release plan

## 1. Purpose

This project has more than source code to reproduce. A reported result depends on legal-source terms, source snapshots, document normalisation, graph extraction and resolution, graph topology, node features, retrieval index, task split, base model revision, adapters, inference settings, annotations, and evaluator code. This plan makes each dependency identifiable, immutable, and traceable.

The guiding rule is:

> A result is reproducible only when another person can identify the exact code, data, graph, retrieval, model, prompt, and evaluator versions that created it.

## 2. Versioned asset classes

| Asset class | Examples | System of record | Version rule |
| --- | --- | --- | --- |
| Source code | pipeline, model code, schemas, tests, docs | Git | commit SHA plus semantic release tag |
| Configurations | YAML/JSON task, graph, retrieval, training configs | Git | content hash + Git commit; immutable once used in a run |
| Source registry | licences, source URLs/IDs, acquisition date, checksums | Git metadata + restricted evidence store | new version for a source/rights change |
| Raw data | crawls, official downloads, annotation source files | access-controlled artifact store | content hash, source snapshot ID; never Git |
| Cleaned data | normalised docs, splits, dedup reports | versioned data store | semantic data version + manifest hash |
| Graphs | nodes, edges, provenance, validation results | versioned data/artifact store | graph semantic version; append/change never overwrites |
| Features/indexes | embedding matrix, FAISS index, PCST candidate cache | artifact store | derived version tied to graph + encoder + config |
| Checkpoints/adapters | GNN, projector, LoRA weights | model/artifact registry | model semantic version tied to all inputs |
| Predictions/results | test predictions, metrics, bootstrap outputs | immutable artifact store | run ID plus evaluator version |
| Human annotations | validation labels, relevance judgments, adjudications | access-controlled data store | annotation-set version with guideline version |
| Release package | code tag, manifests, cards, permitted artifacts | release registry | public release tag and signed manifest |

Git is not a data lake. Do not commit large checkpoints, corpora, embeddings, vector indices, or generated prediction shards to Git. Commit only small fixtures, schemas, manifests, configs, and references to immutable external artifacts.

## 3. Semantic versioning policy

Use `MAJOR.MINOR.PATCH` versions for independently consumable data, graph, index, evaluator, and model artifacts. A release manifest also records the immutable content hash, so semantic labels remain human-friendly rather than being the sole identity.

| Change type | Version bump | Examples |
| --- | --- | --- |
| Incompatible meaning or interface | MAJOR | node/edge schema change; canonical-ID policy change; changed split policy; rewritten citation-validity definition; incompatible model input contract |
| Backward-compatible new content/capability | MINOR | add licensed source data; add a relation/node attribute; add a model task adapter; add a new benchmark without changing old splits |
| Corrective/reproducible fix | PATCH | fix a normaliser bug; correct metadata; repair a serialisation defect; rerun derived index from unchanged graph |
| Experimental/non-final | prerelease suffix | `1.3.0-rc.1`, `1.3.0-exp.pcst2` |

### Special rules

- A data correction that changes train/test membership, citation targets, feature text, or graph topology is at least a MINOR graph/data bump and invalidates derived indexes and model comparability unless rebuilt.
- A metric-definition change is a MAJOR evaluator bump. Historical reports retain their original evaluator version and are never rewritten in place.
- A base-model revision, tokenizer revision, prompt-template change, decoder policy change, or LoRA target change starts a new model experiment family. It may be a new MINOR model version but must not be pooled with an older family without explicit analysis.
- A changed random seed alone does not create a new code/data/model version; it creates a new run under the same frozen experiment specification.

## 4. Git workflow

### Branches

| Branch | Use | Merge requirement |
| --- | --- | --- |
| `main` | protected, releasable code and docs | review, CI, no unresolved conflicts |
| `feature/<area>-<short-name>` | isolated feature work | pull request into `main` |
| `fix/<area>-<issue>` | scoped corrective work | regression test and issue reference |
| `release/<version>` | final release preparation only | all release gates passed; tag from this branch or `main` |
| `experiment/<family>` | optional code changes before they are proven | cannot publish results without merging/reviewing a pinned commit |

Avoid long-lived personal branches with unreviewed evaluator or schema changes. A notebook is exploratory evidence, not the authoritative implementation; promote stable logic into tested source code and a versioned CLI/config before relying on its output.

### Commits and pull requests

- Use focused, imperative conventional-style messages: `feat(graph): add split-safe edge filter`, `fix(resolver): retain unresolved provision references`, `docs: define citation relevance protocol`.
- One pull request should state the user-visible/data-visible impact, contract changes, validation run, expected version bump, and whether old artifacts are invalidated.
- Require review for changes touching licences, source registry, schemas, split rules, evaluator logic, release manifests, or public documentation.
- CI runs format/lint/type checks, unit tests, graph-schema fixtures, leakage tests, model-contract tests where feasible, and documentation-link checks.
- Do not amend a commit after it has been cited by a run manifest. Create a new commit; historical runs must still resolve to their original source state.

### Tags

Create annotated, signed tags where available:

```text
code/v1.2.0             source release
data/india-v1.1.0       cleaned data and split manifest
graph/legal-v1.2.0      graph schema/content release
index/legal-v1.2.0      retrieval index derived release
model/qwen7b-graph-v1.0.0
eval/v1.1.0
release/v1.0.0          externally released research bundle
```

Tags point to the source commit; the corresponding artifact registry entry stores its content hash, URI, manifest, and release state.

## 5. Data and graph lifecycle

### 5.1 Manifests

Every raw, cleaned, graph, feature, index, annotation, model, prediction, and release artifact has a machine-readable manifest. At minimum:

```json
{
  "artifact_type": "graph_snapshot",
  "artifact_version": "1.2.0",
  "content_sha256": "...",
  "created_at": "2026-09-05T00:00:00Z",
  "code_commit": "...",
  "config_sha256": "...",
  "parents": ["data/india-v1.1.0"],
  "schema_version": "2.0.0",
  "split_policy_id": "train-safe-v1",
  "temporal_policy_id": "as-of-decision-v1",
  "licence_summary_ref": "...",
  "storage_uri": "...",
  "status": "validated"
}
```

Manifests must use a stable canonical JSON form before hashing. `parents` captures derivation, not merely a documentation link.

### 5.2 Immutability and corrections

- An artifact marked `validated`, `reported`, or `released` is immutable.
- A correction creates a successor version with a `supersedes` pointer and an explanation of impact: record count, graph edges, benchmark membership, affected runs, and whether a report must be reissued.
- Preserve old restricted artifacts for audit according to the data-retention policy, even when a successor is published. Public availability may be withdrawn if rights require it, but the internal manifest/history remains.
- Never replace a FAISS index, checkpoint, predictions file, or metrics table at the same URI.

### 5.3 Graph derivation chain

```text
source registry + raw snapshots
  -> cleaned corpus + split manifest
  -> extraction/resolution records + manual validation set
  -> graph snapshot
  -> node feature snapshot
  -> eligible-node ANN index / PCST configuration
  -> retrieved-subgraph cache
  -> graph-prefix model checkpoint
  -> predictions, citations, metrics, report
```

Each arrow is a parent-child dependency in the registry. Rebuild downstream artifacts when any upstream semantic input changes. A code-only PATCH change that reproduces bitwise-identical output may retain the same semantic data version but should still produce a new execution manifest.

## 6. Experiment identity and reproducibility

### 6.1 Experiment specification

Before training, commit and freeze an `ExperimentSpec` containing:

```text
experiment family and hypothesis
Git commit and clean/dirty state
dataset/task/split versions and data-manifest hashes
graph/feature/index versions, retrieval config, temporal/split policy
base model and tokenizer revision; quantisation setup
prompt template, output parser, decoder settings, maximum context
GNN/pooler/projector architecture and loss weights
LoRA targets/rank/alpha/dropout; all trainable parameter counts
optimizer, schedule, batch/accumulation, sequence policy, seeds
hardware, software lock, and planned metrics/baseline comparison
```

The runner computes an `experiment_spec_hash`. An experiment run is identified by:

```text
run_id = experiment_spec_hash + seed + attempt_number
```

Changing any listed semantic setting creates a new specification hash. Retries due to infrastructure failure increment `attempt_number` and preserve logs; they do not overwrite prior evidence.

### 6.2 Required run outputs

- Initial and final run manifests, structured logs, stdout/stderr, and environment capture.
- Parameter-accounting report and base-freeze assertion.
- Resumable checkpoint references, including optimiser/scheduler state where permitted.
- Validation metrics and exactly which checkpoint-selection rule was used.
- Immutable test predictions with instance IDs, outputs, parsed citations, retrieval trace IDs, and errors/abstentions.
- Evaluator output: metrics, per-slice metrics, bootstrap samples or deterministic bootstrap inputs, and comparison IDs.

### 6.3 Reproduction levels

| Level | Meaning | Release expectation |
| --- | --- | --- |
| L0: interface | fixtures execute and contracts hold | required on every PR |
| L1: pipeline | data -> graph -> index can be regenerated from permitted inputs | required for graph/data release |
| L2: inference | a pinned checkpoint regenerates predictions within documented numeric tolerance | required for model release |
| L3: training | a clean setup retrains a representative configuration within expected seed variation | required for final research release |

Exact bitwise training equivalence is not a requirement across GPU hardware. Define deterministic settings where practical and document expected tolerance/seed variation. Inference and evaluator outputs should be deterministic under pinned decoding and environments.

## 7. Evaluation and annotation versioning

### Evaluator

Version the evaluator separately because its definitions decide the paper's claims. The evaluator release specifies:

- task input parser and label normaliser;
- metric library and its exact configuration;
- citation extraction/normalisation/resolution code;
- validity eligibility rules, including time and split policies;
- relevance sampling scheme, strata, random seed, and aggregation;
- bootstrap implementation, sample count, seed, confidence level, and paired comparison logic.

### Annotation sets

Store annotation guidelines, training examples, annotator qualifications/roles, calibration records, blind assignment, individual labels where privacy permits, adjudication decisions, and inter-annotator agreement. An annotation-set version changes when labels, guidelines, eligibility, aggregation, or sampling change.

The results table must name both `evaluator_version` and `annotation_set_version`. Citation validity and citation relevance remain separately versioned columns; do not publish a combined ambiguous “faithfulness” score without retaining both constituents.

## 8. Compatibility policy

| Consumer | Compatibility promise | Breaking changes |
| --- | --- | --- |
| pipeline modules | schema-based input/output contracts | renamed fields, changed ID semantics, new required fields |
| retrieval -> GNN | stable `SubgraphBundle` shape/type/mask semantics within a major version | changed node order, edge encoding, feature dimension, mask meaning |
| projector -> LLM | `K x d_model` prefix interface tied to tokenizer/model family | model hidden size, position policy, prefix mask/placement changes |
| checkpoint loader | loads same-major graph/projector/adapter state with parent artifacts | schema/model architecture/LoRA target changes |
| evaluation reports | retains metric column meaning within evaluator major version | changed definitions, slices, eligibility, or comparison protocol |

Use migration scripts for schema conversions and test them against frozen fixtures. A migration emits a new artifact manifest; it must not rewrite old storage in place.

## 9. Release procedure

### 9.1 Release candidate checklist

- [ ] Code is committed, reviewed, and tagged; working tree is clean.
- [ ] Dependency lock/environment image is pinned and vulnerability/licence review is complete.
- [ ] Source registry, provenance, permissions, redaction status, and redistribution decisions are complete.
- [ ] Graph validation, extraction-quality report, split-leakage audit, and temporal-policy checks passed.
- [ ] Artifact manifests resolve every parent and every referenced URI/checksum.
- [ ] Model parameter budget, frozen-base assertion, and graph-prefix contract passed.
- [ ] All baselines/ablations use frozen configs and comparable data/context/decoding policies.
- [ ] Three-seed aggregates, bootstrap tests, negative results, and caveats are included.
- [ ] Citation validity and relevance are reported separately with evaluator/annotation versions.
- [ ] Minimal reproduction passed from a clean environment.
- [ ] Dataset/model cards and “not legal advice” limitations are present.
- [ ] Public bundle contains only assets permitted for redistribution.

### 9.2 Release manifest

Publish one top-level manifest linking code tag, all asset versions, checksums, licences, model cards, data cards, documentation, paper tables, and known limitations. Sign the manifest if the project has signing infrastructure. The manifest is the citation target for a release; individual object-store locations may change only through a versioned mirror record.

### 9.3 Rollback and retraction

- **Code defect before release:** fix on `fix/*`, rerun impacted tests, issue a PATCH release candidate.
- **Incorrect reported metric or evaluator defect:** publish a corrected evaluator version and an erratum; preserve original predictions and state affected claims.
- **Graph/source error:** freeze use of the affected graph, create successor data/graph/index/model artifacts, enumerate affected runs, and rerun required comparisons.
- **Licence or privacy issue:** immediately revoke public access to the affected artifact where possible, preserve internal audit records, notify downstream release users, and publish a replacement without the source.

Never delete the historical release entry merely to conceal an error. Mark it withdrawn/superseded and link the correction.

## 10. Repository controls and automation

Required automation should include:

```text
pre-commit: formatting, lint, secret scan, manifest/schema validation
pull-request CI: unit + contract + leakage fixture tests, docs link checks
nightly: small end-to-end fixture run, dependency/security checks
graph-release CI: deterministic build check, provenance/temporal/split audit
model-run preflight: clean Git state, parameter cap, dependency and parent-artifact validation
release CI: manifests resolve, reproduction test, licence allowlist, checksums, tag verification
```

Protect secrets, private source locators, and restricted raw-text credentials through the platform secret manager. Logs and experiment trackers must not upload raw restricted documents by default. Store a scrubbed retrieval trace for observability and enable fuller traces only under the relevant source permissions.

## 11. Naming conventions

Use descriptive, sortable identifiers:

```text
data-india-statutes-v1.1.0
graph-india-train-safe-v1.2.0
features-legalbert-r3-graph1.2.0
index-faiss-hnsw-pcst-n80-v1.2.0
exp-qwen25-7b-graphprefix-retrieval-v1
run-<spec-hash>-seed-17-attempt-1
model-qwen25-7b-graphprefix-v1.0.0
eval-citation-faithfulness-v1.1.0
annotations-citation-relevance-v1.0.0
```

Names help people; manifests and hashes establish identity. Do not encode credentials, personal names, source text, or mutable dates in identifiers.

## 12. Decisions required before the first public experiment

1. Choose Git hosting/protection rules and an artifact/data/model registry.
2. Choose the environment lock and container strategy for CUDA-dependent runs.
3. Approve the source registry and licence allowlist.
4. Approve graph schema version `1.0.0`, canonical-ID policy, and split/temporal isolation rules.
5. Approve primary benchmark data versions and the citation relevance annotation guide.
6. Define retention/access policies for restricted source text, annotations, and run logs.
7. Assign release authority and the reviewers required for data, schema, evaluator, and public-release changes.

Once these are set, every implementation task in `execution.md` can create artifacts without ambiguity, and every component in `architecture.md` can identify its compatible inputs.
