# LLM Outcome Evaluation — Design

**Date:** 2026-09-20
**Status:** approved, not yet implemented

## Goal

Produce measured, publishable outcome-prediction results for four language-model
arms on the held-out SC-2016 test split, with confidence intervals, paired
significance tests, and figures that slot into the existing figure set.

Today the repository has exactly one measured evaluation — a train-only
multinomial Naive Bayes baseline at macro-F1 0.411 (accuracy 0.468, 47 eligible
validation cases) — and one trained adapter, `qwen3-8b-sc2016-dapt-v1`. No code
path evaluates a language model on the outcome task. Every LLM number in
`configs/evaluation/sc2016-outcome-projection.json` is marked
`projected_not_measured`. This design closes that gap.

## Experiment matrix

A 2×2 factorial over base model and domain adaptation, evaluated alongside the
two baselines already measured.

| Arm | Base model | Adapter | Evidence |
|---|---|---|---|
| `majority-class-prior` | — | — | already measured |
| `nb-train-only` | — | — | already measured |
| `qwen-base` | Qwen3-8B | none | to measure |
| `qwen-dapt` | Qwen3-8B | `qwen3-8b-sc2016-dapt-v1` | to measure |
| `saul-base` | SaulLM-7B-Instruct-v1 | none | to measure |
| `saul-dapt` | SaulLM-7B-Instruct-v1 | `saul-7b-sc2016-dapt-v1` | to measure, **needs training** |

The factorial is the point. Two contrasts carry the paper:

- `qwen-base → qwen-dapt` and `saul-base → saul-dapt` isolate the **effect of
  domain-adaptive pretraining**, with the base model held constant.
- `qwen-dapt ↔ saul-dapt` and `qwen-base ↔ saul-base` isolate the **effect of
  the base model**, with adaptation held constant.

Comparing only `qwen-dapt` against `saul-base` would move both variables at once
and support neither claim.

## Non-goals

- Graph augmentation. The typed R-GCN prefix, PCST, and BM25 variants stay
  projected in this round. This design measures the text-only rungs of the
  ladder.
- Any change to the NB baseline or its committed artifacts.
- Instruction-following or generation quality. Scoring is constrained.

## Architecture

### New module: `src/legal_graph/llm_outcomes.py`

Reuses the baseline's primitives rather than reimplementing them, which is what
makes the LLM arms comparable to NB by construction rather than by assertion:

| Reused from `outcomes.py` | Guarantees |
|---|---|
| `build_input_view(text, limit)` | identical 6000-char masked prefix |
| `infer_disposition(text)` | identical gold-label derivation |
| `evaluate(gold, predicted)` | identical `sc2016-outcome-evaluator-1.0.0` metrics |
| `LABELS` | identical four-label schema |

Two refactors in `outcomes.py`, both pure renames with no behaviour change:

- `_read_cases` → public `read_cases`
- `_labelled_cases` → public `labelled_cases`

`labelled_cases` matters: it derives gold labels from `text[int(len(text)*0.8):]`
— the final 20% — while `build_input_view` exposes only the leading 6000
characters with disposition phrases masked. Gold comes from the tail, input from
the head. The LLM arms must inherit that exact split or the leakage guarantee is
void.

Public surface:

```python
@dataclass(frozen=True, slots=True)
class LlmOutcomeConfig: ...

def load_config(path: Path) -> LlmOutcomeConfig
def build_prompt(view: str) -> str
def score_labels(model, tokenizer, prompt: str, null_prompt: str) -> dict[str, float]
def run(config: LlmOutcomeConfig, workspace_root: Path) -> dict[str, object]
```

### New module: `src/legal_graph/stats.py`

Isolated so it is testable without a GPU or any model:

```python
def paired_bootstrap(per_case: Mapping[str, Sequence[bool]], resamples: int, seed: int) -> dict
def mcnemar_exact(arm_a: Sequence[bool], arm_b: Sequence[bool]) -> dict
```

### New CLI verb

`predict-outcome-llm --config configs/evaluation/<arm>.json`, following the
existing `predict-outcome-baseline` pattern in `cli.py`.

### New notebook builder: `scripts/build_kaggle_eval_notebook.py`

Same embedded-bundle pattern as `build_kaggle_notebook.py`: deterministic
tar.gz of `src/legal_graph/*.py` and `configs/**/*.json`, base64-embedded, sha256
verified on unpack. Adapters mount read-only through **Add Input → Your Work**
rather than being downloaded.

## Scoring

Constrained label scoring. No generation, no parsing, no format compliance
required — base and adapted models are treated identically.

### Prompt template

Versioned `prompt_view_id: "disposition-choice-v1"` and recorded in every
manifest. Pinned by a golden test; changing it is a version bump.

```
The following is the opening portion of a judgment of the Supreme Court of
India. Explicit disposition phrases have been masked.

{view}

Question: What was the final disposition of the matter?
Answer: The matter was {label_surface}
```

Label surfaces, recorded in the manifest:

| Label | Surface |
|---|---|
| `allowed` | `allowed` |
| `dismissed` | `dismissed` |
| `disposed` | `disposed of` |
| `partly_allowed` | `partly allowed` |

### Scoring math

For case *i* and label *ℓ* with tokens *ℓ₁..ℓₙ*:

```
s(ℓ | Pᵢ) = (1/n) Σₜ log p(ℓₜ | Pᵢ, ℓ₍<ₜ₎)          mean token log-prob
c(ℓ | Pᵢ) = s(ℓ | Pᵢ) − s(ℓ | P_null)                contextual calibration
predicted = argmax_ℓ c(ℓ | Pᵢ)
probabilities = softmax over c(· | Pᵢ)
```

`P_null` is the same template with `view = "N/A"`, scored once per arm and
cached. Mean-per-token normalisation removes length bias between `allowed` and
`partly allowed`; subtracting the null score removes the model's intrinsic
prior over label words. Both raw `s` and calibrated `c` are recorded so the
calibration's effect is auditable.

The four labels share a prompt prefix, so the prefix is encoded once and the
four continuations scored against a cached KV state — four short forward passes
per case, not four full ones.

### Output schema

Per-case rows match `test-predictions.jsonl` from the NB baseline exactly
(`case_id`, `document_id`, `title`, `origin_split`, `input_view_id`,
`input_sha256`, `predicted_label`, `confidence`, `label_probabilities`), plus
`raw_label_scores` and `calibrated_label_scores`. Existing downstream tooling
keeps working unchanged.

## Leakage protocol

1. Build and tune **entirely on validation**. Test labels are not read.
2. Freeze all four arms: configs, prompt template, calibration, code.
3. One scored test run. `held_out_test_status` flips from `not_read_for_labels`
   to `read_once_after_freeze` in the new manifests only.
4. The committed NB baseline artifacts are never rewritten.

This is a one-way door and matches the protocol already declared in
`data/reports/evaluation/sc2016-outcome-baseline-v1/baseline.manifest.json`:
"evaluate only after candidate freeze."

## Statistics

Roughly 45 of the 62 test cases will be label-eligible — validation yielded 47
of 63. At that n, a point estimate alone is not a result.

- **95% paired bootstrap CIs**, 10,000 resamples, seed 20260909. The same
  resample indices apply to every arm, so between-arm differences use paired
  resampling and inherit its variance reduction.
- **Exact McNemar** on discordant pairs for the four factorial contrasts.
- Reported deltas always carry the CI on the *difference*, not two separate
  per-arm CIs — overlapping marginal CIs do not imply a null difference.

If intervals overlap, the write-up says they overlap. No claim of a winner on a
delta the data cannot support.

## Figures

Six figures in `figures/measured/`, extending `scripts/render_figures.py` with
its existing palette, `dpi=150`, and white facecolor. **Filled = measured**,
mirroring the existing outlined = projected convention so the two sets are never
confused.

| Figure | Content |
|---|---|
| `fig18_measured_ladder.png` | macro-F1 for all six arms with 95% CIs |
| `fig19_factorial_interaction.png` | 2×2 interaction, base model × adaptation |
| `fig20_per_class_f1.png` | per-class F1, 4 arms × 4 classes |
| `fig21_confusion_matrices.png` | 2×2 grid of confusion matrices |
| `fig22_reliability.png` | confidence vs empirical accuracy per arm |
| `fig23_mcnemar_contrast.png` | discordant-pair contingency for the DAPT effect |

`fig19` is the headline: if the lines cross or diverge, that is the interaction
between base model and domain adaptation, and it is the paper's central claim.

## Testing

TDD throughout. All scoring and statistics tests run against a stub model with
deterministic logits — no weights, no GPU, no network.

- `tests/test_llm_outcomes.py` — prompt golden test; mean-log-prob arithmetic;
  calibration subtraction; argmax and softmax agreement; output-schema parity
  with the NB rows; refusal to run when the config's adapter path is absent.
- `tests/test_stats.py` — bootstrap determinism under fixed seed; CI coverage on
  a synthetic distribution with known mean; McNemar against hand-computed
  contingency tables including the zero-discordant edge case.
- `tests/test_outcomes.py` — extended to cover the two promoted functions.

## Execution order

1. Build and unit-test the evaluator locally (stub model, no GPU).
2. **Saul DAPT run on Kaggle.** Blocks `saul-dapt`. Requires re-importing the
   regenerated `notebooks/kaggle_sc2016_dapt.ipynb` (bundle `a3c741ae…`) with
   `RUN = "saul-7b"`.
3. Four-arm evaluation on Kaggle, **in two passes — one per base model**. Qwen
   (~16 GB) and Saul (~29 GB fp32) together exceed comfortable scratch space, so
   each pass stages one base model, runs its two arms, then clears it. The
   disk-clearing logic in the training notebook's model cell is reused.
4. Render figures locally from the downloaded reports.

Evaluation compute is small — roughly 45 cases × 4 short scored continuations
per arm. Model download dominates wall-clock, not inference.

The local RTX 3060 has 6.4 GB VRAM, too tight for an 8B model plus activations,
so evaluation runs on Kaggle alongside training.

## Artifacts

Per arm, under `data/reports/evaluation/<experiment_id>/`:

- `validation-metrics.json`
- `test-metrics.json` (after freeze only)
- `test-predictions.jsonl`
- `llm-outcome.manifest.json` — pinned base revision, adapter sha256, prompt
  version, label surfaces, corpus sha256, manifest sha256, calibration scores

Cross-arm, under `data/reports/evaluation/sc2016-outcome-llm-comparison-v1/`:

- `comparison.json` — per-arm metrics, paired bootstrap CIs, McNemar results for
  all four contrasts

## Risks

| Risk | Mitigation |
|---|---|
| n ≈ 45 gives low power; a real effect may not reach significance | Paired tests and paired CIs; report overlap honestly; frame contribution as the factorial design, not a win |
| Saul ships fp32 (29 GB), long download, tight scratch disk | Two-pass execution; the existing disk guard aborts early with a clear message |
| Contextual calibration could over- or under-correct | Record raw and calibrated scores; report both; decide on validation before freeze |
| Prompt template silently drifts between arms | Golden test pins it; `prompt_view_id` recorded in every manifest |
| Label-eligible test count lower than expected | Report eligible count alongside every metric, as the NB baseline already does |

## Open dependency

`saul-dapt` cannot be measured until the Saul training run completes. The other
three arms can be built, tested, and measured independently, so implementation
is not blocked — only the final row of the matrix is.
