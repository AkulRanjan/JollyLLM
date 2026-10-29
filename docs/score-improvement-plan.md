# Score improvement plan

Status (2026-10-29): Phases 0-3 and 5 done; Phase 4 skipped because Phase 3 failed its gate. Results are in
`docs/phase1-results.md`, `docs/phase3-results.md` and `docs/phase5-test-results.md`. Written 2026-10-26 from `docs/kaggle-run-status.md`, the README and a read of
`src/legal_graph`. GPU costs are linear extrapolations from past runs, not measurements.

## 1. Why more fine-tuning alone is the weakest lever

| Observation (source: `docs/kaggle-run-status.md`) | Implication |
|---|---|
| Fine-tuned Qwen3-8B scores acc 0.485 / macro-F1 0.360 on test, about the same as TF-IDF + logistic regression (0.483 / 0.379 on the 2015 dev set) | The model is not the limit. A bigger model or more epochs will not help much. |
| Longer input (24k chars) and case details add nothing | More text of the same kind does not help. |
| Learning curve 715 -> 1,430 cases gives +0.06 macro-F1 | More data is the only lever with a demonstrated slope. |
| 3-way schema (partly merged into allowed) gives 0.442; binary allowed vs dismissed gives 0.604 | The 4-label schema caps the score. |
| Validation set is 46 cases (noise about +/-0.1); the 3-epoch original recipe peaked at 0.441 then collapsed to 0.168 | Tuning on this set is mostly noise, and training is fragile. |
| Graph minus text macro-F1 = +0.006, 95% CI [-0.034, +0.046]; only about 36% of cases get a subgraph | The graph has not shown an effect yet. |

The README forecast of roughly 75% accuracy was an assumption (`projected_not_measured`) and does not match
measurements. Realistic targets: 4-way macro-F1 about 0.40-0.45, binary about 0.65-0.70.

## 2. Experiment plan

### Phase 0 - lock the protocol (no compute, about 0.5 day)

Record as a new protocol version per `versionup.md`, before any result is seen:

- **Dev set:** train on 2013-14, dev on 2015 (683 cases), or year-blocked cross-validation. Replaces the 46-case
  validation set for all tuning.
- **Fresh test set:** a new year (for example 2018) from the AWS bucket. Its labels are not looked at until
  Phase 5. The 2016 and 2017 test sets stay retired; nothing is re-scored.
- **Decision rule**, for example: adopt a change only if dev macro-F1 improves by at least 0.03 and the paired
  bootstrap CI excludes 0.
- **Schemas reported side by side:** the original 4-way schema, a 3-way schema (partly allowed merged into
  allowed), and a hierarchy (allowed vs dismissed first, then `disposed`). Never replace the 4-way numbers.

### Phase 1 - cheap local probes (no GPU, 1-2 days)

TF-IDF + logistic regression on the large dev set:

| Probe | Question | Gate |
|---|---|---|
| Schema | How much of the ceiling is label design? | Choose the primary schema |
| Data scale | Slope from 1.4k to 2.5k cases | Decides whether Phase 2 is worth it |
| Input view | Does a reasoning-section view beat the opening-section view? | Only if the leakage audit passes |
| Calibration | Does tuning class biases on dev lift macro-F1? | Free win if yes |

Leakage audit for any new input view: measure how many cases still contain near-disposition phrases in the view,
and compare against a view built only from the masked tokens. If the gain comes from unmasked phrases, discard it.

### Phase 2 - scale the data (Kaggle CPU kernel, about 0 GPU quota)

- Extract more years from `s3://indian-supreme-court-judgments`: earlier (2005-2012) for training, later (2018) for
  the fresh test.
- Keep every training year strictly before every test year.
- Keep the filter: a case is kept only when the official "Disposal Nature" and the text rule agree.
- Pin the new archive hashes in `configs/sources/aws-sc-source.json`.
- Target about 6-10k training cases, up from about 2.4k.

### Phase 3 - cheap neural baselines (about 1-3 GPU-h total)

- InLegalBERT / LegalBERT with a classification head (named in plan 3, never run).
- Frozen LLM embeddings + logistic regression.
- Ensemble with TF-IDF.
- Gate: if none beat TF-IDF by at least 0.03, the lever is data and task design; skip the expensive LLM runs.

### Phase 4 - one expensive LLM run, only if Phase 3 justifies it

| Run | Approx. GPU-h | Fits one week (30 h quota)? |
|---|---|---|
| Qwen3-4B text, 1 epoch, about 8k cases | about 13 | Yes |
| Qwen3-8B text, 1 epoch, about 8k cases | about 22-25 | Barely; split across sessions with `--stop-after-hours` |
| Dev scoring and calibration | 1-2 | Yes |

Recommendation: 4B first. It leaves quota for a control run. Use the existing generative label scoring, or a
classification head if Phase 3 favoured one.

Graph question, only after a stable text baseline:

- Run the skipped random-statute control.
- Try a subgraph that carries earlier cases' outcomes (not just titles). This is allowed only if it is time-safe and
  uses training-split cases only.
- Known weak point: coverage of about 36%.

### Phase 5 - freeze and score once (about 1-2 GPU-h)

- Freeze the winning text and graph runs (`freeze-outcome`).
- Score the fresh test once (`evaluate-outcome`).
- Paired bootstrap, 10,000 resamples. Report all schemas next to the old numbers.

## 3. Budget summary

| Phase | Compute | Rough time |
|---|---|---|
| 0 Protocol | none | 0.5 day |
| 1 Local probes | local CPU | 1-2 days |
| 2 Data scaling | Kaggle CPU | 1 day, mostly waiting |
| 3 Cheap neural baselines | 1-3 GPU-h | 1 day |
| 4 One LLM run + control | 15-30 GPU-h | up to 1 week |
| 5 Freeze and test | 1-2 GPU-h | 0.5 day |

Reference numbers behind the estimates: 8B training about 10 s per example, 4B about 6 s per example, test scoring
about 2.9 s per example. Measure the first few hundred examples of any run before committing the full budget.

## 4. Deprioritised

- Domain-adaptive pretraining (most expensive, least certain).
- More 4B seed runs.
- The longer-input (3072 token) pair, until a cheap proxy shows value.

## 5. Risks

- Changed labels are not comparable to old numbers. Always report the original 4-way schema alongside.
- A reasoning-section view can leak the outcome. Treat a large jump as suspicious until the audit passes.
- A null result is possible. If Phase 1 and 3 show the ceiling is data, "the graph does not help" is a legitimate,
  publishable outcome.
- Cost figures are extrapolations.

## 6. Open decisions

- Is the goal a higher number, or a defensible text-vs-graph comparison? This decides whether a changed label schema
  is acceptable.
- How much GPU quota is left this week? This sets how much of Phase 4 fits.
