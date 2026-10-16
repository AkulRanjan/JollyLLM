# Kaggle run status (checked 2026-10-11)

## 2026-10-13: 8B 2-epoch pair (ranked queue #4), split across sessions

GPU quota on 2026-10-13 at 22:20 was 23:00 / 30 h, so about 7 h were left before the weekly reset (Saturday).
#3 (random-statute control) was skipped because its condition (graph > text on test) did not hold. #4 is next.

- Configs `qwen3-8b-{text,graph-statute}-g3-2ep-bal`: the g3 pair with only `epochs: 2`, the experiment ID and
  the output directory changed. Checkpoints every 96 examples, as before.
- New: `train-outcome --stop-after-hours H` pauses at the first optimiser update past H hours, writes
  `latest-checkpoint` and exits without a manifest. The notebook then skips the freeze. The builder's
  `--part N` writes `lg-<stem>-train-partN`, which attaches part N-1's output and resumes from it (and carries
  `training-log.txt` forward). Evaluate with `--phase evaluate --part <last part>`.
  Test: `ResumeTests.test_time_budget_pauses_and_resumes_to_the_same_result`.
- **Pushed 2026-10-13 22:34 (part 1, `--stop-after-hours 3.0`):** `lg-qwen3-8b-text-g3-2ep-bal-train` and
  `lg-qwen3-8b-graph-statute-g3-2ep-bal-train`. Each session should use about 3.3 GPU-h (about 6.6 h for the pair).
  Tracker log: `checkpoints/kaggle/queue-2ep-part1.log`.
- **Part 1 outcome (checked 2026-10-14 13:33):**
  - Text paused as designed at epoch 1, step 1160, with a checkpoint saved.
  - Graph **stopped with an error**: three non-finite gradients in a row at epoch 1, steps 856–872, so the guard ended the run. Its last checkpoint is at step 864.
  - The 1-epoch graph run, on the same data order, had no such event. Over 2 epochs the warm-up is twice as long, so the learning rate is still near its peak at that point.
  - Resuming unchanged would likely hit the same examples again. Decide a fix before part 2, for example a lower encoder or LoRA learning rate, or skipping that window.
- **After the reset:** each run still needs about 11.7 h of training (2 × 7.3 h, plus validation, minus 3 h).
  Kaggle stops a session at 12 h, so split it into part 2 (`--part 2 --stop-after-hours 6`) and part 3 (`--part 3`,
  no budget). Then run the evaluate kernels once (`--phase evaluate --part 3`). Next week that is about 26 GPU-h.

Source of truth: `py -3.14 -m kaggle kernels list --mine` / `kernels status`, the pulled outputs in
`checkpoints/kaggle/`, and `checkpoints/kaggle/queue-ledger.jsonl`.

- **Finished 2026-10-11 (complete by 23:49):** `lg-qwen3-8b-text-g3-bal-train` and `lg-qwen3-8b-graph-statute-g3-bal-train`
  (7.3 h of training each). They were pulled by hand and both freezes verify (results below).
- **Running since 2026-10-12 00:01:** the one-time test evaluations `lg-qwen3-8b-{text,graph-statute}-g3-bal-evaluate`.
  After them, the queue runs the zero-shot and BM25 baselines (`checkpoints/kaggle/queue-g3-eval.log`).
- **Live log:** `kaggle kernels logs -f <ref>` with `PYTHONUTF8=1`. For a stalled `kernels output`, re-run it
  without `--force`; it skips files it already has.
  Before that, only the two recipe-gate runs had run since 2026-10-07 (2026-10-10 to 10-11).
- **GPU quota:** 0 / 30 h on 2026-10-10 before the gate; about 5.6 h used since, so about 24 h left this week.
- All metrics are **validation only** (46 SC-2016 cases, input view `body-prefix-masked-v2`). No test
  evaluation (`--phase evaluate`) has run, and every manifest records `test_labels_accessed: false`.
- Reference points on the same 46 cases: always predicting `allowed` gives acc 0.413 / macro-F1 0.146
  (a run showing exactly those numbers has collapsed); the train-only Naive Bayes baseline
  (`data/reports/evaluation/sc2016-outcome-baseline-v2`) gives acc 0.435 / macro-F1 **0.397**.
- With 46 cases, a macro-F1 difference under ~0.1 is inside sampling noise.

GPU time is the kernel's total wall time on one T4, including the model download (about 5-10 min).

## Current focus (2026-10-11): Qwen3-8B text vs statute graph on more data

The user narrowed the work to **Qwen3-8B text vs graph**; SaulLM and the 4B seeds come later. Choices made:
train on 2013-2015 judgments as well, add a 2017 test set, and use a statute-only graph.

**Data v0.3** (`training.v3.manifest.json`, corpus `sc-2016-aws/judgments.jsonl`, graph `sc-2016-aws-v3`):

- Source: the public AWS bucket `s3://indian-supreme-court-judgments` (CC-BY-4.0), registered in
  `configs/sources/aws-sc-source.json`. The text is the PDFs' own text layer, extracted on a Kaggle CPU kernel
  (`akxran/lg-aws-sc-extract`, no GPU quota) and pinned by SHA-256. Every archive's hash is in the registry notes.
- Labels: the bucket's official "Disposal Nature" maps onto our four labels. On the 357 SC-2016 training cases it
  agrees with our text rule 333 of 341 times. A bucket case is kept only when both agree.
- Selection (`data/reports/judgments/aws-sc/selection-report.json`), from 3,142 judgments: 2,148 training
  (2013-2015) and 568 test (2017) were kept. Dropped: 211 with no outcome phrase in the text, 162 with another
  disposal (adjourned, referred, ...), 44 where text and official label disagree, 4 duplicates, 2 also in SC-2016
  and 3 decided in another year.
- Checked: the SC-2016 train (357), validation (46) and test (62) examples are identical to v0.2. The new training
  cases run from 2013-01-02 to 2015-12-29. No 2017 case contributes a training-split edge. v0.2 still rebuilds byte
  for byte.
- Training examples: **2,470** (357 SC-2016 + 2,113 bucket), labelled allowed 1,066 / dismissed 809 / disposed
  461 / partly allowed 134. The 2017 test set has 558 usable examples. They are scored with the SC-2016 test in
  the same single evaluation pass and reported separately (`metrics_by_source`;
  `compare-outcomes --source ...`).
- PDF text quality: about 98% of words are clean (PaddleOCR 99.5%). It recovers about 88% of the statute
  citations found in the PaddleOCR text of the same judgments.

**Statute-only graph** (`selector: cocitation`): no case nodes. A provision is a candidate when an eligible
(training-split, earlier) judgment cites it together with a provision cited in the query's visible text. Two
provisions are linked when such a judgment cites both. On validation, coverage is the same 39% as before, about
12.5 provisions per case, and every node is a provision. The old PCST graph had 82% case titles.

| Kernel | What | Epochs | Est. GPU time |
| --- | --- | --- | --- |
| lg-qwen3-8b-text-g3-bal-train | Qwen3-8B text, v0.3 data, -bal recipe | 1 | ~7.5 h |
| lg-qwen3-8b-graph-statute-g3-bal-train | Same, plus statute-only graph tokens | 1 | ~7.5 h |
| (later) both `--phase evaluate` | SC-2016 test and 2017 test, once | — | ~0.7 h text, ~1.2 h graph (also scored with graph tokens removed) |

**Validation result (46 SC-2016 cases, 1 epoch; local paired bootstrap with 5,000 resamples):**

| Run | Acc / macro-F1 | 18 cases with a subgraph | 28 cases without | Predicted (allowed/dismissed/disposed/partly) |
|---|---|---|---|---|
| text-g3-bal | 0.457 / **0.405** | 0.444 / 0.433 | 0.464 / 0.341 | 16 / 13 / 13 / 4 |
| graph-statute-g3-bal | 0.413 / 0.385 | 0.389 / 0.372 | 0.429 / 0.322 | 12 / 11 / 20 / 3 |

- graph − text macro-F1 is −0.020, 95% CI [−0.111, +0.075], so there is no detectable difference.
- Discordant cases are 3 only-graph-right vs 5 only-text-right. On the cases with a subgraph it is 1 vs 2, against 2 vs 9
  for the earlier PCST graph.
- The graph run over-predicts `disposed` (20 predicted, 8 gold).
- Text-g3 (0.405) is above text-bal on v0.2 (0.377) and above Naive Bayes (0.397). Both runs predict all four classes.
- Subgraph coverage matches the local build: train 44.8%, validation 39.1%, 12.5 nodes.
- Neither log has a non-finite step. Charts: `checkpoints/kaggle/figures/g3-final/`.

Training loss over the same windows is nearly identical. Text's average loss over the epoch is higher only because
its first 250 steps started high (8.8 vs 2.6). Recent windows: steps 1500-1750 text 0.673 / graph 0.675; steps
1750-1850 text 0.605 / graph 0.584.

**Test (scored once, 608 cases: 50 SC-2016 + 558 from 2017; analysis in `papers3/data/analysis.json`):**

| System | SC-2016 acc / mF1 | 2017 acc / mF1 | All acc / mF1 |
|---|---|---|---|
| Always `allowed` | 0.500 / 0.167 | 0.462 / 0.158 | 0.465 / 0.159 |
| Zero-shot | 0.340 / 0.250 | 0.367 / 0.268 | 0.365 / 0.269 |
| BM25 statute-evidence prompt | 0.320 / 0.211 | 0.448 / 0.259 | 0.438 / 0.255 |
| QLoRA text | **0.440 / 0.316** | **0.489** / 0.365 | **0.485** / 0.360 |
| QLoRA + statute graph | 0.320 / 0.226 | 0.462 / **0.379** | 0.451 / **0.365** |
| Graph, tokens removed | 0.320 / 0.223 | 0.461 / 0.378 | 0.449 / 0.364 |

Paired comparisons (10,000 bootstrap resamples; McNemar on accuracy):
- Graph − text, 2017: macro-F1 +0.014 [−0.028, +0.058], accuracy −0.027 (p = 0.20).
- Graph − text, SC-2016: macro-F1 −0.090 [−0.179, −0.007], accuracy −0.120 (p = 0.07).
- Graph − text, all: macro-F1 +0.006 [−0.034, +0.046], accuracy −0.035 (p = 0.08).
- Text − zero-shot, all: macro-F1 +0.090 [+0.037, +0.143]. Graph − zero-shot, all: macro-F1 +0.096 [+0.038, +0.153].

What the graph changes:
- On 2017, dismissed F1 rises from 0.272 to 0.395 and partly-allowed F1 from 0.109 to 0.150; allowed F1 falls from 0.630 to 0.551.
- 218 of 608 test cases (35.9%) get a subgraph, averaging 31.7 statute nodes.
- On those 218 cases, macro-F1 is 0.403 for graph, 0.381 for text and 0.385 for graph with tokens removed.
- Removing the graph tokens changes 32 predictions (5.3%).

- Text − BM25, all: macro-F1 +0.104 [+0.057, +0.152]. Graph − BM25, all: macro-F1 +0.110 [+0.056, +0.164].

All test evaluations are done; nothing may be re-scored on test. The zero-shot kernel saved its test result,
then errored on the graph-ablation step, which applies to graph variants only. The builder has been fixed.

**Journal manuscript (2026-10-12):** `papers3/paper.pdf` and `paper.docx` (27 pages, about 11,300 words,
64 references). Every result number is generated from the frozen runs by `papers3/analysis.py` and
`papers3/probes.py`; see `papers3/README.md` to rebuild it.

### Accuracy probes (2026-10-12, local CPU, training split only, no test labels)

TF-IDF (1-2 grams) with balanced logistic regression, trained on the 2013-2014 cases (1,430) and scored on the
2015 cases (683). This dev set is far larger than the 46-case validation set.

| Probe | Acc | Macro-F1 |
|---|---|---|
| Always `allowed` | 0.425 | 0.149 |
| Current view (body prefix, 6,000 chars) | 0.483 | 0.379 |
| Longer prefix (24,000 chars) | 0.493 | 0.372 |
| Last 6,000 chars before the 70% cap | 0.460 | 0.354 |
| Whole body up to 70% | 0.486 | 0.368 |
| Case details only (case type, bench, party words) | 0.341 | 0.276 |
| Current view + case details | 0.488 | 0.381 |
| Diagnostic only: text at 70-80% (next to the label region) | 0.430 | 0.298 |
| Learning curve: 357 / 715 / 1,430 training cases | 0.460 / 0.459 / 0.483 | 0.308 / 0.320 / 0.379 |
| 3-way (partly merged into allowed) | 0.515 | 0.442 |
| Binary allowed vs dismissed (disposed dropped, 548 dev cases) | 0.633 | 0.604 |

What this shows:
- The fine-tuned 8B (test 0.485 / 0.360) scores about the same as this bag-of-words model. On this view the limit is
  the input and the task, not the model or the graph.
- Longer input and case details add nothing.
- More data still helps (+0.06 macro-F1 from 715 to 1,430 cases).
- The task framing moves the numbers most.

### Ranked queue (2026-10-11)

Ranked by importance to the 8B text-vs-graph claim. GPU figures include test evaluation. Test scoring runs at about
2.9 s per example (measured on validation), so the new 620-case test takes about 30 min per pass.

| # | Run | What it answers | GPU |
|---|---|---|---|
| 1 | Pull the g3 pair and compare validation | first look (46 cases, noisy) | 0 |
| 2 | Test-evaluate the g3 pair | headline: statute graph vs text on 620 test cases | ~2 h |
| 3 | 8B random-statute control (only if graph > text) | whether the specific statutes matter, or just the 8 extra tokens; needs a small selector change | ~8.5 h |
| 4 | 8B 2-epoch pair | more epochs; each run spans 2 Kaggle sessions via checkpoint resume | ~31 h |
| 5 | Qwen3-4B text + statute graph, 3 seeds each | whether the gap holds across seeds | ~30 h |
| 6 | SaulLM-7B text + statute graph | a second model family | ~16 h |
| 7 | Longer input (3072 tokens) pair | longer input; T4 memory still unchecked | ~32 h |
| 8 | DAPT init, then a pair | domain pretraining; most expensive and least certain | ~25 h+ |
| 9 | Zero-shot and BM25 on the new test | ladder baselines | ~1.5 h |

Prepared on 2026-10-11 at 21:50 (none pushed yet):
- #2: `kaggle/lg-qwen3-8b-{text,graph-statute}-g3-bal-evaluate`. Their sources are the train kernel and `lg-aws-sc-extract`.
- #9: configs `qwen3-8b-zero-shot-g3` and `qwen3-8b-bm25-prompt-g3`, which are the old baseline configs pointed at
  the v0.3 data, and their train notebooks. Their evaluate notebooks are built after the train kernels finish.

Progress and figures (added 2026-10-11 at 22:05; all four notebooks above were rebuilt with it):
- Test evaluation prints a live line every 25 cases: cases done, time left, and accuracy and macro-F1 so far. The
  model is frozen, so these only settle; they never improve.
- At the end it shows a per-dataset table and figures (also saved under `figures/`): how the score settled, confusion
  matrices per dataset, and per-class F1 with and without graph tokens.
- New training notebooks keep `training-log.txt` and plot a loss and validation curve.
- Locally: `legal_graph.cli report-outcome --figures DIR [--run RUN] [--training-log LOG ...]`. Charts made from the
  g3 live logs at 22:02 are in `checkpoints/kaggle/figures/g3-live-2202/`.

The local GPU (RTX 3060 Laptop, 6 GB) is too small for Qwen3-8B in 4-bit: the embedding and output layers stay 16-bit,
so the weights alone need ~6.1 GB. Qwen3-4B (tied embeddings, ~2.7 GB) should fit.

These are no longer needed:
- 8B graph-untyped re-run: the statute graph has one edge type, so typed and untyped are the same.
- 8B graph-typed expansion: replaced by the cocitation selector.
- v0.2 graph-random runs: replaced by #3.
- Saul graph-typed-PCST (v0.2): replaced by #6.
- The twelve v0.2 4B seed runs: replaced by #5.
- The 15 stale notebooks.

## Recipe gate result (pushed 2026-10-10 22:43, pulled 2026-10-11 12:33)

Both kernels completed, and every file matches its `freeze.json` hash. Neither log shows a skipped or
non-finite step.

| Kernel | Variant | Val macro-F1 per epoch | Best (acc / mF1) | Training time | Predictions on val |
|---|---|---|---|---|---|
| lg-qwen3-8b-text-bal-train | text | 0.192 → 0.206 → **0.377** | e3: 0.478 / **0.377** | 167 min | 3 classes, never `partly_allowed` |
| lg-qwen3-8b-graph-typed-pcst-g2-bal-train | graph typed, PCST, g0.2 | 0.117 → 0.192 → 0.283 | e3: 0.435 / 0.283 | 151 min | all 4 classes (1 `partly_allowed`, wrong) |

Recipe: inverse-frequency class weights, warmup 0.1 + linear decay, lr 2e-4, grad clip 1.0. Configs:
`configs/training/outcome/qwen3-8b-text-bal.json`, `qwen3-8b-graph-typed-pcst-g2-bal.json` (the original 8B
configs with only the recipe fields, experiment ID and output dir changed; checkpoints every 96 examples).
GPU used: about 5.6 h (the local tracker was stopped for low memory; outputs were pulled by hand).

Paired bootstrap on the same 46 validation cases (5,000 resamples, macro-F1):

| Comparison | Δ | 95% CI |
|---|---|---|
| text-bal − text (original) | −0.064 | [−0.190, +0.061] |
| graph-bal − graph (original) | +0.039 | [−0.066, +0.145] |
| graph-bal − text-bal | −0.093 | [−0.223, +0.057] |

**Gate not passed:** the criterion was "8B text-bal matches or beats 0.441". It reached 0.377. The gap is
inside noise, and the new recipe is stable (it improves every epoch, while the original peaked at epoch 2
and fell to 0.168). Under both recipes the graph run is below the text run. Per the gate rule, the recipe
choice goes back to the user before any more GPU time is spent.

## Graph diagnosis (2026-10-11, local, no GPU, validation only)

Each validation case's PCST subgraph was rebuilt from the run's own `config.json`. The coverage reproduces
the manifest exactly (39.1% non-empty, 6.7 nodes, 25.5 edges). Predictions were then split by whether the
case received a subgraph:

| Subset | n | text | text-bal | graph | graph-bal |
|---|---|---|---|---|---|
| With subgraph (acc / mF1) | 18 | 0.556 / 0.457 | 0.333 / 0.273 | 0.278 / 0.172 | 0.222 / 0.157 |
| Empty subgraph (acc / mF1) | 28 | 0.571 / 0.437 | 0.571 / 0.418 | 0.536 / 0.286 | 0.571 / 0.390 |

- **Where the graph is absent, graph runs track text runs.** On the 28 empty cases, graph-bal and text-bal
  have the same accuracy (0.571). Discordant cases split evenly (only-graph-right 10 vs only-text-right 11,
  both recipes combined).
- **Where the graph is present, it hurts.** On the 18 cases with a subgraph, only-graph-right is 2 vs
  only-text-right 9 (both recipes combined, so each case counts twice). Graph-bal reaches 0.222 accuracy there,
  below that subset's majority rate (7/18 = 0.389). With n = 18 this is suggestive, not conclusive.
- **What the graph carries:** 251 of 307 subgraph nodes (82%) are other judgments that cite the same
  provisions. By design, a case node contributes only its title (no outcome text). There are 51 provision
  nodes (IPC 35, Constitution 10, CrPC 6), all with real text from the v0.2 provision corpus. There is
  no leak and no missing-text bug. The prefix is mostly co-citing case titles, learned from the ~137
  training cases that have a subgraph.
- **Implication:** running the rest of the ladder unchanged (untyped, random, 4B seeds; ~24 GPU-h) will most
  likely repeat "graph ≤ text". Fixing what the graph carries comes before the expensive seeds.

## Finished

| Kernel | Model | Variant | Epochs | Val macro-F1 per epoch | Best (acc / mF1) | GPU time | Predictions on val |
|---|---|---|---|---|---|---|---|
| lg-qwen3-8b-text-train | Qwen3-8B | text | 3 | 0.146 → **0.441** → 0.168 | e2: 0.565 / **0.441** | 3.12 h | 3 classes, never `partly_allowed` |
| lg-qwen3-8b-graph-typed-pcst-g2-train | Qwen3-8B | graph typed, PCST, g0.2 | 3 | 0.173 → 0.180 → 0.244 | e3: 0.435 / 0.244 | 3.06 h | only `allowed` and `dismissed` |
| lg-qwen3-8b-graph-untyped-g2-train | Qwen3-8B | graph untyped, g0.2 | 3 | 0.146 → 0.146 → 0.146 | **invalid** (loss NaN from step ~50) | 2.48 h | all `allowed`; must re-run |
| lg-qwen3-8b-zero-shot-train | Qwen3-8B | zero-shot (no training) | 0 | 0.318 | 0.348 / 0.318 | 0.09 h | all 4 classes |
| lg-qwen3-8b-bm25-prompt-g2-train | Qwen3-8B | BM25 evidence prompt (no training) | 0 | 0.214 | 0.391 / 0.214 | 0.09 h | mostly `allowed` |
| lg-saul-7b-text-train | Saul-7B | text | 3 | 0.183 → 0.146 → 0.181 | e1: 0.435 / 0.183 | 2.81 h | 43 of 46 `allowed` |
| lg-qwen25-0-5b-smoke-graph-typed-g2-train | Qwen2.5-0.5B | smoke, graph typed g0.2 | 1 | 0.193 | 0.370 / 0.193 | 0.05 h | pipeline check only |

Diagnostics (Qwen3-4B, text; excluded from the paper aggregate):

| Kernel | What it tests | Epochs | Val macro-F1 per epoch | Best (acc / mF1) | GPU time |
|---|---|---|---|---|---|
| lg-diag-qwen3-4b-text-overfit32-train | can it fit 32 examples | 8 | 0.146 … 0.246 (train loss to 0.06) | e8: 0.261 / 0.246 | 0.70 h |
| lg-diag-qwen3-4b-text-ga1-train | gradient accumulation 1 | 2 | 0.117 → 0.146 | e2: 0.413 / 0.146 (all `allowed`) | 1.27 h |
| lg-diag-qwen3-4b-text-balanced-train | inverse-frequency class weights, lr 1e-4 | 3 | 0.139 → 0.150 → 0.230 | e3: 0.304 / 0.230 | 1.99 h |
| lg-diag-qwen3-4b-text-balanced-sched-train | class weights + warmup 0.1 + linear decay, lr 2e-4 | 3 | 0.146 → 0.098 → 0.275 | e3: 0.348 / 0.275 | 1.85 h |

Infrastructure: `lg-outcome-selftest`, `lg-fetch-statutes`, `lg-unittest-cpu` (164 tests OK, 7 skipped).
GPU used 2026-10-06 to 10-07: about **17.7 h** (previous quota week).

## Measured vs projected (validation macro-F1)

The paper's projection (`data/reports/evaluation/sc2016-outcome-projection-v1`) names Qwen2.5-7B; the
measured rungs use Qwen3-8B. The rungs are paired, not the models.

| Rung | Projected | Measured | Status |
|---|---|---|---|
| Zero-shot prompt | 0.449 | 0.318 | measured |
| + BM25 statute evidence | 0.508 | 0.214 | measured |
| + QLoRA text | 0.569 | 0.441 | measured (unstable: falls to 0.168 at epoch 3) |
| + untyped graph prefix | 0.641 | — | invalid run (NaN) |
| + typed R-GCN prefix + PCST | 0.657 | 0.244 | measured; below text and below Naive Bayes |
| Saul-7B, same graph stack | 0.696 | — | not run (Saul text: 0.183) |

So far the measured graph rung is **below** the text rung, the reverse of the projection. Two structural
limits are worth knowing: only ~39% of examples get a non-empty subgraph (mean ~7 nodes), so most graph-run
inputs equal the text input; and the v0.2 graph has three relation types (`contains`, `refers_to_provision`,
`decided_by` to a single court node), so typed and untyped differ only slightly.

## Still to run (training phase) — superseded by the ranked queue above (v0.3 data, statute graph)

All 15 never-run notebooks in `kaggle/` were built on 2026-10-07 01:21, before commits c61c008 (gradient
clipping, NaN guard) and f611336 (class weighting, LR schedule). A decoded source bundle confirms they lack
both. **Rebuild with `scripts/build_kaggle_notebook.py --outcome <stem>` before pushing.** Their configs
still use the original recipe (lr 1e-4, no class weights, no schedule) that collapsed in the runs above.
Every config writes resumable checkpoints every 96 examples.

| Kernel | Model | Variant | Epochs | Est. GPU time |
|---|---|---|---|---|
| lg-qwen3-8b-graph-untyped-g2-train (re-run) | Qwen3-8B | graph untyped | 3 | ~3.1 h |
| lg-qwen3-8b-graph-random-g2-train | Qwen3-8B | graph random (control) | 3 | ~3.1 h |
| lg-qwen3-8b-graph-typed-g2-train | Qwen3-8B | graph typed, expansion selector | 3 | ~3.1 h |
| lg-saul-7b-graph-typed-pcst-g2-train | Saul-7B | graph typed, PCST | 3 | ~2.9 h |
| lg-qwen3-4b-text-s{1,2,3}-train | Qwen3-4B | text, 3 seeds | 3 each | ~2.0 h each |
| lg-qwen3-4b-graph-typed-pcst-g2-s{1,2,3}-train | Qwen3-4B | graph typed, PCST, 3 seeds | 3 each | ~2.0 h each |
| lg-qwen3-4b-graph-untyped-g2-s{1,2,3}-train | Qwen3-4B | graph untyped, 3 seeds | 3 each | ~2.0 h each |
| lg-qwen3-4b-graph-random-g2-s{1,2,3}-train | Qwen3-4B | graph random, 3 seeds | 3 each | ~2.0 h each |

Training subtotal: 12.2 h (8B and Saul) + 24 h (twelve 4B runs) = **~36 GPU-hours**, about 18 h of wall
time at 2 concurrent sessions. If the recipe changes, re-running the 8B text and 8B graph-typed-PCST controls
under the same recipe adds ~6.2 h (Saul text: +2.8 h).

Measured per-epoch times on a T4 (357 training examples): Qwen3-8B about 60 min, Saul-7B about 54 min,
Qwen3-4B about 36-39 min. Class weighting and the LR schedule do not change the time.

## After training

1. **Test evaluation:** one `--phase evaluate` kernel per frozen candidate (about 20 runs), mostly model
   download, about 0.1-0.2 h each, so roughly **3 GPU-hours**.
2. **Aggregate:** run `legal_graph.cli collect-outcomes` locally (no GPU). The current
   `data/reports/evaluation/sc2016-outcome-measured-v1/measured-results.md` is an empty table from 2026-10-07.
3. **Figures:** `scripts/render_figures.py` draws figures 8-16 from the *projection*, not from measured runs.
   Measured-result figures still need to be added.

Grand total remaining: **~39 GPU-hours** (about 45 h with the control re-runs), so more than one quota week.

## Graph-generation files

Knowledge graph (judgments and statutes):

| File | What it builds |
|---|---|
| `src/legal_graph/judgments.py` (`build_judgment_graph`), CLI `ingest-sc2016` | SC-2016 judgment graph: `data/graphs/judgments/sc-2016` (v0.1: 1,990 nodes, 2,123 edges) and `sc-2016-v2` (v0.2: 3,215 nodes, 5,361 edges) |
| `src/legal_graph/statutes.py`, `scripts/build_statute_layer_v2.py` | Statute layer: `data/graphs/india_statutes` and `india_statutes-v2` (BNS, BNSS, CrPC, IPC, Evidence Act, Constitution) |
| `src/legal_graph/extraction.py`, `builder.py` | Citation extraction (extractor v2) and graph assembly |
| `src/legal_graph/query_graph.py`, `pcst.py` | Per-case, split-safe and time-safe query subgraphs; greedy PCST selector |
| `src/legal_graph/graph_encoder.py`, `prefix.py` | R-GCN encoder and soft-token graph prefix |

Figures (charts): `scripts/render_figures.py` writes `figures/corpus/fig1-7` (measured corpus statistics),
`figures/projection/fig8-16` (projected results), and `figures/diagrams/fig17` (pipeline diagram).
The corpus figures read graph and statute layer **v0.1** (`sc-2016`, `india_statutes`), so fig3 (graph
structure) and fig6 (citation resolution) do not show the v0.2 graph that the `*-g2` runs use.
