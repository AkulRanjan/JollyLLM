# Phase 5: final test on 2018 (scored once)

Run 2026-10-29 on Kaggle (CPU) as `vanshag08/jolly-phase5-test`. The system was fixed in
`docs/protocol-v2-score-improvement.md` (amendment of 2026-10-29) before any 2018 label was read, and the models were
hashed into `freeze.json` (frozen 11:54:01) before the test cases were loaded. A smoke run with 2014 as a stand-in
(`jolly-phase5-test-smoke`) checked the code first. **The 2018 set is now spent.**

System: TF-IDF (1-2 grams) + balanced logistic regression, view `body-prefix-masked-v2` (6,000 chars), trained on
9,689 cases decided in 2005-2015 and 2017. Test: 645 usable cases decided in 2018 (allowed 271, dismissed 194,
disposed 150, partly allowed 30). Intervals: 10,000 bootstrap resamples.

| Schema | n | Accuracy [95% CI] | Macro-F1 [95% CI] | Always-`allowed` macro-F1 |
|---|---|---|---|---|
| 4-way (comparable with earlier work) | 645 | 0.460 [0.422, 0.499] | **0.421** [0.376, 0.465] | 0.148 |
| 3-way flat (partly merged into allowed) | 645 | 0.515 [0.474, 0.553] | **0.502** [0.461, 0.540] | 0.212 |
| 3-way hierarchy | 645 | 0.501 [0.462, 0.540] | 0.487 [0.447, 0.526] | 0.212 |
| Binary allowed vs dismissed | 495 | 0.620 [0.576, 0.663] | **0.602** [0.556, 0.645] | 0.378 |

Per-class F1, 4-way: allowed 0.501, dismissed 0.392, disposed 0.515, partly allowed 0.276.

## Against earlier numbers

| System | Test | 4-way acc / macro-F1 |
|---|---|---|
| QLoRA Qwen3-8B text (earlier) | SC-2016 + 2017, 608 cases | 0.485 / 0.360 |
| QLoRA Qwen3-8B + statute graph (earlier) | same | 0.451 / 0.365 |
| **TF-IDF + LR, more data (this run)** | 2018, 645 cases | 0.460 / **0.421** |

Different test years, so this is not a paired comparison. The macro-F1 gain mostly comes from the minority classes
(`disposed`, `partly_allowed`), which the 8B runs handled poorly. Accuracy is about the same.

## What the whole plan found

- Model capacity is not the limit: fine-tuned 8B, InLegalBERT and frozen embeddings all match or trail TF-IDF.
- More, recent training data and a balanced linear model give the 4-way macro-F1 of 0.42.
- Task design moves the number most: 0.50 for 3-way, 0.60 for binary.
- The statute graph was not re-tested, since Phase 3 failed its gate and no graph run was frozen.
