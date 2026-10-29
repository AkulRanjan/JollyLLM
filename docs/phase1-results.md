# Phase 1 results: TF-IDF probes on the 2015 development set

Run 2026-10-29 on Kaggle (CPU) as `vanshag08/jolly-phase1-probes`, reading the output of `vanshag08/jolly-aws-extract`.
Protocol: `docs/protocol-v2-score-improvement.md`. Every number is **dev only** (train 2013-2014, 1,430 cases;
dev 2015, 683 cases). No 2017 or 2018 file was opened. Raw results: `phase1-results.json` in the kernel output.

Reproduction check: the case counts match the earlier probe exactly (1,430 / 683). The 4-way score is acc 0.499 /
macro-F1 0.395 against the earlier 0.483 / 0.379. The earlier probe script is not in the repo, so the small gap
(+0.016) is probably its vectoriser settings; this notebook uses TF-IDF 1-2 grams, min_df 2, sublinear tf, balanced
logistic regression, C = 1.

## Outcome against the decision rule (macro-F1 gain of at least 0.03, bootstrap interval excluding 0)

**Nothing was adopted.**

| Probe | Setting | Delta vs baseline | 95% interval |
|---|---|---|---|
| Data scale | train 2005-14 (8,448) vs 2013-14 (1,430) | +0.012 | [-0.043, +0.068] |
| Input view | opening 24,000 chars | -0.007 | [-0.037, +0.025] |
| Input view | whole body to the 70% cap | -0.013 | [-0.042, +0.018] |
| Input view | last 6,000 chars before the cap | -0.044 | [-0.083, -0.005] |
| Calibration | offsets tuned out-of-fold, train 2013-14 | +0.004 | [-0.029, +0.039] |
| Calibration | offsets tuned out-of-fold, train 2005-14 | 0.000 | [-0.019, +0.020] |

## Label schema (same features, same split)

| Schema | Acc | Macro-F1 |
|---|---|---|
| 4-way | 0.499 | 0.395 |
| 3-way flat (partly merged into allowed) | 0.521 | 0.451 |
| 3-way hierarchy (disposed first, then allowed vs dismissed) | 0.511 | 0.428 |
| Binary allowed vs dismissed (548 cases) | 0.624 | 0.593 |
| Binary always-`allowed` floor | 0.597 | 0.374 |

Schemas are not comparable with each other. Only the 4-way number is comparable with earlier work.

## Data scale

| Training window | Cases | 4-way macro-F1 |
|---|---|---|
| 2014 | 699 | 0.359 |
| 2013-14 | 1,430 | 0.395 |
| 2011-14 | 2,731 | 0.408 |
| 2009-14 | 4,532 | 0.420 |
| 2007-14 | 6,973 | 0.411 |
| 2005-14 | 8,448 | 0.407 |

Random subsets of the 2005-14 pool (5 seeds): 357 -> 0.318, 715 -> 0.317, 1,430 -> 0.343, 2,860 -> 0.378,
4,300 -> 0.391, 8,448 -> 0.407.

- More data helps a bag-of-words model up to about 4,500 cases and then flattens at about 0.41-0.42.
- Recency matters: 1,430 random cases from 2005-14 score 0.343, while the 1,430 cases of 2013-14 score 0.395. Part of the
  earlier "+0.06 from 715 to 1,430" slope was probably the move to more recent years, not size alone.

## Input view and leakage audit

- No view still contains a literal disposition phrase (0% in all four views), but 65-91% contain outcome words
  (allow*, dismiss*, dispos*, set aside ...) in other contexts.
- A model using only those words scores 0.232; removing them from the baseline changes nothing (0.396). The signal
  does not come from residual outcome words.
- Top baseline features are topical: `insurance company`, `accident`, `tribunal`, `suit` push toward allowed; `arms act`,
  `leave petition`, `these appeals` toward disposed; `award`, `compensation`, `imprisonment`, `sentence` toward partly
  allowed. The `dismissed` features (`september`, `1995`, `her`, `exchange`) look like noise.
- Reading: the model mostly learns case type and procedural posture, which fits the ceiling around 0.40-0.42.

## What this means for the plan

- The ceiling for a bag-of-words model on the 4-way task is about 0.40-0.42, whatever the data size beyond a few thousand
  cases, input view or calibration. The label schema moves the number most (3-way 0.45, binary 0.59).
- Phase 2's data scaling is already done (2005-2015 extracted, hashes verified), but on this evidence more training cases
  alone will not lift a TF-IDF model. They may still help a neural model (Phase 3 tests that).
- Phase 3 gate stands: a neural model has to beat TF-IDF by at least 0.03 with a bootstrap interval excluding 0. If it does
  not, the lever is task design, not model capacity.

## Caveats

- One dev year, one seed for the vectoriser settings; the bootstrap resamples cases, not years.
- Older years (2005-2012) may have a different text layout from 2013-2015; the recency effect above could partly be that.
- The dev set may contain a few cases that are also in SC-2016 (the 2026-10-11 selection removed 2 of about 3,100).
