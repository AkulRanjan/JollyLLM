# Round 2 results: header metadata, recency weighting, label audit

Run 2026-10-29 locally (CPU) from `kaggle-notebooks/jolly-round2-probes`; output in
`checkpoints/kaggle/jolly-2026-10-29/local-round2/` (git-ignored). Probes were fixed in the protocol v2 amendment
"round 2" before any result was seen. Dev only (train 2005-2014, 8,448 cases; dev 2015, 683 cases). No file after 2015
was opened by the probes.

## Decision rule (macro-F1 gain of at least 0.03, bootstrap interval excluding 0)

**Nothing was adopted.**

| Candidate | Acc | Macro-F1 | Delta vs text | 95% interval |
|---|---|---|---|---|
| baseline: text TF-IDF + LR | 0.455 | 0.409 | | |
| (a) header metadata only | 0.376 | 0.324 | -0.085 | [-0.136, -0.031] |
| (a) text + metadata | 0.416 | 0.353 | -0.057 | [-0.105, -0.006] |
| (b) text, recency half-life 4 years | 0.458 | 0.412 | +0.003 | [-0.025, +0.032] |
| (c) text + metadata, half-life 2 years | 0.436 | 0.360 | -0.049 | [-0.098, +0.002] |

The half-life was chosen on the inner split (train 2005-2013, score 2014): text 0.350 / 0.383 / 0.390 / 0.383 / 0.382
for half-life 1 / 2 / 4 / 8 / none. On the other schemas, recency also changes little (3-way 0.452 -> 0.464, binary
0.589 -> 0.600).

## Header metadata

Fields found: case type, age, bench and party types 100%; judges 93%; court appealed from 87%; counsel for the appellant
76%, for the respondent 68%; connected matters 21%.

- On its own the header carries real signal (0.324 against 0.148 for always-`allowed`), but less than the body text.
- Added to the text it **hurts**, and the inner split showed the same (0.328-0.352 against 0.382-0.390). The strongest
  metadata weights are individual counsel names, which points to over-fitting on thousands of rare tokens.
- Likely design flaw, found after the result: the binary metadata block was not scaled against the length-normalised
  TF-IDF rows, so a case's roughly 25 metadata tokens can outweigh its text. A version with a block weight or without
  the counsel tokens, chosen on the inner split, is a new probe and has not been run.
- Dropping one field group at a time changes macro-F1 by at most 0.018 (judges -0.015, court +0.018). No single field
  carries much.

## Label audit (training years 2005-2014)

Each label maps to exactly one official "Disposal Nature" value, so the labels are the registry's own. Reading a fixed
random sample of judgment endings (40 `disposed`, 25 `partly allowed`, read by Claude, not by a lawyer):

| `disposed` ending, what the court actually did | Count of 40 |
|---|---|
| Directions, settlement terms, or writ/contempt closed without a winner | about 19 |
| Order set aside and matter sent back to a lower court | about 7 |
| Partial relief (sentence reduced, conviction altered, amount changed) | about 5 |
| Batch with mixed results (one appeal allowed, another dismissed) | about 4 |
| Effectively allowed or dismissed (for example "appeals filed by the assessees succeed", "appeal infructuous and dismissed") | about 5 |

- About half of `disposed` is a coherent "no winner, directions issued" class. The rest, about 40%, would fit another
  label just as well: partial relief reads like `partly allowed`, remands often read like `allowed`.
- `partly allowed` is consistent: 25 of 25 are partial relief (sentence cut, compensation or interest changed, relief to
  some appellants). But the same kind of outcome also appears under `disposed`.
- So part of the 4-way ceiling is how the Registry labels outcomes, which no model can fix from the text. One reading of
  40 cases is a rough estimate.

## 2019 test set

`vanshag08/jolly-aws-extract-2019` extracted 1,050 records for 2019 and 571 for 2020 (counts only, no labels printed).
Archive hashes are pinned in `configs/sources/aws-sc-source.json`. **Since nothing was adopted, 2019 was not scored and
stays unopened** for the next change that passes the dev rule.

## Round 2b: scaled metadata without counsel names, and a relabelling attempt

Pre-registered after round 2 (protocol amendment "round 2b"); output in `local-round2b/` (the rerun reproduced every
round-2 number exactly).

| Candidate | Acc | Macro-F1 | Delta vs text | 95% interval |
|---|---|---|---|---|
| text + metadata (no counsel, block L2-normalised x 0.2) | 0.476 | 0.422 | +0.013 | [-0.016, +0.042] |

The weight was chosen on the inner split: 0.383 / 0.388 / 0.380 / 0.372 / 0.369 for 0.1 / 0.2 / 0.35 / 0.5 / 0.75, against
0.382 for text alone. **Not adopted**: the gain is below 0.03 and the interval includes 0. Scaled properly, metadata no
longer hurts and may add about 0.01, which this dev set cannot distinguish from 0.

**Relabelling `disposed`, dropped before any model used it.** A regex rule over the operative closing passage
(mixed batch -> `disposed`; partial relief -> `partly allowed`; remand or set aside -> `allowed`; clear dismissal or
allowance) was written on the 40 audited cases. On a fresh sample of 40 training-year `disposed` cases, labelled by hand
blind before the rule's output was looked at, it agreed on 26 of 40. Of the 15 cases it moved, about 6 went to the wrong
label (mostly partial relief or mixed batches sent to `allowed`). That is too noisy to define a task on. The blind reading
put 20 of 40 as no clear winner or mixed, 11 as partial relief, 9 as allowed (mostly remands), and none as dismissed.

## What this means

- The 4-way text score is stable at about 0.41 on dev and 0.42 on 2018. Metadata as built here and recency weighting
  do not move it.
- Scaled metadata gives +0.013 on dev, which is not significant. Every model-side lever tried so far lands within about
  0.015 of the text baseline.
- The remaining lever is the labels: about half of `disposed` is partial relief or a remand. Regexes cannot relabel it
  reliably. It would need case-by-case annotation (by people, or an LLM reading only the operative paragraph, with a
  hand-checked sample). Such a task would be reported next to the 4-way numbers, never instead of them.
- 2019 stays unopened.
