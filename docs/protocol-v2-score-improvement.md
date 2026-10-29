# Evaluation protocol v2 (score-improvement work)

Written 2026-10-29, before any Phase 1 result was seen. It supersedes nothing: the protocol that produced the
2016/2017 test numbers stays as it was, and those numbers are never recomputed. This protocol governs only the work in
`docs/score-improvement-plan.md`. Any later change is added under "Amendments" with its date and the reason; the
sections above it are not edited.

## 1. Data windows

| Role | Decision years | Source | Rule |
| --- | --- | --- | --- |
| Development set | 2015 (about 680 cases) | AWS bucket | All tuning and all probe decisions are made here |
| Development training | 2013-2014 | AWS bucket | Reproduces the earlier probe (1,430 / 683 cases) |
| Wider training pool | 2005-2014 | AWS bucket | Used by the data-scale probe and any later training |
| Retired test sets | SC-2016 test (62), 2017 (568) | existing | Already scored once. Never re-scored. May be used as training data only for the final model, which is then tested on 2018 |
| Fresh test set | 2018 | AWS bucket | Extracted, never labelled or inspected until Phase 5. Scored once |
| Not taken | 2016 | AWS bucket | The SC-2016 year comes from its own source |

Training years are strictly earlier than the year they are scored on. The extraction notebook
(`kaggle-notebooks/jolly-aws-extract`) prints counts only for 2017 and 2018, and the probe notebook
(`kaggle-notebooks/jolly-phase1-probes`) never opens those files.

## 2. Case selection

Unchanged from data v0.3: a case is kept when the official "Disposal Nature" maps to one of the four labels and the
text rule (`infer_disposition` on the last 20% of the text) reads the same label. Input view is
`body-prefix-masked-v2` at 6,000 characters unless a probe changes it. Cases without a usable body view are dropped.
Archive hashes are pinned in `configs/sources/aws-sc-source.json`; years added by this work get their hashes added
there from the extraction notebook's verify table before any training run uses them.

## 3. Schemas, always reported side by side

| Schema | Definition |
| --- | --- |
| 4-way (primary, comparable to all earlier numbers) | allowed, dismissed, disposed, partly allowed |
| 3-way | partly allowed merged into allowed |
| Hierarchy | disposed versus contested first, then allowed versus dismissed |
| Binary | allowed versus dismissed, disposed dropped (reported with its own always-allowed floor) |

A change of schema never replaces the 4-way numbers. Numbers under different schemas are not compared with each other.

## 4. Decision rule

A change is adopted only if dev macro-F1 improves by at least **0.03** and the 95% interval of a paired bootstrap
(2,000 resamples of dev cases in probes, 10,000 on the final test) excludes 0. Anything else is reported as "no
detectable effect". The rule is applied to the 4-way schema unless a probe is explicitly about a different schema, in
which case the probe states which schema its comparison uses.

## 5. Leakage rule for input views

A new input view is accepted only if (a) the share of views containing a literal disposition phrase after masking is
reported, (b) the score with residual outcome words (allow*, dismiss*, dispos*, set aside, quash*, uphold*, affirm*,
remand*) removed is reported next to the score with them, and (c) the gain survives (b). A large jump that disappears
under (b) is discarded.

## 6. Calibration

Class-offset tuning uses out-of-fold predictions inside the training years only. The dev set is scored once with the
chosen offsets. Tuning offsets on dev itself is not allowed.

## 7. Test

Phase 5 freezes the chosen text and graph runs (`freeze-outcome`), then scores 2018 once with
`evaluate-outcome`. All four schemas and the old 4-way numbers are reported together. If the 2018 labels are opened
earlier for any reason, the 2018 set is spent and this document must say so under Amendments.

## Amendments

- **2026-10-29, final system fixed before any 2018 label was opened.** Phase 3 failed its gate (`docs/phase3-results.md`),
  so Phase 4 is skipped and no graph run is frozen. The system scored on 2018 is TF-IDF (1-2 grams, min_df 2, sublinear
  tf, max 300,000 features) + balanced logistic regression (C = 1), input view `body-prefix-masked-v2` at 6,000 characters,
  trained on every usable AWS case decided in 2005-2015 and 2017. One model per schema (4-way, 3-way flat, hierarchy,
  binary). Models are fitted and their hashes written to `freeze.json` before the 2018 labels are read
  (`kaggle-notebooks/jolly-phase5-test`). Reported: accuracy and macro-F1 per schema with 10,000-resample bootstrap
  intervals, the always-`allowed` floor, and the paired gain over that floor. After this run the 2018 set is spent.
- **2026-10-29, 2018 scored once** (`docs/phase5-test-results.md`). The 2018 set is spent; any later work needs a new
  untouched year (2019 onward).
- **2026-10-29, round 2 fixed before any of its results were seen.** New fresh test year: **2019**, extracted by
  `vanshag08/jolly-aws-extract-2019` (2019 and 2020; counts only, no labels). 2020 is kept unopened as a spare. Probes on
  the same dev split (`kaggle-notebooks/jolly-round2-probes`), each compared with TF-IDF + LR trained on 2005-2014 by the
  section 4 rule:
  (a) **header metadata** - case type and filing-year age from the case number, bench size, judges named in the header,
  coarse party type of each side of the title (state/government, tax or customs authority, insurer, bank, company,
  other body, individual) and the court appealed from. Only the header before the headnote and the "From the Judgment
  and Order ... of <court>" line are read; the headnote, page count and anything after the decision are not used.
  (b) **recency weighting** - training cases weighted by 0.5^((last training year - year) / h); h is chosen on an inner
  split (train 2005-2013, score 2014) from {1, 2, 4, 8, none} and then applied once to the dev set.
  (c) (a) and (b) together. A label audit of `disposed` and `partly allowed` reads training years (2005-2014) only.
  Added the same day, still before any round-2 result: (a) also uses the counsel named "for the Appellant(s)" and "for
  the Respondent(s)" in the header and whether connected matters are listed ("WITH ...").
- **2026-10-29, round 2 result** (`docs/round2-results.md`): nothing adopted, so 2019 was not scored and stays unopened.
- **2026-10-29, round 2b, fixed before its result was seen.** One more metadata probe, motivated by the round-2 failure:
  counsel tokens are dropped; the metadata block is L2-normalised per case and multiplied by a weight w chosen on the
  inner split (train 2005-2013, score 2014) from {0.1, 0.2, 0.35, 0.5, 0.75}; no recency weighting. Compared once on dev
  with text TF-IDF trained on 2005-2014 under the section 4 rule. A rule-based relabelling of `disposed` was tried and
  dropped before any model used it: on 40 fresh training-year cases it agreed with a blind hand reading on 26.
- **2026-10-29, round 2b result**: +0.013 [-0.016, +0.042] on dev, not adopted (`docs/round2-results.md`). 2019 stays
  unopened.
