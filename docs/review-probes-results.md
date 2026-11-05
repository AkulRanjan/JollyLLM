# Review probes: linear evidence for the review of 2026-10-21

Run 2026-10-30 locally (CPU) from `kaggle-notebooks/jolly-review-probes`; output in
`checkpoints/kaggle/jolly-2026-10-29/local-review/` (git-ignored). Fixed in protocol v2, amendment "review probes", before
any result. Dev only (2015, 683 cases). 2017 and 2018 were read for exclusion counts only; 2019 was not opened.
These are linear stand-ins. **They do not replace LLM seeds or an LLM randomised-graph control, neither of which has run.**

Reproduction check: retained counts 731 / 699 / 683 (2013-15), 558 (2017), 645 (2018) match the paper and Phase 5; TF-IDF
0.395 (2013-14) and 0.409 (2005-14) match Phase 1 and round 2.

## Statute and graph features (decision rule: +0.03, interval excluding 0)

Nothing adopted. Block weight 0.1 for both, the lowest on the grid: on the inner split every block scored below text
(0.382 against 0.367-0.376), so the weight limits the harm and the effects below are small partly for that reason.

| Features | 2005-14 macro-F1 | Delta vs T | 2013-14 macro-F1 | Delta vs T |
|---|---|---|---|---|
| T: TF-IDF | 0.409 | | 0.395 | |
| S: cited provisions only | 0.301 | | 0.272 | |
| G: provisions + co-citation neighbours | 0.271 | | 0.230 | |
| T+S | 0.403 | -0.006 [-0.014, 0.000] | 0.395 | 0.000 [-0.003, +0.004] |
| T+G | 0.403 | -0.006 [-0.013, 0.000] | 0.390 | -0.005 [-0.012, +0.001] |
| T+R: randomised statutes (mean of 5, SD) | 0.405 (0.003) | | 0.391 (0.002) | |
| T+H: shuffled graph (mean of 5, SD) | 0.406 (0.002) | | 0.395 (0.001) | |

T+G minus first-draw T+R: +0.003 [-0.006, +0.011] and -0.001 [-0.006, +0.005]. T+G minus T+H: -0.002 [-0.009, +0.005] and
-0.005 [-0.012, +0.001]. T+G minus T+S: 0.000 [-0.005, +0.006] and -0.006 [-0.012, +0.001].

Cited statutes do carry signal alone (0.30 against the 0.149 floor), but it adds nothing to text, which already holds the
same citations as tokens. The graph's neighbours add nothing to the statutes, and random statutes or shuffled edges do as well.

## Subgroups (2005-14 training)

| Subset | n | T acc / mF1 | T+G acc / mF1 | Delta |
|---|---|---|---|---|
| With subgraph | 261 | 0.517 / 0.482 | 0.513 / 0.476 | -0.005 [-0.020, +0.005] |
| Without subgraph | 422 | 0.417 / 0.338 | 0.410 / 0.332 | -0.007 [-0.015, 0.000] |

Coverage on dev is 38.2%. Cases that cite a provision are easier for plain text too, so the subgroups differ by more than
the graph: comparing them across subsets says nothing about the graph.

## Exclusions

- Across 12,134 extracted judgments (2005-2015, 2017, 2018), 10,334 (85.2%) are retained: 24 duplicates, 627 no four-way
  disposal, 528 no disposition phrase, 202 text-official disagreement, 419 no usable view. Per-year table in the JSON.
- Retention is 76-90%. It is lowest in 2009 (75.9%, 206 no usable view) and 2017 (76.2%, 89 no phrase, 69 other disposal).
- Rejected cases with a four-way label, pool plus dev: disposed 266 of 576 (46%), against 16% of retained cases. The filter
  under-represents `disposed`.
- Dev 2015: 33 rejected cases score acc 0.303 / macro-F1 0.233 against 0.455 / 0.409 for retained. The union (716) scores
  0.448 / 0.402, so including them costs 0.007. With 33 cases this is rough. Adding 543 rejected cases to training moved
  retained dev macro-F1 from 0.409 to 0.419, with no interval computed.
- The 627 "other disposal" cases have no four-way label and cannot be scored.

## Stability, time, validation size

- Ten resamples of the 2013-14 training cases: T+G - T gap mean +0.001, SD 0.005, positive in 5 of 10. Each model's own
  macro-F1 spans about 0.30-0.36. A linear fit has no seed, so this says nothing about LLM seed variance directly.
- One text pipeline, train 2005-2009: 2010-2015 macro-F1 0.376, 0.370, 0.395, 0.346, 0.368, 0.399, no clear trend. Rolling
  two-year training: 0.377, 0.363, 0.359 at 1, 2, 3 years ahead (SD over years about 0.02). Decay inside one pipeline is
  about 0.02 over two years, well below the -0.090 between SC-2016 and 2017 graph-text gaps.
- 46-case draws from dev: macro-F1 SD 0.093, 95% range 0.232-0.584 for a model with 0.409 on all 683. A true gap of 0.014
  between two models reverses in 47% of draws; the delta's 95% range is -0.196 to +0.224.
