# Phase 3 results: neural baselines against TF-IDF

Run 2026-10-29 on a Kaggle T4 as `vanshag08/jolly-phase3-neural` (6,079 s, about 1.7 GPU-h; a CPU smoke run,
`jolly-phase3-neural-smoke`, checked the code first). Same cases, split, view and gate as Phase 1
(`docs/phase1-results.md`). Dev only (2015, 683 cases); no year after 2015 was opened.

TF-IDF reference: train 2013-14 macro-F1 0.395, train 2005-14 macro-F1 0.415.

| Candidate | Train | Macro-F1 | Delta vs TF-IDF | 95% interval |
|---|---|---|---|---|
| InLegalBERT frozen + LR | 2013-14 | 0.316 | -0.079 | [-0.135, -0.022] |
| InLegalBERT frozen + LR | 2005-14 | 0.314 | -0.101 | [-0.146, -0.051] |
| Qwen3-Embedding-0.6B frozen + LR | 2013-14 | 0.268 | -0.127 | [-0.181, -0.074] |
| Qwen3-Embedding-0.6B frozen + LR | 2005-14 | 0.274 | -0.140 | [-0.193, -0.084] |
| InLegalBERT fine-tuned, seeds 1 / 2 | 2013-14 | 0.312 / 0.324 | -0.083 / -0.071 | both below 0 |
| InLegalBERT fine-tuned, seeds 1 / 2 | 2005-14 | 0.350 / 0.387 | -0.065 / -0.028 | seed 1 below 0 |
| best ensemble: fine-tuned (2005-14, seed 2) + TF-IDF | 2005-14 | 0.419 | +0.004 | [-0.035, +0.046] |

**Gate not passed.** No candidate beats TF-IDF by 0.03; most are significantly worse. Averaging with TF-IDF adds nothing
detectable.

Notes:
- Fine-tuning is still improving at epoch 3 (train loss 1.14-1.28; 2005-14 seed 2 went 0.287 -> 0.345 -> 0.387), so longer
  training might close the gap, but there is no sign it would clear TF-IDF + 0.03.
- InLegalBERT reads only the first 512 tokens (about a third of the 6,000-character view); TF-IDF reads all of it.
- Qwen3-Embedding cost 58 of the 101 minutes and was the weakest candidate.

Conclusion under the plan: the lever is the data and the task design, not model capacity. Per the Phase 3 gate, the
expensive LLM runs of Phase 4 are not justified by this evidence.
