# SaulLM-7B probe: frozen embeddings against TF-IDF

Run 2026-10-29/30 on a Kaggle T4 as `vanshag08/jolly-saullm-probe` (16,576 s, about 4.6 GPU-h; a CPU smoke run,
`jolly-saullm-probe-smoke`, checked the code first). Fixed in protocol v2, amendment "SaulLM probe", before any result.
Dev only (2015, 683 cases); no year after 2015 was opened. Output in `checkpoints/kaggle/jolly-2026-10-29/saullm/`
(git-ignored), including the embeddings.

Model: `Equall/Saul-7B-Base`, frozen, 4-bit NF4, first 1,536 tokens of the `body-prefix-masked-v2` view, mean-pooled
hidden states. Embedding took about 1.3-1.9 s per case. The 3-hour budget covered 2008-2014, so the wide setting is
2008-14 rather than 2005-14. Layer and C were chosen by 5-fold out-of-fold macro-F1 inside the training years: layer 24,
C = 0.01 in both settings.

| Train | Candidate | Acc | Macro-F1 | Delta vs TF-IDF | 95% interval |
|---|---|---|---|---|---|
| 2013-14 (1,430) | TF-IDF + LR | 0.499 | 0.395 | | |
| | SaulLM + LR | 0.438 | 0.345 | -0.050 | [-0.106, +0.006] |
| | SaulLM averaged with TF-IDF | 0.451 | 0.348 | -0.047 | [-0.100, +0.006] |
| 2008-14 (5,826) | TF-IDF + LR | 0.467 | 0.415 | | |
| | SaulLM + LR | 0.407 | 0.352 | -0.064 | [-0.113, -0.013] |
| | SaulLM averaged with TF-IDF | 0.435 | 0.365 | -0.051 | [-0.096, -0.004] |

**Not adopted.** SaulLM is below TF-IDF in every setting, significantly so with the wider training set. It is the best
frozen-embedding model so far (InLegalBERT 0.314-0.316, Qwen3-Embedding 0.268-0.274 in Phase 3), but still about 0.05-0.06
below the bag of words. Averaging with TF-IDF does not help; the embedding head's probabilities are more confident and
dominate the average. Per the protocol, QLoRA fine-tuning of SaulLM is not justified by this result. 2019 stays unopened.

Caveats: one run, one pooling method (mean), 4-bit weights; SaulLM's legal pre-training is mostly US, UK and EU text, not
Indian judgments.
