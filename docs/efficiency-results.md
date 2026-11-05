# Efficiency of the statute-graph prefix (CPU, 2026-11-03)

Scripts: `scripts/benchmark_efficiency.py` (retrieval, graph index, encoder, token counts, analytic cost) and
`scripts/benchmark_prefill.py` (controlled CPU prefill). Raw numbers: `docs/efficiency-results.json`.
Inputs: judgments of 2013-2015 only (graph window 2013-2014, queries are the 683 development views of 2015). No model was
scored, and 2017-2019 were not opened. No GPU was available, so nothing here is a GPU latency or GPU memory figure.

| Quantity | Graph prefix | Text-based alternative |
|---|---|---|
| Added prompt tokens | 8 | BM25 evidence 384 (cap, reached by every query); subgraph as labels 210; with 48 tokens of text per node 1,424; labels and edges 5,281 |
| Prefill computation, Qwen3-8B, 1,536-token view (analytic) | +0.5% | +26% (BM25 appended), +14% / +98% / +392% for the three text renderings |
| Measured CPU prefill, Qwen3-0.6B, min of 6 interleaved rounds | 1.011 | 1.119 (labels), 1.919 (node texts); noise control 0.998 |
| Retrieval, median per query | 4.5 ms over all queries (61.6% have no seed), 21.0 ms for the 262 non-empty | BM25 44.3 ms (three acts, 1,222 provisions, so a lower bound) |
| Encoder forward, non-empty subgraph | 30.8 ms (16 threads), 258 ms (1 thread); 0.005 ms for null tokens | none |
| Expected graph path per query | about 23 ms (10.9 retrieval + 0.384 x 30.7 encoding) | 44 ms |
| Index on disk | 1.22 MB pickled graph with 1,430 case nodes (26.8 kB compressed edge list) | 1.70 MB BM25 index, three acts |
| Trainable parameters | encoder 6,340,097 (+14.5% over 43,646,976 LoRA); matches the paper's count | |
| Encoder weights | 25.4 MB fp32, 12.7 MB fp16; node feature table for 2,619 nodes 42.9 MB fp32 | |
| Training time per epoch (from the paper) | 7.43 h | text model 7.38 h (+0.7%) |

## What this does and does not show

- The graph prefix is **not cheaper than text-only prompting**. It adds tokens, parameters and an encoder pass.
- It is the **cheapest way to carry retrieved context into the prompt** among those compared, and on CPU its expected
  retrieval-plus-encoding cost is about half that of BM25. A repeat run moved the medians by about 20% (BM25 35 ms,
  graph 5 ms), so only the ordering is firm.
- The first prefill measurement ran each variant once and gave 8.9 s and 19.7 s for two identical 1,536-token prompts; it was
  discarded. The controlled version interleaves variants and includes a duplicate: its minimum over six rounds is within 0.2% of
  the original, while individual rounds still vary by up to a factor of 1.9, so the medians are not reported.
- Not measured: Qwen3-8B on a GPU, peak GPU memory, the trained adapter files (on a private Kaggle account), the six-act statute
  layer (BNS, BNSS and CrPC need PDFs that are not stored locally), and the full training window.

## Non-inferiority of graph to text (from the paper's reported paired intervals, no new scoring)

Margin 0.05 macro-F1, chosen after the intervals were seen. Non-inferior on 2017 (lower bound -0.028) and equivalent on the
pooled 608 cases (-0.034 to +0.046). Not shown for accuracy (lower bounds -0.065 and -0.071) or on the 50 SC-2016 cases.
Single training run per system, so this is support and not proof. The same margin is proposed as the prespecified test for
the matched-seed run.
