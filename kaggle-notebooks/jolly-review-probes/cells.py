# %% [markdown]
# # Jolly review probes (CPU)
#
# Linear evidence for the review of 2026-10-21 (protocol v2, amendment "review probes"). These do not replace LLM seeds;
# they ask what can be asked without a GPU:
#
# * **(a)** how many cases each selection step removes, and whether the removed cases are harder;
# * **(b)** whether cited statutes, and their co-citation neighbours, add to TF-IDF;
# * **(c)** whether a randomised statute set or shuffled graph does as well;
# * **(d)** the same split by whether a case has a usable subgraph;
# * **(e)** how much the answer moves over resampled training sets;
# * **(f)** how fast scores decay over time inside one text pipeline;
# * **(g)** what a 46-case validation set can resolve.
#
# Dev split: 2015. No model is scored on 2017, 2018 or 2019; 2017 and 2018 are read for exclusion counts only.

# %% settings
import json, os, re, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
from legal_graph.aws_judgments import official_label, read_aws_records
from legal_graph.extraction import extract_indian_references_v2
from legal_graph.outcomes import LABELS, build_input_view_v2, infer_disposition

DEV_YEAR = 2015
POOL_YEARS = list(range(2005, 2015))
COUNT_ONLY_YEARS = [2017, 2018]
BASE_CHARS, BUDGET = 6000, 32
MIN_GAIN, BOOTSTRAPS, SEED = 0.03, 2000, 20261030
WEIGHTS, CONTROL_DRAWS, RESAMPLES, VAL_SIZE, VAL_DRAWS = [0.1, 0.2, 0.35, 0.5], 5, 10, 46, 5000
OUT = Path("/kaggle/working/review") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
labels4 = list(LABELS)
label_index = {label: i for i, label in enumerate(labels4)}
results = {}

# %% [markdown]
# ## 1. Load the cases and count the selection funnel
# The steps follow `_skip_reason`, with a usable-view check at the end. Cases from 2017 and 2018 are only counted.

# %% load
def find_record_files():
    root = Path(os.environ["JOLLY_RECORDS"]) if not ON_KAGGLE else Path("/kaggle/input")
    return [f for f in sorted(root.rglob("records*.jsonl.gz")) if not re.search(r"(2016|2019|202\d)", f.name)]

def provisions(text, document_id):
    return sorted({c.canonical_hint for c in extract_indian_references_v2(text, document_id) if c.canonical_hint.startswith("provision:")})

record_files = find_record_files()
assert record_files, "attach the jolly-aws-extract kernel (Kaggle) or set JOLLY_RECORDS (local)"
funnel, rows, seen = Counter(), [], set()
for path in record_files:
    for record in read_aws_records(path):
        decided = str(record.get("decision_date") or "")[:4]
        if not decided.isdigit() or int(decided) not in POOL_YEARS + [DEV_YEAR] + COUNT_ONLY_YEARS:
            funnel["other_year", "all"] += 1
            continue
        year = int(decided)
        funnel["extracted", year] += 1
        if record["path"].lower() in seen:
            funnel["duplicate_report", year] += 1
            continue
        seen.add(record["path"].lower())
        text, label = str(record.get("text") or ""), official_label(str(record.get("disposal_nature", "")))
        if not text.strip():
            funnel["no_text", year] += 1
            continue
        if label is None:
            funnel["no_four_way_disposal", year] += 1
            continue
        inferred = infer_disposition(text[int(len(text) * 0.8):])
        status = "kept" if inferred == label else "no_phrase" if inferred is None else "disagrees"
        view = build_input_view_v2(text, BASE_CHARS)
        if status != "kept":
            funnel[{"no_phrase": "no_disposition_phrase", "disagrees": "text_disagrees"}[status], year] += 1
        elif view is None:
            funnel["no_usable_view", year] += 1
        else:
            funnel["retained", year] += 1
        if year in COUNT_ONLY_YEARS or view is None:
            continue
        rows.append({"path": record["path"], "year": year, "label": label, "view": view, "status": status,
                     "cites": provisions(text, record["path"]) if status == "kept" else [],
                     "seeds": provisions(view, record["path"])})
cases = pd.DataFrame(rows)
kept = cases[cases.status == "kept"].reset_index(drop=True)
print(cases.groupby(["status"]).size().to_dict())

steps = ["extracted", "duplicate_report", "no_text", "no_four_way_disposal", "no_disposition_phrase", "text_disagrees",
         "no_usable_view", "retained"]
years = POOL_YEARS + [DEV_YEAR] + COUNT_ONLY_YEARS
table = pd.DataFrame({y: {s: funnel[s, y] for s in steps} for y in years}).T
table.loc["all"] = table.sum()
table["retained_pct"] = (100 * table.retained / table.extracted).round(1)
print(table.to_string())
results["funnel"] = table.reset_index().rename(columns={"index": "year"}).astype({"year": str}).to_dict("records")
results["excluded_labels"] = {s: cases[cases.status == s].label.value_counts().to_dict() for s in ("kept", "no_phrase", "disagrees")}

# %% [markdown]
# ## 2. Models and statistics

# %% helpers
def macro_f1(y, pred):
    """y and pred are integer arrays. Classes absent from both score 0, as sklearn does with zero_division=0."""
    n = len(labels4)
    matrix = np.bincount(n * y + pred, minlength=n * n).reshape(n, n)
    tp, gold, predicted = np.diag(matrix), matrix.sum(1), matrix.sum(0)
    return float(np.mean(2 * tp / np.maximum(gold + predicted, 1)))

def per_class(y, pred):
    n = len(labels4)
    matrix = np.bincount(n * y + pred, minlength=n * n).reshape(n, n)
    tp, gold, predicted = np.diag(matrix), matrix.sum(1), matrix.sum(0)
    return {"confusion": matrix.tolist(),
            "precision": (tp / np.maximum(predicted, 1)).round(3).tolist(),
            "recall": (tp / np.maximum(gold, 1)).round(3).tolist(),
            "f1": (2 * tp / np.maximum(gold + predicted, 1)).round(3).tolist()}

def score(y, pred):
    return {"n": len(y), "acc": round(float((y == pred).mean()), 3), "macro_f1": round(macro_f1(y, pred), 3)}

def paired(y, a, b, n=BOOTSTRAPS):
    generator = np.random.default_rng(SEED)
    deltas = [macro_f1(y[p], a[p]) - macro_f1(y[p], b[p]) for p in (generator.integers(0, len(y), len(y)) for _ in range(n))]
    low, high = np.percentile(deltas, [2.5, 97.5])
    delta = macro_f1(y, a) - macro_f1(y, b)
    return {"delta": round(delta, 3), "low": round(float(low), 3), "high": round(float(high), 3),
            "adopt": bool(delta >= MIN_GAIN and low > 0)}

def targets(frame):
    return frame.label.map(label_index).values

def fit_predict(x_train, y_train, x_eval):
    return LogisticRegression(C=1, class_weight="balanced", max_iter=3000).fit(x_train, y_train).predict(x_eval)

def vectors(train, evaluate):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000, dtype=np.float32)
    return tfidf.fit_transform(train.view), tfidf.transform(evaluate.view)

# %% [markdown]
# ## 3. Time-safe co-citation selector
# The same ranking as the paper's selector, on matrices. A query from year Y sees only window cases from before Y, so
# no case can select through itself, and validation or later cases never contribute. Seeds come from the view; the
# citation matrix comes from the full text of the training cases.

# %% selector
vocabulary = sorted({p for items in kept.cites for p in items} | {p for items in kept.seeds for p in items})
vocab_index = {p: i for i, p in enumerate(vocabulary)}

def incidence(lists):
    row = [i for i, items in enumerate(lists) for p in items if p in vocab_index]
    col = [vocab_index[p] for items in lists for p in items if p in vocab_index]
    return csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(len(lists), len(vocabulary)))

def select(queries, window, seeds=None, permutation=None, draw=None):
    """Binary query x provision matrix of seeds plus the top co-cited candidates, up to the node budget."""
    citations = incidence(window.cites.tolist())
    if permutation is not None:
        citations = citations[:, permutation]
    seed_matrix = incidence(queries.seeds.tolist()) if seeds is None else seeds
    chosen = lil_matrix((len(queries), len(vocabulary)), dtype=np.float32)
    for year in sorted(queries.year.unique()):
        members = np.where(queries.year.values == year)[0]
        earlier = citations[window.year.values < year]
        if earlier.shape[0] == 0:
            continue
        frequency = np.asarray(earlier.sum(0)).ravel()
        eligible = csr_matrix((frequency > 0).astype(np.float32))
        co_cited = ((earlier.T @ earlier) > 0).astype(np.float32).tocsr()
        for start in range(0, len(members), 256):
            batch = members[start:start + 256]
            sources = seed_matrix[batch].multiply(eligible).tocsr()
            if draw is not None:
                dense = np.zeros(sources.shape, dtype=np.float32)
                weights = frequency / frequency.sum()
                for i, k in enumerate(np.asarray(sources.sum(1)).ravel().astype(int)):
                    if k:
                        dense[i, draw.choice(len(frequency), min(k, int((frequency > 0).sum())), replace=False, p=weights)] = 1
                sources = csr_matrix(dense)
            distinct = np.asarray((sources @ co_cited).todense(), dtype=np.float64)
            touching = ((earlier @ sources.T) > 0).astype(np.float32)
            support = np.asarray((earlier.T @ touching).todense(), dtype=np.float64).T
            rank = distinct * 1e6 + support
            own = np.asarray(sources.todense()) > 0
            rank[own] = 0
            for i, member in enumerate(batch):
                seeded = np.where(own[i])[0][:BUDGET]
                if len(seeded) == 0:
                    continue
                room = BUDGET - len(seeded)
                order = np.argsort(-rank[i], kind="stable")[:room]
                chosen[member, np.concatenate([seeded, order[rank[i][order] > 0]])] = 1
    return chosen.tocsr()

def seeds_only(queries, window):
    """Seeds that occur in earlier window cases, the same rule that decides whether a query has a subgraph."""
    citations = incidence(window.cites.tolist())
    seed_matrix = incidence(queries.seeds.tolist())
    kept_rows = lil_matrix(seed_matrix.shape, dtype=np.float32)
    for year in sorted(queries.year.unique()):
        members = np.where(queries.year.values == year)[0]
        frequency = np.asarray(citations[window.year.values < year].sum(0)).ravel()
        kept_rows[members] = seed_matrix[members].multiply(csr_matrix((frequency > 0).astype(np.float32)))
    return kept_rows.tocsr()

def join(text, block, weight):
    return hstack([text, normalize(block) * weight]).tocsr()

# %% [markdown]
# ## 4. Choose the block weights on the inner split (train 2005-2013, score 2014)

# %% inner
inner_train = kept[kept.year <= 2013].reset_index(drop=True)
inner_eval = kept[kept.year == 2014].reset_index(drop=True)
x_inner, x_inner_eval = vectors(inner_train, inner_eval)
y_inner, y_inner_eval = targets(inner_train), targets(inner_eval)
inner_s = seeds_only(inner_train, inner_train), seeds_only(inner_eval, inner_train)
inner_g = select(inner_train, inner_train), select(inner_eval, inner_train)
inner = {"text": score(y_inner_eval, fit_predict(x_inner, y_inner, x_inner_eval))}
for name, (block_train, block_eval) in {"S": inner_s, "G": inner_g}.items():
    inner[name] = {w: score(y_inner_eval, fit_predict(join(x_inner, block_train, w), y_inner, join(x_inner_eval, block_eval, w)))["macro_f1"]
                   for w in WEIGHTS}
weight_s = max(WEIGHTS, key=lambda w: inner["S"][w])
weight_g = max(WEIGHTS, key=lambda w: inner["G"][w])
print("inner", inner, "chosen", weight_s, weight_g)
results["inner"] = {"scores": {k: {str(w): s for w, s in v.items()} if k != "text" else v for k, v in inner.items()},
                    "weight_s": weight_s, "weight_g": weight_g}

# %% [markdown]
# ## 5. Statute features, graph features and controls on dev 2015

# %% window
dev = kept[kept.year == DEV_YEAR].reset_index(drop=True)
y_dev = targets(dev)

def evaluate_window(train):
    y_train = targets(train)
    x_train, x_dev = vectors(train, dev)
    s_train, s_dev = seeds_only(train, train), seeds_only(dev, train)
    g_train, g_dev = select(train, train), select(dev, train)
    predictions = {
        "T": fit_predict(x_train, y_train, x_dev),
        "S": fit_predict(s_train, y_train, s_dev),
        "G": fit_predict(g_train, y_train, g_dev),
        "T+S": fit_predict(join(x_train, s_train, weight_s), y_train, join(x_dev, s_dev, weight_s)),
        "T+G": fit_predict(join(x_train, g_train, weight_g), y_train, join(x_dev, g_dev, weight_g)),
    }
    output = {"coverage_train": round(float((s_train.sum(1) > 0).mean()), 3), "coverage_dev": round(float((s_dev.sum(1) > 0).mean()), 3),
              "mean_nodes_dev": round(float(np.asarray(g_dev.sum(1)).ravel()[np.asarray(s_dev.sum(1)).ravel() > 0].mean()), 1),
              "scores": {k: score(y_dev, p) for k, p in predictions.items()},
              "vs_T": {k: paired(y_dev, predictions[k], predictions["T"]) for k in ("T+S", "T+G")},
              "TG_vs_TS": paired(y_dev, predictions["T+G"], predictions["T+S"])}
    controls = {"R": [], "H": []}
    for index in range(CONTROL_DRAWS):
        generator = np.random.default_rng(SEED + index)
        r_train = select(train, train, draw=generator)
        r_dev = select(dev, train, draw=generator)
        permutation = generator.permutation(len(vocabulary))
        h_train, h_dev = select(train, train, permutation=permutation), select(dev, train, permutation=permutation)
        for name, block_train, block_dev in (("R", r_train, r_dev), ("H", h_train, h_dev)):
            controls[name].append(fit_predict(join(x_train, block_train, weight_g), y_train, join(x_dev, block_dev, weight_g)))
    for name, draws in controls.items():
        values = [macro_f1(y_dev, p) for p in draws]
        output[f"T+{name}"] = {"macro_f1_each": [round(v, 3) for v in values], "mean": round(float(np.mean(values)), 3),
                               "sd": round(float(np.std(values, ddof=1)), 3),
                               "T+G_vs_first_draw": paired(y_dev, predictions["T+G"], draws[0]),
                               "first_draw_vs_T": paired(y_dev, draws[0], predictions["T"])}
    with_seed = np.asarray(s_dev.sum(1)).ravel() > 0
    groups = {}
    for name, mask in (("with_subgraph", with_seed), ("without_subgraph", ~with_seed)):
        groups[name] = {"n": int(mask.sum()), "labels": dev.label[mask].value_counts().to_dict(),
                        "T": score(y_dev[mask], predictions["T"][mask]), "T+G": score(y_dev[mask], predictions["T+G"][mask]),
                        "T+G_vs_T": paired(y_dev[mask], predictions["T+G"][mask], predictions["T"][mask])}
    output["subgroups"] = groups
    return output, predictions

wide_train = kept[kept.year.isin(POOL_YEARS)].reset_index(drop=True)
narrow_train = kept[kept.year.isin([2013, 2014])].reset_index(drop=True)
results["wide"], wide_predictions = evaluate_window(wide_train)
print("wide", json.dumps(results["wide"], indent=1)[:2500])
results["narrow"], narrow_predictions = evaluate_window(narrow_train)
print("narrow", json.dumps(results["narrow"], indent=1)[:2500])
results["per_class_T_wide"] = per_class(y_dev, wide_predictions["T"])
results["per_class_TG_wide"] = per_class(y_dev, wide_predictions["T+G"])

# %% [markdown]
# ## 6. Are the excluded cases harder?
# Dev cases with a four-way official label and a usable view that the text rule rejected, scored by the model trained on
# retained cases; then the same with such training cases included (labels are the official ones).

# %% exclusions
rejected = cases[(cases.year == DEV_YEAR) & (cases.status != "kept")].reset_index(drop=True)
pool_rejected = cases[cases.year.isin(POOL_YEARS) & (cases.status != "kept")].reset_index(drop=True)
x_train, x_dev = vectors(wide_train, pd.concat([dev, rejected], ignore_index=True))
union_y = np.concatenate([y_dev, targets(rejected)])
union_pred = fit_predict(x_train, targets(wide_train), x_dev)
retained_mask = np.arange(len(union_y)) < len(dev)
widened = pd.concat([wide_train, pool_rejected], ignore_index=True)
x_wider, x_union = vectors(widened, pd.concat([dev, rejected], ignore_index=True))
wider_pred = fit_predict(x_wider, targets(widened), x_union)
results["exclusion_sensitivity"] = {
    "retained_cases": score(union_y[retained_mask], union_pred[retained_mask]),
    "rejected_cases": score(union_y[~retained_mask], union_pred[~retained_mask]),
    "union": score(union_y, union_pred),
    "train_with_rejected": {"retained_cases": score(union_y[retained_mask], wider_pred[retained_mask]),
                            "rejected_cases": score(union_y[~retained_mask], wider_pred[~retained_mask]),
                            "union": score(union_y, wider_pred)},
    "training_rejected_added": len(pool_rejected),
}
print(json.dumps(results["exclusion_sensitivity"], indent=1))

# %% [markdown]
# ## 7. Training-sample stability
# Each resample redraws the 2013-14 training cases with replacement. A linear fit has no random seed of its own, so
# this shows how far the data alone move the T+G - T gap. It is a lower bound on LLM seed variance, not a measure of it.

# %% stability
y_narrow = targets(narrow_train)
gaps = []
generator = np.random.default_rng(SEED)
for _ in range(RESAMPLES):
    picked = generator.integers(0, len(narrow_train), len(narrow_train))
    sample = narrow_train.iloc[picked].reset_index(drop=True)
    x_sample, x_sample_dev = vectors(sample, dev)
    block_train, block_dev = select(sample, narrow_train), select(dev, narrow_train)
    plain = fit_predict(x_sample, targets(sample), x_sample_dev)
    graph = fit_predict(join(x_sample, block_train, weight_g), targets(sample), join(x_sample_dev, block_dev, weight_g))
    gaps.append((macro_f1(y_dev, plain), macro_f1(y_dev, graph)))
gaps = np.array(gaps)
results["stability"] = {"T": gaps[:, 0].round(3).tolist(), "T+G": gaps[:, 1].round(3).tolist(),
                        "gap_mean": round(float((gaps[:, 1] - gaps[:, 0]).mean()), 3),
                        "gap_sd": round(float((gaps[:, 1] - gaps[:, 0]).std(ddof=1)), 3),
                        "gap_positive": int(((gaps[:, 1] - gaps[:, 0]) > 0).sum()), "resamples": RESAMPLES}
print(results["stability"])

# %% [markdown]
# ## 8. Temporal decay inside one text pipeline
# Every case here is PDF text, so a change over time cannot be a change of text pipeline.

# %% temporal
fixed = kept[kept.year <= 2009].reset_index(drop=True)
curve = {}
for year in range(2010, DEV_YEAR + 1):
    frame = kept[kept.year == year].reset_index(drop=True)
    x_train, x_eval = vectors(fixed, frame)
    curve[year] = score(targets(frame), fit_predict(x_train, targets(fixed), x_eval))
gap_scores = {1: [], 2: [], 3: []}
for gap, values in gap_scores.items():
    for year in range(2009, DEV_YEAR + 1):
        train = kept[kept.year.isin([year - gap - 1, year - gap])].reset_index(drop=True)
        frame = kept[kept.year == year].reset_index(drop=True)
        x_train, x_eval = vectors(train, frame)
        values.append(macro_f1(targets(frame), fit_predict(x_train, targets(train), x_eval)))
results["temporal"] = {"fixed_2005_2009": {str(k): v for k, v in curve.items()},
                       "rolling_two_year": {str(gap): {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3),
                                                       "each": [round(x, 3) for x in v]} for gap, v in gap_scores.items()}}
print(json.dumps(results["temporal"], indent=1))

# %% [markdown]
# ## 9. What 46 validation cases can resolve
# Two fixed models with a known difference on all 683 dev cases, compared on random draws of 46 of those cases.

# %% validation size
generator = np.random.default_rng(SEED)
better, weaker = wide_predictions["T"], narrow_predictions["T"]
if macro_f1(y_dev, better) < macro_f1(y_dev, weaker):
    better, weaker = weaker, better
draws = [generator.choice(len(y_dev), VAL_SIZE, replace=False) for _ in range(VAL_DRAWS)]
single = np.array([macro_f1(y_dev[p], better[p]) for p in draws])
delta = np.array([macro_f1(y_dev[p], better[p]) - macro_f1(y_dev[p], weaker[p]) for p in draws])
results["validation_size"] = {"full_dev_macro_f1": round(macro_f1(y_dev, better), 3), "full_dev_delta": round(macro_f1(y_dev, better) - macro_f1(y_dev, weaker), 3),
                              "single_sd": round(float(single.std(ddof=1)), 3),
                              "single_interval": [round(float(x), 3) for x in np.percentile(single, [2.5, 97.5])],
                              "delta_sd": round(float(delta.std(ddof=1)), 3),
                              "delta_interval": [round(float(x), 3) for x in np.percentile(delta, [2.5, 97.5])],
                              "order_flipped": round(float((delta < 0).mean()), 3), "tied": round(float((delta == 0).mean()), 3)}
print(results["validation_size"])

# %% save
(OUT / "review-results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print("wrote", OUT / "review-results.json")
