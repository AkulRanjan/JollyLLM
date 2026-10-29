# %% [markdown]
# # Jolly Phase 1: cheap TF-IDF probes on a large development set (CPU)
#
# Question behind each probe (`docs/score-improvement-plan.md`, Phase 1): how much of the score ceiling is label
# design, data scale, the input view or class calibration?
#
# * **Training and development years only.** Decision years 2005-2015 are loaded. 2017 (retired test) and 2018
#   (fresh test) are never read here.
# * **Dev set:** train on 2013-2014, develop on 2015, to reproduce the earlier probe (0.483 / 0.379 on 1,430 / 683 cases).
# * **Decision rule:** a change counts as an improvement only if dev macro-F1 rises by at least 0.03 and the paired
#   bootstrap 95% interval excludes 0.
# * Every number is dev-only. Nothing here is a test score.

# %% settings
import json, os, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
from legal_graph.aws_judgments import official_label, read_aws_records
from legal_graph.outcomes import LABELS, build_input_view_v2, infer_disposition

TRAIN_YEARS = [2013, 2014]
DEV_YEAR = 2015
POOL_YEARS = list(range(2005, 2015))  # everything before the dev year
ALL_YEARS = POOL_YEARS + [DEV_YEAR]
BASE_CHARS = 6000  # input view body-prefix-masked-v2, the one the 8B runs used
MIN_GAIN, BOOTSTRAPS, SEED = 0.03, 2000, 20261029
OUT = Path("/kaggle/working/phase1") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
results = {}

# %% [markdown]
# ## 1. Load the cases
# A case is kept when the official disposal value maps to one of the four labels **and** the project's text rule
# reads the same label from the last 20% of the judgment (the rule `select_aws_records` uses). Cases without a
# usable judgment-body view are dropped, as in the earlier runs. Cases already in SC-2016 cannot be identified
# here; the 2026-10-11 selection removed 2 of them.

# %% load
def find_record_files():
    root = Path(os.environ["JOLLY_RECORDS"]) if not ON_KAGGLE else Path("/kaggle/input")
    files = sorted(root.rglob("records*.jsonl.gz"))
    # only partitions that may be read; 2017 and 2018 files are never opened
    return [f for f in files if not re.search(r"(2016|2017|2018|2019|202\d)", f.name)]

record_files = find_record_files()
assert record_files, "no records*.jsonl.gz found: attach the jolly-aws-extract kernel (Kaggle) or set JOLLY_RECORDS (local)"
rows, seen = [], set()
for path in record_files:
    for record in read_aws_records(path):
        decided = record["decision_date"][:4]
        if not decided.isdigit() or int(decided) not in ALL_YEARS or record["path"].lower() in seen:
            continue
        seen.add(record["path"].lower())
        text = record["text"]
        label = official_label(record["disposal_nature"])
        if label is None or not text.strip():
            continue
        if infer_disposition(text[int(len(text) * 0.8):]) != label:
            continue
        view = build_input_view_v2(text, BASE_CHARS)
        if view is None:
            continue
        rows.append({"path": record["path"], "year": int(decided), "label": label, "text": text, "view": view})
cases = pd.DataFrame(rows)
cases["label3"] = cases.label.replace({"partly_allowed": "allowed"})
print(len(cases), "cases from", [p.name for p in find_record_files()])
display(pd.crosstab(cases.year, cases.label, margins=True))

# %% [markdown]
# ## 2. Baseline and helpers
# TF-IDF over 1-2 grams with balanced logistic regression, as in the earlier probe.

# %% helpers
def make_model(C=1.0, **tfidf):
    options = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000, dtype=np.float32)
    options.update(tfidf)
    return make_pipeline(TfidfVectorizer(**options), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))

def macro_f1(y, pred, labels):
    return f1_score(y, pred, labels=labels, average="macro", zero_division=0)

def score(y, pred, labels):
    return {"acc": round(accuracy_score(y, pred), 3), "macro_f1": round(macro_f1(y, pred, labels), 3)}

def paired_bootstrap(y, pred_a, pred_b, labels, n=BOOTSTRAPS, seed=SEED):
    """macro-F1(a) - macro-F1(b) over resampled cases: point estimate and 95% interval."""
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        pick = generator.integers(0, len(y), len(y))
        deltas.append(macro_f1(y[pick], pred_a[pick], labels) - macro_f1(y[pick], pred_b[pick], labels))
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = macro_f1(y, pred_a, labels) - macro_f1(y, pred_b, labels)
    return {"delta": round(point, 3), "ci_low": round(low, 3), "ci_high": round(high, 3),
            "adopt": bool(point >= MIN_GAIN and low > 0)}

train = cases[cases.year.isin(TRAIN_YEARS)].reset_index(drop=True)
dev = cases[cases.year == DEV_YEAR].reset_index(drop=True)
print(f"train {len(train)} (2013-2014)   dev {len(dev)} (2015)")

majority = pd.Series(["allowed"] * len(dev))
baseline_model = make_model().fit(train.view, train.label)
baseline_pred = baseline_model.predict(dev.view)
baseline = {"always_allowed": score(dev.label, majority, list(LABELS)),
            "tfidf_lr": score(dev.label, baseline_pred, list(LABELS))}
results["baseline"] = {"train_n": len(train), "dev_n": len(dev), **baseline}
print(json.dumps(baseline, indent=1))
print("earlier probe, same split and view: acc 0.483 / macro-F1 0.379 on 1,430 / 683")

# %% [markdown]
# ## 3. Probe: label schema
# Same features and training years, four ways of defining the task. Never replaces the 4-way numbers.
# The hierarchy asks `disposed` versus contested first, then allowed versus dismissed among contested cases.

# %% schema
labels4, labels3 = list(LABELS), ["allowed", "dismissed", "disposed"]
schema = {}
schema["4-way"] = score(dev.label, baseline_pred, labels4)

model3 = make_model().fit(train.view, train.label3)
pred3 = model3.predict(dev.view)
schema["3-way flat (partly merged into allowed)"] = score(dev.label3, pred3, labels3)

stage_a = make_model().fit(train.view, train.label3.eq("disposed"))
contested = train[train.label3 != "disposed"]
stage_b = make_model().fit(contested.view, contested.label3)
pred_h = np.where(stage_a.predict(dev.view), "disposed", stage_b.predict(dev.view))
schema["3-way hierarchy (disposed? then allowed/dismissed)"] = score(dev.label3, pred_h, labels3)

binary_train, binary_dev = train[train.label3 != "disposed"], dev[dev.label3 != "disposed"]
binary_pred = make_model().fit(binary_train.view, binary_train.label3).predict(binary_dev.view)
schema[f"binary allowed vs dismissed (n={len(binary_dev)})"] = score(binary_dev.label3, binary_pred, ["allowed", "dismissed"])
schema["binary always-allowed floor"] = score(binary_dev.label3, ["allowed"] * len(binary_dev), ["allowed", "dismissed"])
schema_table = pd.DataFrame(schema).T
results["schema"] = schema
display(schema_table)

# %% [markdown]
# ## 4. Probe: data scale
# (a) Train on progressively earlier windows ending in 2014 and score 2015. (b) A size-matched control: random
# subsets of the whole 2005-2014 pool, so size is separated from recency.

# %% scale
windows = {"2014": [2014], "2013-14": [2013, 2014], "2011-14": list(range(2011, 2015)),
           "2009-14": list(range(2009, 2015)), "2007-14": list(range(2007, 2015)), "2005-14": POOL_YEARS}
scale_rows = []
for name, years in windows.items():
    subset = cases[cases.year.isin(years)]
    model = make_model().fit(subset.view, subset.label)
    pred = model.predict(dev.view)
    scale_rows.append({"window": name, "n_train": len(subset), "kind": "recent-years",
                       "macro_f1_4way": macro_f1(dev.label, pred, labels4),
                       "macro_f1_3way": macro_f1(dev.label3, pd.Series(pred).replace({"partly_allowed": "allowed"}), labels3),
                       "acc_4way": accuracy_score(dev.label, pred)})
    print(name, len(subset), round(scale_rows[-1]["macro_f1_4way"], 3), flush=True)

pool = cases[cases.year.isin(POOL_YEARS)].reset_index(drop=True)
sizes = [n for n in (357, 715, 1430, 2860, 4300) if n < len(pool)] + [len(pool)]
generator = np.random.default_rng(SEED)
for n in sizes:
    for seed in range(5):
        subset = pool.iloc[generator.permutation(len(pool))[:n]]
        pred = make_model().fit(subset.view, subset.label).predict(dev.view)
        scale_rows.append({"window": f"random {n}", "n_train": n, "kind": "random-subset", "seed": seed,
                           "macro_f1_4way": macro_f1(dev.label, pred, labels4),
                           "macro_f1_3way": macro_f1(dev.label3, pd.Series(pred).replace({"partly_allowed": "allowed"}), labels3),
                           "acc_4way": accuracy_score(dev.label, pred)})
    print("random", n, round(np.mean([r["macro_f1_4way"] for r in scale_rows if r["window"] == f"random {n}"]), 3), flush=True)
scale = pd.DataFrame(scale_rows)
display(scale[scale.kind == "recent-years"].round(3))
by_size = scale[scale.kind == "random-subset"].groupby("n_train")[["macro_f1_4way", "macro_f1_3way", "acc_4way"]].agg(["mean", "std"]).round(3)
display(by_size)
results["scale"] = {"recent_years": scale[scale.kind == "recent-years"].round(4).to_dict("records"),
                    "random_subsets": json.loads(by_size.to_json())}

fig, axis = plt.subplots(figsize=(6.5, 4))
recent = scale[scale.kind == "recent-years"]
axis.plot(recent.n_train, recent.macro_f1_4way, "o-", color="#2a6f97", label="recent years (4-way)")
spread = scale[scale.kind == "random-subset"].groupby("n_train").macro_f1_4way.agg(["mean", "std"])
axis.errorbar(spread.index, spread["mean"], yerr=spread["std"], fmt="s--", color="#c26b00", capsize=3, label="random subset (4-way, 5 seeds)")
axis.set_xscale("log"); axis.set_xlabel("training cases"); axis.set_ylabel("dev macro-F1 (2015)")
axis.set_title("Data-scale probe"); axis.legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT / "scale.png", dpi=140); plt.show()

# %% [markdown]
# ## 5. Probe: input view, with a leakage audit
# Views are all built after the project's disposition masking and stop at 70% of the judgment. A view that scores
# higher is suspect until the audit shows the gain does not come from outcome words left in the text.

# %% views
OUTCOME_WORDS = r"(?:allow\w*|dismiss\w*|dispos\w*|set\s+aside|quash\w*|uphold\w*|upheld|affirm\w*|remand\w*)"
view_builders = {
    "opening 6,000 (baseline)": lambda t: build_input_view_v2(t, BASE_CHARS),
    "opening 24,000": lambda t: build_input_view_v2(t, 24_000),
    "whole body to 70% cap": lambda t: build_input_view_v2(t, 10**9),
    "reasoning: last 6,000 before cap": lambda t: (lambda v: None if v is None else v[-BASE_CHARS:])(build_input_view_v2(t, 10**9)),
}
view_rows, view_pred = [], {}
for name, builder in view_builders.items():
    train_v = train.text.map(builder); dev_v = dev.text.map(builder)
    keep_t, keep_d = train_v.notna(), dev_v.notna()
    model = make_model().fit(train_v[keep_t], train.label[keep_t])
    pred = pd.Series(model.predict(dev_v[keep_d]), index=dev.index[keep_d])
    view_pred[name] = pred.reindex(dev.index).fillna("allowed")  # unscorable cases count as the majority label
    residual = dev_v[keep_d].str.contains(OUTCOME_WORDS, case=False, regex=True)
    still_phrase = dev_v[keep_d].map(lambda v: infer_disposition(v) is not None)
    versus = {}
    if name != "opening 6,000 (baseline)":
        versus = {f"vs_baseline_{k}": v for k, v in paired_bootstrap(dev.label, view_pred[name], baseline_pred, labels4).items()}
    view_rows.append({"view": name, "dev_n_with_view": int(keep_d.sum()),
                      **score(dev.label, view_pred[name], labels4),
                      "views_with_outcome_words_%": round(100 * residual.mean(), 1),
                      "views_with_literal_disposition_%": round(100 * still_phrase.mean(), 1), **versus})
    print(name, view_rows[-1]["macro_f1"], flush=True)
views = pd.DataFrame(view_rows)
display(views)
results["views"] = views.to_dict("records")

# %% audit
# Where does the baseline's signal come from? (a) only the residual outcome words, (b) everything except them.
words_only = make_model(token_pattern=rf"(?u)\b{OUTCOME_WORDS}\b", ngram_range=(1, 1), min_df=1).fit(train.view, train.label)
strip = lambda text: re.sub(OUTCOME_WORDS, " ", text, flags=re.I)
without_words = make_model().fit(train.view.map(strip), train.label)
audit = {"baseline": score(dev.label, baseline_pred, labels4),
         "outcome words only": score(dev.label, words_only.predict(dev.view), labels4),
         "baseline minus outcome words": score(dev.label, without_words.predict(dev.view.map(strip)), labels4)}
display(pd.DataFrame(audit).T)
results["audit"] = audit
vectorizer, classifier = baseline_model.steps[0][1], baseline_model.steps[1][1]
names = np.array(vectorizer.get_feature_names_out())
top = {label: names[np.argsort(classifier.coef_[i])[-12:][::-1]].tolist() for i, label in enumerate(classifier.classes_)}
for label, features in top.items():
    print(f"{label:15s}", ", ".join(features))
results["top_features"] = top

# %% [markdown]
# ## 6. Probe: calibration
# Per-class score offsets are tuned on **out-of-fold predictions inside the training years** (5-fold), then applied
# once to the dev set. Nothing is tuned on dev itself.

# %% calibration
def oof_log_probs(frame, folds=5):
    out = np.zeros((len(frame), len(labels4)))
    for fit_idx, hold_idx in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(frame.view, frame.label):
        model = make_model().fit(frame.view.iloc[fit_idx], frame.label.iloc[fit_idx])
        out[hold_idx] = model.predict_log_proba(frame.view.iloc[hold_idx])
    return out

def tune_offsets(log_probs, y, rounds=3, grid=np.linspace(-1.5, 1.5, 31)):
    offsets = np.zeros(len(labels4))
    for _ in range(rounds):
        for k in range(len(labels4)):
            best = (-1, 0.0)
            for value in grid:
                trial = offsets.copy(); trial[k] = value
                f = macro_f1(y, np.array(labels4)[(log_probs + trial).argmax(1)], labels4)
                best = max(best, (f, value))
            offsets[k] = best[1]
    return offsets

calibration = {}
for name, frame in {"train 2013-14": train, "train 2005-14": pool}.items():
    fitted = make_model().fit(frame.view, frame.label)
    dev_log = fitted.predict_log_proba(dev.view)
    offsets = tune_offsets(oof_log_probs(frame), frame.label.values)
    plain = np.array(fitted.classes_)[dev_log.argmax(1)]
    tuned = np.array(fitted.classes_)[(dev_log + offsets).argmax(1)]
    calibration[name] = {"offsets": dict(zip(fitted.classes_, offsets.round(2))), "plain": score(dev.label, plain, labels4),
                         "tuned": score(dev.label, tuned, labels4), **paired_bootstrap(dev.label, tuned, plain, labels4)}
    print(name, json.dumps(calibration[name]), flush=True)
results["calibration"] = calibration
display(pd.DataFrame({k: {"plain mF1": v["plain"]["macro_f1"], "tuned mF1": v["tuned"]["macro_f1"], "delta": v["delta"],
                          "ci": f"[{v['ci_low']}, {v['ci_high']}]", "adopt": v["adopt"]} for k, v in calibration.items()}).T)

# %% [markdown]
# ## 7. Summary against the decision rule

# %% summary
summary_rows = []
def add(probe, setting, delta_record):
    summary_rows.append({"probe": probe, "setting": setting, **delta_record})
for _, row in views.iterrows():
    if "vs_baseline_delta" in row and not pd.isna(row["vs_baseline_delta"]):
        add("input view", row["view"], {"delta": row["vs_baseline_delta"], "ci_low": row["vs_baseline_ci_low"],
                                         "ci_high": row["vs_baseline_ci_high"], "adopt": row["vs_baseline_adopt"]})
for name, record in calibration.items():
    add("calibration", name, {k: record[k] for k in ("delta", "ci_low", "ci_high", "adopt")})
# data scale: full pool against the 2013-14 baseline, on the 4-way dev set
full_pred = make_model().fit(pool.view, pool.label).predict(dev.view)
add("data scale", f"train 2005-14 (n={len(pool)}) vs 2013-14 (n={len(train)})", paired_bootstrap(dev.label, full_pred, baseline_pred, labels4))
summary = pd.DataFrame(summary_rows)
display(summary)
results["summary_vs_decision_rule"] = summary.to_dict("records")
results["protocol"] = {"dev": "train 2013-14, dev 2015", "min_gain": MIN_GAIN, "bootstraps": BOOTSTRAPS,
                       "test_years_read": False}
(OUT / "phase1-results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
print("saved", OUT / "phase1-results.json")
