# %% [markdown]
# # Jolly Phase 5: freeze the final system and score the 2018 test set once (CPU)
#
# Fixed in advance (`docs/protocol-v2-score-improvement.md`, amendment of 2026-10-29): TF-IDF + balanced logistic
# regression on the `body-prefix-masked-v2` view, trained on every usable case decided in 2005-2015 and 2017. One model per
# schema. The models are fitted and hashed **before** any 2018 label is read. After this run the 2018 set is spent; do
# not re-run this notebook with changed settings.

# %% settings
import hashlib, json, os, pickle, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
from legal_graph.aws_judgments import official_label, read_aws_records
from legal_graph.outcomes import LABELS, build_input_view_v2, infer_disposition

TRAIN_YEARS = list(range(2005, 2016)) + [2017]
TEST_YEAR = 2018
BASE_CHARS, BOOTSTRAPS, SEED = 6000, 10_000, 20261029
SMOKE = os.environ.get("JOLLY_SMOKE") == "1"  # code check only: 2014 stands in for the test year, 2018 is never read
if SMOKE:
    TRAIN_YEARS, TEST_YEAR, BOOTSTRAPS = [2012, 2013], 2014, 200
print("SMOKE" if SMOKE else "FINAL", "train", TRAIN_YEARS, "test", TEST_YEAR)
OUT = Path("/kaggle/working/phase5") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
labels4, labels3, labels2 = list(LABELS), ["allowed", "dismissed", "disposed"], ["allowed", "dismissed"]

def record_files(years):
    root = Path(os.environ["JOLLY_RECORDS"]) if not ON_KAGGLE else Path("/kaggle/input")
    files = sorted(root.rglob("records-*.jsonl.gz"))
    return [f for f in files if int(re.search(r"records-(\d{4})", f.name).group(1)) in years]

def load_cases(years):
    files = record_files(years)
    assert files, "no records files found: attach the jolly-aws-extract kernel (Kaggle) or set JOLLY_RECORDS (local)"
    rows, seen = [], set()
    for path in files:
        for record in read_aws_records(path):
            decided = record["decision_date"][:4]
            if not decided.isdigit() or int(decided) not in years or record["path"].lower() in seen:
                continue
            seen.add(record["path"].lower())
            text, label = record["text"], official_label(record["disposal_nature"])
            if label is None or not text.strip() or infer_disposition(text[int(len(text) * 0.8):]) != label:
                continue
            view = build_input_view_v2(text, BASE_CHARS)
            if view is not None:
                rows.append({"path": record["path"], "year": int(decided), "label": label, "view": view})
    frame = pd.DataFrame(rows)
    frame["label3"] = frame.label.replace({"partly_allowed": "allowed"})
    return frame

def make_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000, dtype=np.float32),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))

# %% [markdown]
# ## 1. Fit and freeze (no 2018 data loaded yet)

# %% freeze
train = load_cases(TRAIN_YEARS)
print(len(train), "training cases")
display(pd.crosstab(train.year, train.label, margins=True))
started = time.time()
models = {"4-way": make_model().fit(train.view, train.label),
          "3-way": make_model().fit(train.view, train.label3),
          "hierarchy-stage-a": make_model().fit(train.view, train.label3.eq("disposed"))}
contested = train[train.label3 != "disposed"]
models["hierarchy-stage-b"] = make_model().fit(contested.view, contested.label3)
models["binary"] = models["hierarchy-stage-b"]  # allowed (incl. partly) vs dismissed, disposed dropped: the same model
freeze = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_years": TRAIN_YEARS, "train_n": len(train),
          "train_label_counts": train.label.value_counts().to_dict(), "test_labels_accessed_before_freeze": False, "models": {}}
for name, model in models.items():
    blob = pickle.dumps(model)
    (OUT / f"model-{name}.pkl").write_bytes(blob)
    freeze["models"][name] = hashlib.sha256(blob).hexdigest()
(OUT / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
print(json.dumps(freeze["models"], indent=1), f"\nfitted in {time.time() - started:.0f}s")

# %% [markdown]
# ## 2. Score 2018 once

# %% score
test = load_cases([TEST_YEAR])
print(len(test), "test cases (2018)")
display(test.label.value_counts().to_frame("2018 cases"))

def bootstrap_interval(y, pred, labels, metric):
    y, pred = np.asarray(y), np.asarray(pred)
    generator, values = np.random.default_rng(SEED), []
    for _ in range(BOOTSTRAPS):
        pick = generator.integers(0, len(y), len(y))
        values.append(metric(y[pick], pred[pick], labels))
    return [round(v, 3) for v in np.percentile(values, [2.5, 97.5])]

macro = lambda y, p, labels: f1_score(y, p, labels=labels, average="macro", zero_division=0)
accuracy = lambda y, p, labels: accuracy_score(y, p)

def paired_gain(y, pred_a, pred_b, labels):
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    generator, deltas = np.random.default_rng(SEED), []
    for _ in range(BOOTSTRAPS):
        pick = generator.integers(0, len(y), len(y))
        deltas.append(macro(y[pick], pred_a[pick], labels) - macro(y[pick], pred_b[pick], labels))
    return [round(v, 3) for v in np.percentile(deltas, [2.5, 97.5])]

binary_test = test[test.label3 != "disposed"]
stage_a = models["hierarchy-stage-a"].predict(test.view)
predictions = {
    "4-way": (test.label, models["4-way"].predict(test.view), labels4),
    "3-way flat": (test.label3, models["3-way"].predict(test.view), labels3),
    "3-way hierarchy": (test.label3, np.where(stage_a, "disposed", models["hierarchy-stage-b"].predict(test.view)), labels3),
    "binary allowed vs dismissed": (binary_test.label3, models["binary"].predict(binary_test.view), labels2),
}
report = {}
for name, (y, pred, labels) in predictions.items():
    floor = np.array(["allowed"] * len(y))
    report[name] = {"n": len(y), "acc": round(accuracy_score(y, pred), 3), "acc_ci": bootstrap_interval(y, pred, labels, accuracy),
                    "macro_f1": round(macro(y, pred, labels), 3), "macro_f1_ci": bootstrap_interval(y, pred, labels, macro),
                    "always_allowed_acc": round(accuracy_score(y, floor), 3), "always_allowed_macro_f1": round(macro(y, floor, labels), 3),
                    "gain_over_floor_ci": paired_gain(y, pred, floor, labels),
                    "per_class_f1": dict(zip(labels, np.round(f1_score(y, pred, labels=labels, average=None, zero_division=0), 3).tolist()))}
table = pd.DataFrame(report).T
display(table[["n", "acc", "acc_ci", "macro_f1", "macro_f1_ci", "always_allowed_acc", "always_allowed_macro_f1", "gain_over_floor_ci"]])
for name, row in report.items():
    print(f"{name:30s}", row["per_class_f1"])

# %% plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for axis, (name, (y, pred, labels)) in zip(axes, [(k, predictions[k]) for k in ("4-way", "binary allowed vs dismissed")]):
    matrix = confusion_matrix(y, pred, labels=labels)
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right"); axis.set_yticks(range(len(labels)), labels)
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, value, ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    axis.set_title(f"2018 test, {name}"); axis.set_xlabel("predicted"); axis.set_ylabel("gold")
fig.tight_layout(); fig.savefig(OUT / "phase5-confusion.png", dpi=140); plt.show()

# %% save
result = {"freeze": freeze, "test_year": TEST_YEAR, "test_n": len(test), "test_label_counts": test.label.value_counts().to_dict(),
          "bootstraps": BOOTSTRAPS, "report": report}
(OUT / "phase5-results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
print("saved", OUT / "phase5-results.json")
