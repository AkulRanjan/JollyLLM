# %% [markdown]
# # Jolly round 2: header metadata and recency weighting (CPU)
#
# Phase 1-3 found that TF-IDF over the judgment body tops out near 0.41 4-way macro-F1 and mostly learns the case type.
# This notebook tests whether information the body view leaves out helps (protocol v2, amendment "round 2"):
#
# * **(a) header metadata** known before the decision: case type and age, bench, judges, party types, the court
#   appealed from, counsel for each side, connected matters. The headnote (which states the result) is never read.
# * **(b) recency weighting** of training cases, half-life chosen on an inner split (train 2005-2013, score 2014).
# * **(c)** both.
#
# Same dev split as Phase 1 (dev 2015), compared with TF-IDF + LR trained on 2005-2014. Decision rule: macro-F1 gain of
# at least 0.03 with the paired bootstrap 95% interval excluding 0. No file after 2015 is opened.

# %% settings
import json, os, re, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
from legal_graph.aws_judgments import official_label, read_aws_records
from legal_graph.outcomes import LABELS, body_start, build_input_view_v2, infer_disposition

DEV_YEAR = 2015
POOL_YEARS = list(range(2005, 2015))
BASE_CHARS = 6000
MIN_GAIN, BOOTSTRAPS, SEED = 0.03, 2000, 20261029
HALF_LIVES = [1, 2, 4, 8, None]
OUT = Path("/kaggle/working/round2") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
labels4, labels3 = list(LABELS), ["allowed", "dismissed", "disposed"]
results = {}

# %% [markdown]
# ## 1. Load the cases and read the header
# Selection is the Phase 1 rule (official label = text rule on the last 20%, usable body view). The header fields come
# from the AWS metadata (title, case number, bench) and from regexes over the text before the judgment body.

# %% header
PARTY_TYPES = [
    ("state", r"\b(?:state of|union of india|government|govt\.?|u\.?o\.?i\.?|collector|district magistrate|"
              r"superintendent of police|c\.?b\.?i\.?|central bureau|enforcement directorate|n\.?c\.?b\.?|lt\.? governor|police)\b"),
    ("tax", r"\b(?:commissioner|income tax|customs|central excise|sales tax|revenue|assessing officer|c\.?i\.?t\.?)\b"),
    ("insurer", r"\b(?:insurance|assurance|l\.?i\.?c\.?)\b"),
    ("bank", r"\bbank\b"),
    ("body", r"\b(?:municipal|corporation|authority|board|university|council|commission|society|trust|"
             r"development|nigam|samiti|panchayat|cooperative|co-operative)\b"),
    ("company", r"\b(?:ltd|limited|pvt|private|m/s|company|industries|enterprises|& co)\b"),
]
JUDGES = re.compile(r"\[([^\[\]]{5,250}?),?\s*(?:C\.?\s*)?J\s*J?\s*\.?\s*[\]\)]")
FROM_LINE = re.compile(r"From the (?:final |impugned )?(?:Judgment|Order)[^\n]{0,300}?(?:of|by) the ([^\n]{3,120}?)(?: in |\.\s*$|\n)", re.I | re.M)
COUNSEL = re.compile(r"^([^\n]{3,600}?)\s+for the (Appellants?|Respondents?|Petitioners?|Complainants?)\b", re.I | re.M)
CASE_NO = re.compile(r"(\d{4})\s*$")

def party_type(side):
    side = side.lower()
    for name, pattern in PARTY_TYPES:
        if re.search(pattern, side):
            return name
    return "person"

def slug(value):
    return re.sub(r"[^a-z]", "", value.lower())

def judge_tokens(text):
    match = JUDGES.search(text[:2500])
    if not match:
        return ["judge_none"]
    names = re.split(r"\s*(?:,|\band\b)\s*", match.group(1), flags=re.I)
    return [f"judge_{slug(name.split()[-1])}" for name in names if name.strip() and slug(name.split()[-1])]

def counsel_tokens(header):
    tokens = []
    for names, side in COUNSEL.findall(header):
        side = "app" if side.lower().startswith(("appellant", "petitioner")) else "resp"
        for name in re.split(r",|\band\b", names, flags=re.I):
            words = [w for w in re.findall(r"[A-Za-z]+", name) if len(w) > 1]
            # a name, not a sentence such as "It was contended for the appellant"
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words):
                tokens.append(f"c{side}_{slug(''.join(words))}")
    return tokens or ["counsel_none"]

def header_meta(record, decision_year):
    text = record["text"]
    start = body_start(text) or 0
    header = text[: start + 2500]  # the jurisdiction block, from-line and counsel sit around the body start
    case_no = record["case_number"]
    ctype = slug(re.sub(r"\bno\b.*$", "", case_no, flags=re.I)) or "unknown"
    filed = CASE_NO.search(case_no)
    age = decision_year - int(filed.group(1)) if filed and 1950 < int(filed.group(1)) <= decision_year else None
    age_bucket = "na" if age is None else str(min(age, 8)) if age < 5 else "5_7" if age < 8 else "8plus"
    bench = re.sub(r"\D", "", record["bench"]) or "na"
    left, _, right = record["title"].partition(" v. ")
    tokens = [f"ctype_{ctype}", f"age_{age_bucket}", f"bench_{bench}",
              f"app_{party_type(left)}", f"resp_{party_type(right)}"]
    for side, name in ((left, "app_multi"), (right, "resp_multi")):
        if re.search(r"\b(?:ors|others|anr)\b", side, re.I):
            tokens.append(name)
    tokens += judge_tokens(text)
    court = FROM_LINE.search(header)
    tokens += [f"from_{w}" for w in re.findall(r"[a-z]{3,}", court.group(1).lower())] if court else ["from_none"]
    tokens += counsel_tokens(header)
    if re.search(r"^\s*WITH\s*$|^\s*WITH\b[^\n]{0,40}\bNo", header, re.M):
        tokens.append("connected")
    return " ".join(tokens)

# %% load
def find_record_files():
    root = Path(os.environ["JOLLY_RECORDS"]) if not ON_KAGGLE else Path("/kaggle/input")
    return [f for f in sorted(root.rglob("records*.jsonl.gz")) if not re.search(r"(2016|2017|2018|2019|202\d)", f.name)]

record_files = find_record_files()
assert record_files, "attach the jolly-aws-extract kernel (Kaggle) or set JOLLY_RECORDS (local)"
rows, seen = [], set()
for path in record_files:
    for record in read_aws_records(path):
        decided = record["decision_date"][:4]
        if not decided.isdigit() or int(decided) not in POOL_YEARS + [DEV_YEAR] or record["path"].lower() in seen:
            continue
        seen.add(record["path"].lower())
        text, label = record["text"], official_label(record["disposal_nature"])
        if label is None or not text.strip() or infer_disposition(text[int(len(text) * 0.8):]) != label:
            continue
        view = build_input_view_v2(text, BASE_CHARS)
        if view is None:
            continue
        rows.append({"path": record["path"], "year": int(decided), "label": label, "view": view,
                     "meta": header_meta(record, int(decided)), "disposal_nature": record["disposal_nature"],
                     "ending": text[-700:]})
cases = pd.DataFrame(rows)
cases["label3"] = cases.label.replace({"partly_allowed": "allowed"})
pool = cases[cases.year.isin(POOL_YEARS)].reset_index(drop=True)
dev = cases[cases.year == DEV_YEAR].reset_index(drop=True)
print(f"pool {len(pool)} (2005-2014)   dev {len(dev)} (2015)")

# %% coverage
groups = ["ctype", "age", "bench", "app", "resp", "judge", "from", "capp", "cresp", "connected"]
def group_of(token):
    return token.split("_")[0] if "_" in token else token
coverage = {g: round(100 * cases.meta.map(lambda m: any(group_of(t) == g and not t.endswith("_none") and not t.endswith("_na")
                                                          for t in m.split())).mean(), 1) for g in groups}
print("share of cases with each header field found (%):", coverage)
results["coverage_pct"] = coverage
for g in ["ctype", "app", "resp", "judge", "from", "capp"]:
    counts = Counter(t for m in pool.meta for t in m.split() if group_of(t) == g)
    print(g, counts.most_common(8))
print(pool.meta.iloc[0])

# %% [markdown]
# ## 2. Models and helpers
# Text: TF-IDF 1-2 grams, min_df 2, sublinear tf, balanced logistic regression (C = 1), as in Phases 1-5. Metadata:
# binary bag of header tokens (min_df 2), stacked next to the text features in the same logistic regression.

# %% helpers
class Model:
    def __init__(self, text=True, meta=False, C=1.0, meta_weight=None):
        # meta_weight None: raw binary block (round 2); a number: block L2-normalised per case, times the weight (round 2b)
        self.text, self.meta, self.C, self.meta_weight = text, meta, C, meta_weight

    def _features(self, frame, fit):
        parts = []
        if self.text:
            if fit:
                self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000, dtype=np.float32)
            parts.append(self.tfidf.fit_transform(frame.view) if fit else self.tfidf.transform(frame.view))
        if self.meta:
            if fit:
                self.bag = CountVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, binary=True, dtype=np.float32)
            block = self.bag.fit_transform(frame.meta) if fit else self.bag.transform(frame.meta)
            if self.meta_weight is not None:
                block = normalize(block) * self.meta_weight
            parts.append(block)
        return hstack(parts).tocsr()

    def fit(self, frame, y, weights=None):
        self.lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000)
        self.lr.fit(self._features(frame, True), y, sample_weight=weights)
        return self

    def predict(self, frame):
        return self.lr.predict(self._features(frame, False))

def recency(frame, half_life, last_year):
    return None if half_life is None else 0.5 ** ((last_year - frame.year.values) / half_life)

def macro_f1(y, pred, labels):
    return f1_score(y, pred, labels=labels, average="macro", zero_division=0)

def score(y, pred, labels):
    return {"acc": round(accuracy_score(y, pred), 3), "macro_f1": round(macro_f1(y, pred, labels), 3)}

def paired_bootstrap(y, pred_a, pred_b, labels, n=BOOTSTRAPS, seed=SEED):
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

# %% [markdown]
# ## 3. Choose the recency half-life on the inner split (train 2005-2013, score 2014)
# The dev year is not used for this choice.

# %% halflife
inner_train, inner_dev = pool[pool.year <= 2013], pool[pool.year == 2014]
inner = {}
for meta in (False, True):
    for h in HALF_LIVES:
        pred = Model(meta=meta).fit(inner_train, inner_train.label, recency(inner_train, h, 2013)).predict(inner_dev)
        inner[(meta, h)] = macro_f1(inner_dev.label, pred, labels4)
        print(f"meta={meta!s:5}  half-life {h}: inner macro-F1 {inner[(meta, h)]:.3f}", flush=True)
best_h = {meta: max(HALF_LIVES, key=lambda h: (inner[(meta, h)], h is None)) for meta in (False, True)}
print("chosen half-life: text", best_h[False], " text+meta", best_h[True])
results["inner_halflife"] = {f"meta={m} h={h}": round(v, 4) for (m, h), v in inner.items()}
results["chosen_halflife"] = {"text": best_h[False], "text+meta": best_h[True]}

# %% [markdown]
# ## 4. Dev results against the decision rule (train 2005-2014, dev 2015)

# %% dev
candidates = {
    "baseline: text": dict(meta=False, text=True, h=None),
    "(a) metadata only": dict(meta=True, text=False, h=None),
    "(a) text + metadata": dict(meta=True, text=True, h=None),
    "(b) text, recency": dict(meta=False, text=True, h=best_h[False]),
    "(c) text + metadata, recency": dict(meta=True, text=True, h=best_h[True]),
}
preds = {}
for name, spec in candidates.items():
    preds[name] = Model(text=spec["text"], meta=spec["meta"]).fit(pool, pool.label, recency(pool, spec["h"], 2014)).predict(dev)
    print(name, score(dev.label, preds[name], labels4), flush=True)
rows = []
for name in candidates:
    row = {"candidate": name, **score(dev.label, preds[name], labels4)}
    if name != "baseline: text":
        row.update(paired_bootstrap(dev.label, preds[name], preds["baseline: text"], labels4))
    row["per_class_f1"] = dict(zip(labels4, f1_score(dev.label, preds[name], labels=labels4, average=None, zero_division=0).round(3)))
    rows.append(row)
dev_table = pd.DataFrame(rows)
display(dev_table)
results["dev"] = dev_table.to_dict("records")

# %% [markdown]
# ## 5. Which header fields carry the signal? (report only; nothing is chosen from this)
# Text + metadata with one field group removed at a time.

# %% ablation
ablation = {}
full = macro_f1(dev.label, preds["(a) text + metadata"], labels4)
for g in groups:
    keep = lambda m: " ".join(t for t in m.split() if group_of(t) != g)
    p, d = pool.assign(meta=pool.meta.map(keep)), dev.assign(meta=dev.meta.map(keep))
    ablation[g] = round(macro_f1(dev.label, Model(meta=True).fit(p, p.label).predict(d), labels4) - full, 3)
print("macro-F1 change when the group is dropped:", ablation)
results["ablation_drop_group"] = ablation
meta_only = Model(text=False, meta=True).fit(pool, pool.label)
names = np.array(meta_only.bag.get_feature_names_out())
top = {label: names[np.argsort(meta_only.lr.coef_[i])[-12:][::-1]].tolist() for i, label in enumerate(meta_only.lr.classes_)}
for label, features in top.items():
    print(f"{label:15s}", ", ".join(features))
results["top_meta_features"] = top

# %% [markdown]
# ## 6. Other schemas for the best candidate (report only)

# %% schemas
best = max([n for n in candidates if n != "baseline: text"], key=lambda n: macro_f1(dev.label, preds[n], labels4))
spec = candidates[best]
schemas = {}
for name, spec_ in {"baseline: text": candidates["baseline: text"], best: spec}.items():
    m3 = Model(text=spec_["text"], meta=spec_["meta"]).fit(pool, pool.label3, recency(pool, spec_["h"], 2014))
    contested_pool, contested_dev = pool[pool.label3 != "disposed"], dev[dev.label3 != "disposed"]
    mb = Model(text=spec_["text"], meta=spec_["meta"]).fit(contested_pool, contested_pool.label3, recency(contested_pool, spec_["h"], 2014))
    schemas[name] = {"3-way": score(dev.label3, m3.predict(dev), labels3),
                     f"binary (n={len(contested_dev)})": score(contested_dev.label3, mb.predict(contested_dev), ["allowed", "dismissed"])}
display(pd.DataFrame({k: {s: v["macro_f1"] for s, v in d.items()} for k, d in schemas.items()}))
results["schemas"] = schemas

# %% [markdown]
# ## 7. Label audit: `disposed` and `partly allowed` (training years only)
# What official disposal values sit behind each label, and a fixed random sample of judgment endings to read by hand.

# %% audit
display(pd.crosstab(pool.disposal_nature, pool.label))
sample = pd.concat([pool[pool.label == "disposed"].sample(40, random_state=SEED),
                    pool[pool.label == "partly_allowed"].sample(25, random_state=SEED)])
with open(OUT / "label-audit-sample.txt", "w", encoding="utf-8") as handle:
    for _, row in sample.iterrows():
        handle.write(f"=== {row.label} | {row.year} | {row.path} | {row.disposal_nature}\n{row.ending}\n\n")
print("wrote", OUT / "label-audit-sample.txt", len(sample), "endings")

# %% [markdown]
# ## 8. Round 2b: scaled metadata without counsel names
# Added after round 2 failed (protocol amendment "round 2b"). Counsel tokens are dropped, the metadata block is
# L2-normalised per case and weighted; the weight is chosen on the inner split, then dev is scored once.

# %% round2b
META_WEIGHTS = [0.1, 0.2, 0.35, 0.5, 0.75]
no_counsel = lambda m: " ".join(t for t in m.split() if group_of(t) not in ("capp", "cresp") and t != "counsel_none")
pool_b, dev_b = pool.assign(meta=pool.meta.map(no_counsel)), dev.assign(meta=dev.meta.map(no_counsel))
inner_train_b, inner_dev_b = pool_b[pool_b.year <= 2013], pool_b[pool_b.year == 2014]
inner_text = inner[(False, None)]
inner_b = {}
for w in META_WEIGHTS:
    pred = Model(meta=True, meta_weight=w).fit(inner_train_b, inner_train_b.label).predict(inner_dev_b)
    inner_b[w] = macro_f1(inner_dev_b.label, pred, labels4)
    print(f"weight {w}: inner macro-F1 {inner_b[w]:.3f}   (text alone {inner_text:.3f})", flush=True)
best_w = max(META_WEIGHTS, key=lambda w: inner_b[w])
pred_b = Model(meta=True, meta_weight=best_w).fit(pool_b, pool_b.label).predict(dev_b)
round2b = {"inner": {str(w): round(v, 4) for w, v in inner_b.items()}, "chosen_weight": best_w,
           "dev": score(dev.label, pred_b, labels4),
           "per_class_f1": dict(zip(labels4, f1_score(dev.label, pred_b, labels=labels4, average=None, zero_division=0).round(3))),
           **paired_bootstrap(dev.label, pred_b, preds["baseline: text"], labels4)}
print(json.dumps(round2b, indent=1, default=str))
results["round2b_scaled_metadata"] = round2b

# %% save
results["protocol"] = {"dev": "train 2005-14, dev 2015", "min_gain": MIN_GAIN, "bootstraps": BOOTSTRAPS, "test_years_read": False}
(OUT / "round2-results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
print("saved", OUT / "round2-results.json")
