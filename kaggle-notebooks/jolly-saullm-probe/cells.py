# %% [markdown]
# # Jolly: SaulLM-7B frozen embeddings against TF-IDF (GPU)
#
# SaulLM-7B is Mistral-7B with continued pre-training on legal text. Phase 3 found that neither InLegalBERT nor
# Qwen3-Embedding beats TF-IDF; this asks the same of a legal-domain 7B model (protocol v2, amendment "SaulLM probe").
#
# * Same cases, input view and dev year as Phases 1-3 (dev 2015). Years 2016 and later are never opened.
# * SaulLM is frozen and loaded in 4-bit. Features are mean-pooled hidden states of one layer; the layer and the head's C
#   are chosen by cross-validation inside the training years, never on dev.
# * Training years: 2013-2014, then earlier years, most recent first, while the embedding budget lasts. TF-IDF is trained
#   on exactly the same years.
# * Decision rule: dev macro-F1 gain of at least 0.03 with the paired bootstrap 95% interval excluding 0.

# %% settings
import gc, json, os, re, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

SMOKE = os.environ.get("JOLLY_SMOKE") == "1"  # small CPU model, few cases: checks the code path only
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
from legal_graph.aws_judgments import official_label, read_aws_records
from legal_graph.outcomes import LABELS, build_input_view_v2, infer_disposition

DEV_YEAR, CORE_YEARS = 2015, [2014, 2013]
EXTRA_YEARS = [2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005]  # added most recent first while the budget lasts
BASE_CHARS, MAX_TOKENS = 6000, 1536
MIN_GAIN, BOOTSTRAPS, SEED = 0.03, 2000, 20261029
MODELS = ["Qwen/Qwen2.5-0.5B"] if SMOKE else ["Equall/Saul-7B-Base", "Equall/Saul-7B-Instruct-v1"]
LAYERS = [2, 4, 6, 8] if SMOKE else [8, 16, 24, 32]
BUDGET_SECONDS = 120 if SMOKE else 3 * 3600
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/saullm") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
labels4 = list(LABELS)
results, started_all = {"smoke": SMOKE}, time.time()
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", "| smoke:", SMOKE)

# %% [markdown]
# ## 1. Load the cases (same rules as Phases 1-3)

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
        if not decided.isdigit() or int(decided) not in CORE_YEARS + EXTRA_YEARS + [DEV_YEAR] or record["path"].lower() in seen:
            continue
        seen.add(record["path"].lower())
        text, label = record["text"], official_label(record["disposal_nature"])
        if label is None or not text.strip() or infer_disposition(text[int(len(text) * 0.8):]) != label:
            continue
        view = build_input_view_v2(text, BASE_CHARS)
        if view is not None:
            rows.append({"path": record["path"], "year": int(decided), "label": label, "view": view})
cases = pd.DataFrame(rows)
if SMOKE:
    cases = pd.concat([g.sample(min(len(g), 40), random_state=SEED) for _, g in cases.groupby("year")])
cases = cases.reset_index(drop=True)
print(cases.groupby("year").size().to_dict())

# %% [markdown]
# ## 2. Load SaulLM in 4-bit and embed, dev first
# Each case is embedded once; the mean-pooled hidden state of every candidate layer is kept. Embeddings are saved per
# year, so the output holds them even if the session stops.

# %% model
def load(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    if device.type == "cuda":
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                                   bnb_4bit_use_double_quant=True)
        model = AutoModel.from_pretrained(name, quantization_config=quant, device_map={"": 0}, torch_dtype=torch.float16)
    else:
        model = AutoModel.from_pretrained(name, torch_dtype=torch.float32)
    return tokenizer, model.eval()

for name in MODELS:
    try:
        tokenizer, model = load(name)
        model_name = name
        break
    except Exception as error:  # gated or missing: try the next one
        print("could not load", name, "->", type(error).__name__, str(error)[:200])
else:
    raise RuntimeError("no SaulLM checkpoint could be loaded")
results["model"] = model_name
print("loaded", model_name, "layers:", model.config.num_hidden_layers)

@torch.no_grad()
def embed(texts, batch=4):
    encoded = [tokenizer(t, add_special_tokens=True, truncation=True, max_length=MAX_TOKENS)["input_ids"] for t in texts]
    order = np.argsort([len(e) for e in encoded])[::-1]  # longest first: an out-of-memory error shows up at once
    out = np.zeros((len(LAYERS), len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch):
        picked = order[start:start + batch]
        width = max(len(encoded[j]) for j in picked)
        ids = torch.tensor([encoded[j] + [tokenizer.pad_token_id] * (width - len(encoded[j])) for j in picked], device=device)
        mask = torch.tensor([[1] * len(encoded[j]) + [0] * (width - len(encoded[j])) for j in picked], device=device)
        hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        weights = mask.unsqueeze(-1).float()
        for k, layer in enumerate(LAYERS):
            pooled = (hidden[layer].float() * weights).sum(1) / weights.sum(1)
            out[k, picked] = pooled.cpu().numpy()
    return out

embedded, timings, budget_start = {}, {}, time.time()
for year in [DEV_YEAR] + CORE_YEARS + EXTRA_YEARS:
    if year not in CORE_YEARS + [DEV_YEAR] and time.time() - budget_start > BUDGET_SECONDS:
        print(f"budget used: stop before {year}")
        break
    part = cases[cases.year == year]
    t = time.time()
    embedded[year] = embed(part.view.tolist())
    timings[year] = round(time.time() - t)
    np.save(OUT / f"emb-{year}.npy", embedded[year].astype(np.float16))
    print(f"{year}: {len(part)} cases in {timings[year]}s ({timings[year] / max(len(part), 1):.2f} s/case)", flush=True)
results["embedding_seconds"] = timings
train_years = [y for y in embedded if y != DEV_YEAR]
results["train_years"] = train_years
del model; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None

# %% [markdown]
# ## 3. Heads and comparison with TF-IDF on the same training years

# %% compare
def macro_f1(y, pred):
    return f1_score(y, pred, labels=labels4, average="macro", zero_division=0)

def score(y, pred):
    return {"acc": round(accuracy_score(y, pred), 3), "macro_f1": round(macro_f1(y, pred), 3)}

def paired_bootstrap(y, pred_a, pred_b, n=BOOTSTRAPS, seed=SEED):
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    generator, deltas = np.random.default_rng(seed), []
    for _ in range(n):
        pick = generator.integers(0, len(y), len(y))
        deltas.append(macro_f1(y[pick], pred_a[pick]) - macro_f1(y[pick], pred_b[pick]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = macro_f1(y, pred_a) - macro_f1(y, pred_b)
    return {"delta": round(point, 3), "ci_low": round(low, 3), "ci_high": round(high, 3), "adopt": bool(point >= MIN_GAIN and low > 0)}

def to_labels(proba):
    return np.array(labels4)[np.asarray(proba).argmax(1)]

def head(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))

dev = cases[cases.year == DEV_YEAR].reset_index(drop=True)
rows_out = []
settings = {"2013-14": CORE_YEARS, f"{min(train_years)}-14": train_years} if len(train_years) > 2 else {"2013-14": CORE_YEARS}
for setting, years in settings.items():
    train = pd.concat([cases[cases.year == y] for y in years]).reset_index(drop=True)
    y_train = train.label.values
    X_dev_all = embedded[DEV_YEAR]
    X_train_all = np.concatenate([embedded[y] for y in years], axis=1)
    best = (-1.0, None, None)
    for k, layer in enumerate(LAYERS):
        for C in (0.01, 0.1, 1.0):
            oof = np.empty(len(y_train), dtype=object)
            for fit_idx, hold_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_train_all[k], y_train):
                oof[hold_idx] = head(C).fit(X_train_all[k][fit_idx], y_train[fit_idx]).predict(X_train_all[k][hold_idx])
            best = max(best, (macro_f1(y_train, oof), k, C), key=lambda b: b[0])
    oof_f1, k, C = best
    saul = head(C).fit(X_train_all[k], y_train)
    assert list(saul.classes_) == labels4
    saul_proba = saul.predict_proba(X_dev_all[k])
    tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000, dtype=np.float32),
                          LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)).fit(train.view, y_train)
    tfidf_proba = tfidf.predict_proba(dev.view)
    base = to_labels(tfidf_proba)
    for method, proba in (("TF-IDF", tfidf_proba), ("SaulLM + LR", saul_proba), ("SaulLM + LR, averaged with TF-IDF", (saul_proba + tfidf_proba) / 2)):
        row = {"setting": setting, "n_train": len(train), "method": method, **score(dev.label, to_labels(proba))}
        if method != "TF-IDF":
            row.update(paired_bootstrap(dev.label, to_labels(proba), base))
            row.update({"layer": LAYERS[k], "C": C, "train_oof_macro_f1": round(oof_f1, 3)})
        rows_out.append(row)
        print(row, flush=True)
table = pd.DataFrame(rows_out)
display(table)
results["dev"] = rows_out
results["total_seconds"] = round(time.time() - started_all)
results["protocol"] = {"dev": "dev 2015", "min_gain": MIN_GAIN, "bootstraps": BOOTSTRAPS, "test_years_read": False}
(OUT / "saullm-results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
print(f"total {results['total_seconds']}s")
