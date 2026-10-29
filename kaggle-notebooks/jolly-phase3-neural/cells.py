# %% [markdown]
# # Jolly Phase 3: cheap neural baselines against TF-IDF (GPU)
#
# Phase 1 showed a bag-of-words model tops out near 0.40-0.42 4-way macro-F1 on the 2015 development set. This
# notebook asks whether a neural model can beat it, before anyone spends 15-30 GPU hours on an LLM run
# (`docs/score-improvement-plan.md`, Phase 3).
#
# * Same cases, split and input view as Phase 1: train 2013-2014 (or 2005-2014), develop on 2015. Years 2016 and later
#   are never opened.
# * Candidates: frozen InLegalBERT embeddings, frozen Qwen3-Embedding-0.6B embeddings, InLegalBERT fine-tuned
#   (2 seeds), and each of them averaged with TF-IDF.
# * Gate (`docs/protocol-v2-score-improvement.md`): a candidate counts only if dev macro-F1 beats TF-IDF by at least
#   0.03 with a paired bootstrap 95% interval above 0. For fine-tuning, every seed has to pass.
# * Dev only. No test number appears here.

# %% settings
import gc, json, os, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")
from legal_graph.aws_judgments import official_label, read_aws_records
from legal_graph.outcomes import LABELS, build_input_view_v2, infer_disposition

SMOKE = os.environ.get("JOLLY_SMOKE") == "1"  # tiny local run that only checks the code path
TRAIN_YEARS, DEV_YEAR, POOL_YEARS = [2013, 2014], 2015, list(range(2005, 2015))
BASE_CHARS, MIN_GAIN, BOOTSTRAPS, SEED = 6000, 0.03, 2000, 20261029
INLEGAL, QWEN_EMB = "law-ai/InLegalBERT", "Qwen/Qwen3-Embedding-0.6B"
MAX_CHUNKS, CHUNK_TOKENS = (1 if SMOKE else 3), 510
FT_EPOCHS, FT_LR, FT_BATCH, FT_SEEDS = (1 if SMOKE else 3), 2e-5, 16, ([1] if SMOKE else [1, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/phase3") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
labels4 = list(LABELS)
label_index = {label: i for i, label in enumerate(labels4)}
results, started_all = {"smoke": SMOKE}, time.time()
print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", "| smoke:", SMOKE)

# %% [markdown]
# ## 1. Load the cases (same rules as Phase 1)

# %% load
def find_record_files():
    root = Path(os.environ["JOLLY_RECORDS"]) if not ON_KAGGLE else Path("/kaggle/input")
    return [f for f in sorted(root.rglob("records*.jsonl.gz")) if not re.search(r"(2016|2017|2018|2019|202\d)", f.name)]

record_files = find_record_files()
assert record_files, "no records*.jsonl.gz found: attach the jolly-aws-extract kernel (Kaggle) or set JOLLY_RECORDS (local)"
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
        if view is not None:
            rows.append({"path": record["path"], "year": int(decided), "label": label, "view": view})
cases = pd.DataFrame(rows)
if SMOKE:
    cases = pd.concat([g.sample(min(len(g), 60), random_state=SEED) for _, g in cases.groupby("year")])
train = cases[cases.year.isin(TRAIN_YEARS)].reset_index(drop=True)
pool = cases[cases.year.isin(POOL_YEARS)].reset_index(drop=True)
dev = cases[cases.year == DEV_YEAR].reset_index(drop=True)
print(f"pool 2005-14: {len(pool)}   train 2013-14: {len(train)}   dev 2015: {len(dev)}")

# %% [markdown]
# ## 2. TF-IDF reference and helpers

# %% helpers
def macro_f1(y, pred, labels=labels4):
    return f1_score(y, pred, labels=labels, average="macro", zero_division=0)

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

def tfidf_proba(train_frame):
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000, dtype=np.float32),
                          LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)).fit(train_frame.view, train_frame.label)
    assert list(model.classes_) == labels4
    return model.predict_proba(dev.view)

settings = {"2013-14": train, "2005-14": pool}
tfidf = {name: tfidf_proba(frame) for name, frame in settings.items()}
reference = {name: score(dev.label, to_labels(proba)) for name, proba in tfidf.items()}
results["tfidf_reference"] = reference
print("TF-IDF dev reference:", json.dumps(reference))
candidates = []  # one record per candidate

def record(method, setting, proba, seed=None, extra=None):
    base = tfidf[setting]
    ensemble = (np.asarray(proba) + base) / 2
    candidates.append({"method": method, "setting": setting, "seed": seed, **score(dev.label, to_labels(proba)),
                       **{f"vs_tfidf_{k}": v for k, v in paired_bootstrap(dev.label, to_labels(proba), to_labels(base)).items()},
                       **(extra or {})})
    candidates.append({"method": method + " + TF-IDF", "setting": setting, "seed": seed, **score(dev.label, to_labels(ensemble)),
                       **{f"vs_tfidf_{k}": v for k, v in paired_bootstrap(dev.label, to_labels(ensemble), to_labels(base)).items()}})
    print(candidates[-2]["method"], setting, seed, candidates[-2]["acc"], candidates[-2]["macro_f1"], "|", candidates[-1]["macro_f1"], flush=True)

def free_gpu():
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Frozen embeddings plus a logistic-regression head
# Each view (about 1,400 tokens) is split into up to three 510-token chunks for InLegalBERT and the chunk vectors are
# averaged. Qwen3-Embedding reads the whole view at once (last-token pooling). The head's C is chosen by 5-fold
# out-of-fold macro-F1 inside the training years, never on dev.

# %% embed
@torch.no_grad()
def embed_bert(texts, name, batch=32):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 10**6
    model = AutoModel.from_pretrained(name).to(device).eval()
    if device.type == "cuda":
        model = model.half()
    chunks, owner = [], []
    for i, text in enumerate(texts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][: MAX_CHUNKS * CHUNK_TOKENS]
        for start in range(0, max(len(ids), 1), CHUNK_TOKENS):
            chunks.append([tokenizer.cls_token_id] + ids[start:start + CHUNK_TOKENS] + [tokenizer.sep_token_id]); owner.append(i)
    order = np.argsort([len(c) for c in chunks])
    vectors = np.zeros((len(chunks), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch):
        picked = order[start:start + batch]
        width = max(len(chunks[j]) for j in picked)
        ids = torch.tensor([chunks[j] + [tokenizer.pad_token_id] * (width - len(chunks[j])) for j in picked], device=device)
        mask = (ids != tokenizer.pad_token_id).long()
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        vectors[picked] = pooled.cpu().numpy()
    owner = np.array(owner)
    out = np.stack([vectors[owner == i].mean(0) for i in range(len(texts))])
    del model; free_gpu()
    return out

@torch.no_grad()
def embed_last_token(texts, name, max_tokens=1536, batch=4):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device).eval()
    encoded = [tokenizer(t, add_special_tokens=False)["input_ids"][: max_tokens - 1] + [tokenizer.eos_token_id] for t in texts]
    order = np.argsort([len(e) for e in encoded])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(order), batch):
        picked = order[start:start + batch]
        width = max(len(encoded[j]) for j in picked)
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(encoded[j])) + encoded[j] for j in picked], device=device)
        mask = (ids != tokenizer.pad_token_id).long()
        last = model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1].float()
        out[picked] = torch.nn.functional.normalize(last, dim=-1).cpu().numpy()
    del model; free_gpu()
    return out

everything = pd.concat([pool, dev]).reset_index(drop=True)  # embed each case once
embeddings, timings = {}, {}
for key, function, name in [("InLegalBERT (frozen)", embed_bert, INLEGAL), ("Qwen3-Embedding-0.6B (frozen)", embed_last_token, QWEN_EMB)]:
    if SMOKE and "Qwen" in key:
        continue
    t = time.time()
    embeddings[key] = function(everything.view.tolist(), name)
    timings[key] = round(time.time() - t)
    print(f"{key}: {embeddings[key].shape} in {timings[key]}s", flush=True)
results["embedding_seconds"] = timings
pool_rows = np.arange(len(pool)); dev_rows = np.arange(len(pool), len(everything))
train_rows = pool_rows[pool.year.isin(TRAIN_YEARS).values]

# %% head
def fit_head(X_train, y_train, X_dev):
    best = (-1, 1.0)
    for C in (0.01, 0.1, 1.0):
        oof = np.empty(len(y_train), dtype=object)
        for fit_idx, hold_idx in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_train, y_train):
            head = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)).fit(X_train[fit_idx], y_train[fit_idx])
            oof[hold_idx] = head.predict(X_train[hold_idx])
        best = max(best, (macro_f1(y_train, oof), C))
    head = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], class_weight="balanced", max_iter=2000)).fit(X_train, y_train)
    assert list(head.classes_) == labels4
    return head.predict_proba(X_dev), best

for key, matrix in embeddings.items():
    for setting, rows_ in (("2013-14", train_rows), ("2005-14", pool_rows)):
        proba, (oof_f1, chosen_c) = fit_head(matrix[rows_], everything.label.values[rows_], matrix[dev_rows])
        record(key + " + LR", setting, proba, extra={"head_C": chosen_c, "train_oof_macro_f1": round(oof_f1, 3)})

# %% [markdown]
# ## 4. Fine-tuning InLegalBERT
# The first 512 tokens of the view, class-weighted cross-entropy, 3 epochs, learning rate 2e-5 with warm-up and linear
# decay, gradient clipping 1.0. The **last epoch** is the reported result; the per-epoch dev scores are shown only to
# watch stability (picking the best epoch on dev would be tuning on the evaluation set).

# %% finetune
def finetune(train_frame, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(INLEGAL)
    model = AutoModelForSequenceClassification.from_pretrained(INLEGAL, num_labels=len(labels4)).to(device)
    encode = lambda texts: tokenizer(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    train_x, dev_x = encode(train_frame.view.tolist()), encode(dev.view.tolist())
    train_y = torch.tensor([label_index[l] for l in train_frame.label])
    counts = np.bincount(train_y.numpy(), minlength=len(labels4)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(labels4) * np.maximum(counts, 1)), device=device)
    steps = FT_EPOCHS * int(np.ceil(len(train_y) / FT_BATCH))
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    for epoch in range(1, FT_EPOCHS + 1):
        model.train(); order = np.random.permutation(len(train_y)); losses = []
        for start in range(0, len(order), FT_BATCH):
            picked = order[start:start + FT_BATCH]
            batch = {k: v[picked].to(device) for k, v in train_x.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), train_y[picked].to(device), weight=weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); schedule.step()
            losses.append(loss.item())
        model.eval(); probs = []
        with torch.no_grad():
            for start in range(0, len(dev_x["input_ids"]), 64):
                batch = {k: v[start:start + 64].to(device) for k, v in dev_x.items()}
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    probs.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
        proba = np.concatenate(probs)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 3), **score(dev.label, to_labels(proba)),
                        "predicted_classes": int(len(set(to_labels(proba))))})
        print(f"  seed {seed} epoch {epoch}: {history[-1]}", flush=True)
    del model, optimizer; free_gpu()
    return proba, history

finetune_history = {}
for setting, frame in settings.items():
    for seed in FT_SEEDS:
        t = time.time()
        proba, history = finetune(frame, seed)
        finetune_history[f"{setting}/seed{seed}"] = history
        record("InLegalBERT fine-tuned", setting, proba, seed=seed, extra={"seconds": round(time.time() - t)})
results["finetune_history"] = finetune_history

# %% [markdown]
# ## 5. Against the gate

# %% gate
table = pd.DataFrame(candidates)
display(table[["method", "setting", "seed", "acc", "macro_f1", "vs_tfidf_delta", "vs_tfidf_ci_low", "vs_tfidf_ci_high", "vs_tfidf_adopt"]])
verdict = []
for (method, setting), group in table.groupby(["method", "setting"], sort=False):
    verdict.append({"method": method, "setting": setting, "mean_macro_f1": round(group.macro_f1.mean(), 3),
                    "tfidf_macro_f1": reference[setting]["macro_f1"], "passes_gate": bool(group.vs_tfidf_adopt.all())})
verdict = pd.DataFrame(verdict)
display(verdict)
results["candidates"] = table.to_dict("records")
results["verdict"] = verdict.to_dict("records")
results["total_seconds"] = round(time.time() - started_all)
results["protocol"] = {"dev": "train 2013-14 or 2005-14, dev 2015", "min_gain": MIN_GAIN, "bootstraps": BOOTSTRAPS, "test_years_read": False}
(OUT / "phase3-results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")

fig, axis = plt.subplots(figsize=(8, 4))
shown = verdict.sort_values("mean_macro_f1")
axis.barh([f"{m} [{s}]" for m, s in zip(shown.method, shown.setting)], shown.mean_macro_f1, color="#2a6f97")
axis.axvline(reference["2013-14"]["macro_f1"], color="#c26b00", ls="--", label="TF-IDF (2013-14)")
axis.axvline(reference["2005-14"]["macro_f1"], color="#7a7a7a", ls=":", label="TF-IDF (2005-14)")
axis.set_xlabel("dev macro-F1 (2015, 4-way)"); axis.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "phase3.png", dpi=140); plt.show()
print(f"total {results['total_seconds']}s")
