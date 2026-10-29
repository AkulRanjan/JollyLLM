# %% [markdown]
# # Jolly Phase 2: Supreme Court judgments from the AWS bucket (CPU)
#
# Downloads each year's metadata and English PDF archives from the public bucket
# `s3://indian-supreme-court-judgments` (CC-BY-4.0), reads the PDF text layer and writes one gzipped JSONL file
# per year. Needs no GPU quota.
#
# Protocol (see `docs/score-improvement-plan.md`): 2005-2015 are for training and development. 2017 is the retired
# test year. 2018 is the fresh test year. This notebook prints **no label information** for 2017 and 2018.

# %% settings
import json, time
from pathlib import Path

TRAIN_DEV_YEARS = list(range(2005, 2016))  # 2005-2015
TEST_YEARS = [2017, 2018]  # counted, never labelled here
if "JOLLY_YEARS" in globals():  # set by a derived notebook (jolly-aws-extract-2019): test years only
    TRAIN_DEV_YEARS, TEST_YEARS = [], JOLLY_YEARS
YEARS = TRAIN_DEV_YEARS + TEST_YEARS  # 2016 is the SC-2016 year and is not taken from the bucket
PYMUPDF_VERSION = "1.28.2"  # the version the registry hashes were made with

# %% install
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"pymupdf=={PYMUPDF_VERSION}"], check=True)
import pymupdf
print("pymupdf", pymupdf.__doc__.split(":")[0])

# %% [markdown]
# ## Extract one year at a time
# Each year is written to its own file, so a stopped session can resume and later notebooks can read only the
# years they are allowed to see.

# %% extract
import legal_graph.aws_judgments as aws

OUT = Path("/kaggle/working/aws-sc") if ON_KAGGLE else Path(LOCAL_OUT)
OUT.mkdir(parents=True, exist_ok=True)
summaries = {}
for year in YEARS:
    target = OUT / f"records-{year}.jsonl.gz"
    summary_path = OUT / f"summary-{year}.json"
    if target.exists() and summary_path.exists():
        summaries[year] = json.loads(summary_path.read_text())
        print(year, "already extracted, skipped")
        continue
    started = time.time()
    summary = aws.extract_aws_years([year], WORK / "archives", target)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summaries[year] = summary
    print(f"{year}: {summary['record_count']} records in {time.time() - started:.0f}s", flush=True)

# %% [markdown]
# ## Check the archives against the hashes pinned in `configs/sources/aws-sc-source.json`
# The bucket is updated every two months, so a differing hash means the archive changed since 2026-10-11.

# %% verify
rows = []
for year, summary in summaries.items():
    for relative, info in summary["archives"].items():
        pinned = PINNED_SHA256.get(relative)
        rows.append({"year": year, "archive": relative.split("/")[0], "MB": round(info["bytes"] / 1e6, 1),
                     "sha256": info["sha256"][:16], "pinned": None if pinned is None else pinned[:16],
                     "matches_pin": None if pinned is None else pinned == info["sha256"]})
import pandas as pd
verify = pd.DataFrame(rows)
display(verify)
checked = verify.dropna(subset=["matches_pin"])
print(f"{int(checked.matches_pin.sum())} of {len(checked)} pinned archives match")

# %% [markdown]
# ## What was extracted
# Years up to 2015 also show how many judgments carry one of the four usable official disposal values and how
# many of those agree with the project's text rule on the last 20% of the text. 2017 and 2018 show counts only.

# %% stats
from legal_graph.outcomes import infer_disposition

def year_table(year):
    records = aws.read_aws_records(OUT / f"records-{year}.jsonl.gz")
    row = {"year": year, "records": len(records),
           "median_chars": int(pd.Series([len(r["text"]) for r in records]).median()),
           "empty_text": sum(not r["text"].strip() for r in records),
           "no_date": sum(not r["decision_date"] for r in records)}
    counts = summaries[year]["counts_by_year"][str(year)]
    row.update({key: counts.get(key, 0) for key in ("pdfs", "unreadable_pdf", "pdf_without_metadata")})
    if year in TRAIN_DEV_YEARS:
        official = [aws.official_label(r["disposal_nature"]) for r in records]
        row["official_four_way"] = sum(label is not None for label in official)
        row["text_agrees"] = sum(
            label is not None and infer_disposition(r["text"][int(len(r["text"]) * 0.8):]) == label
            for r, label in zip(records, official))
    return row

table = pd.DataFrame([year_table(year) for year in YEARS]).set_index("year")
display(table)

# %% plot
import matplotlib.pyplot as plt
fig, axis = plt.subplots(figsize=(9, 3.6))
axis.bar(table.index.astype(str), table["records"], color="#9aa5b1", label="records")
if "text_agrees" in table:
    axis.bar(table.index.astype(str), table["text_agrees"].fillna(0), color="#2a6f97", label="usable: official label = text rule")
axis.set_ylabel("judgments")
axis.set_title("AWS Supreme Court judgments per decision-year partition")
axis.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "extract-per-year.png", dpi=140)
plt.show()

# %% save
overview = {"years": YEARS, "pymupdf": PYMUPDF_VERSION, "table": json.loads(table.to_json(orient="index")),
            "verify": json.loads(verify.to_json(orient="records")),
            "records_sha256": {str(year): summaries[year]["records_sha256"] for year in YEARS}}
(OUT / "overview.json").write_text(json.dumps(overview, indent=2) + "\n")
print(sorted(path.name for path in OUT.iterdir()))
