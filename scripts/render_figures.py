"""Render every analysis figure for the graph-augmented legal LLM project as local PNGs.

Reads only committed local artifacts: the SC-2016 judgment corpus and graph, the statute
provision corpus and BM25 index, the citation extraction report, the measured outcome
baseline, and the outcome projection report. Writes PNGs under figures/ and nothing else.

    py -3 scripts/render_figures.py

Figures whose numbers come from the projection report are forecasts, not measurements.
Every such figure carries that statement in its footer.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_graph.outcomes import LABELS, infer_disposition  # noqa: E402

OUT = ROOT / "figures"
MEASURED = "#eb6834"
PROJECTED = "#2a78d6"
PRIMARY = "#1baf7a"
BOUND = "#79848f"
INK = "#101720"
MUTED = "#6b7480"
GRID = "#dfe4e9"
BLUES = LinearSegmentedColormap.from_list("legalblues", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])
CLASS_COLORS = {"allowed": "#2a78d6", "dismissed": "#eb6834", "disposed": "#1baf7a", "partly_allowed": "#4a3aa7"}
PRETTY = {"allowed": "allowed", "dismissed": "dismissed", "disposed": "disposed", "partly_allowed": "partly allowed"}
SPLITS = ("train", "validation", "test")
SPLIT_TITLE = {"train": "train", "validation": "validation", "test": "held-out test"}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#b8c0c8",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    }
)


# ---------------------------------------------------------------- helpers


def tidy(ax, *, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(False)
        ax.grid(True, axis=grid_axis)
    return ax


def footer(fig, text, *, y=0.012):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8, color=MUTED, wrap=True)


def save(fig, folder, name):
    target = OUT / folder
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- data load


def load_everything():
    judgments = read_jsonl(ROOT / "data/clean/judgments/sc-2016/judgments.jsonl")
    for row in judgments:
        text = row["text"]
        row["_length"] = len(text)
        row["_label"] = infer_disposition(text[int(len(text) * 0.8) :])
        date = str(row.get("decision_date") or "")
        row["_month"] = int(date[5:7]) if len(date) == 10 else None
    data = {
        "judgments": judgments,
        "manifest": read_json(ROOT / "data/clean/judgments/training.manifest.json"),
        "judgment_graph": read_json(ROOT / "data/graphs/judgments/sc-2016/graph.json"),
        "statute_graph": read_json(ROOT / "data/graphs/india_statutes/graph.json"),
        "provisions": read_jsonl(ROOT / "data/clean/india_statutes/provisions.jsonl"),
        "bm25": read_json(ROOT / "data/indexes/india_statutes/bm25/bm25.index.json"),
        "extraction": read_json(ROOT / "data/reports/judgments/sc-2016/extraction-report.json"),
        "measured": read_json(ROOT / "data/reports/evaluation/sc2016-outcome-baseline-v1/validation-metrics.json"),
        "projection": read_json(ROOT / "data/reports/evaluation/sc2016-outcome-projection-v1/outcome-projection.json"),
    }
    return data


def supports(projection, split):
    return {label: projection["label_supports"][split][label] for label in LABELS}


def variant(projection, variant_id):
    return next(v for v in projection["variants"] if v["variant_id"] == variant_id)


# ================================================================ corpus


def fig1_corpus_overview(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    counts = data["manifest"]["split_counts"]
    proj = data["projection"]

    ax = tidy(axes[0][0])
    bars = ax.bar([SPLIT_TITLE[s] for s in SPLITS], [counts[s] for s in SPLITS], color=[PROJECTED, MEASURED, PRIMARY], width=0.6)
    for bar, split in zip(bars, SPLITS):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, f"{counts[split]}", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_title("(a) Judgments per split")
    ax.set_ylabel("documents")
    ax.set_ylim(0, max(counts.values()) * 1.16)
    ax.text(0.98, 0.94, f"{sum(counts.values())} total\nsha256-prefix-modulo-10 split", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)

    ax = tidy(axes[0][1])
    width = 0.26
    positions = np.arange(len(LABELS))
    for offset, split in zip((-width, 0, width), SPLITS):
        values = [proj["label_supports"][split][label] for label in LABELS]
        ax.bar(positions + offset, values, width=width * 0.92, label=SPLIT_TITLE[split],
               color={"train": PROJECTED, "validation": MEASURED, "test": PRIMARY}[split])
    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[label] for label in LABELS])
    ax.set_title("(b) Recoverable disposition labels")
    ax.set_ylabel("cases")
    ax.set_yscale("log")
    ax.set_ylim(3, 620)
    ax.legend(loc="upper right", ncols=3, fontsize=8.5)

    ax = tidy(axes[1][0])
    lengths = [row["_length"] for row in data["judgments"]]
    ax.hist(lengths, bins=np.logspace(math.log10(min(lengths)), math.log10(max(lengths)), 34), color=PROJECTED, edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    median = statistics.median(lengths)
    ax.axvline(median, color=MEASURED, linewidth=1.8)
    ax.set_title("(c) Judgment length")
    ax.set_xlabel("characters (log scale)")
    ax.set_ylabel("documents")
    ax.text(0.02, 0.96, f"median {median:,.0f} chars", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MEASURED, fontweight="bold")
    ax.text(0.02, 0.88, f"max {max(lengths):,} chars", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)

    ax = tidy(axes[1][1])
    labelled = [sum(proj["label_supports"][s][label] for label in LABELS) for s in SPLITS]
    missing = [proj["label_supports"][s]["_unlabelled"] for s in SPLITS]
    names = [SPLIT_TITLE[s] for s in SPLITS]
    ax.barh(names, labelled, color=PRIMARY, label="explicit disposition phrase")
    ax.barh(names, missing, left=labelled, color="#d9dee4", label="no explicit phrase")
    for i, split in enumerate(SPLITS):
        total = counts[split]
        ax.text(total + 6, i, f"{labelled[i] / total:.1%} evaluable", va="center", fontsize=9, fontweight="bold", color=INK)
    ax.set_title("(d) Evaluable share of each split")
    ax.set_xlabel("documents")
    ax.set_xlim(0, max(counts.values()) * 1.42)
    ax.legend(loc="upper right", fontsize=8.5)
    tidy(ax, grid_axis="x")

    fig.suptitle("SC-2016 judgment corpus", fontsize=14, fontweight="bold", y=0.98)
    footer(fig, "All four panels are measured properties of the validated local corpus. A label is recoverable when the final 20% of the judgment states an explicit disposition phrase.")
    save(fig, "corpus", "fig1_corpus_overview.png")


def fig2_label_imbalance(data):
    proj = data["projection"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = tidy(axes[0])
    bottom = np.zeros(len(SPLITS))
    for label in LABELS:
        shares = np.array([proj["label_supports"][s][label] for s in SPLITS], dtype=float)
        totals = np.array([sum(proj["label_supports"][s][l] for l in LABELS) for s in SPLITS], dtype=float)
        shares = shares / totals
        ax.bar([SPLIT_TITLE[s] for s in SPLITS], shares, bottom=bottom, color=CLASS_COLORS[label], label=PRETTY[label], width=0.55, edgecolor="white", linewidth=1.2)
        for i, value in enumerate(shares):
            if value > 0.055:
                ax.text(i, bottom[i] + value / 2, f"{value:.0%}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        bottom += shares
    ax.set_title("(a) Disposition mix per split")
    ax.set_ylabel("share of evaluable cases")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=4, fontsize=8.5)

    ax = tidy(axes[1])
    test = supports(proj, "test")
    total = sum(test.values())
    positions = np.arange(len(LABELS))
    bars = ax.bar(positions, [test[label] for label in LABELS], color=[CLASS_COLORS[label] for label in LABELS], width=0.58)
    for bar, label in zip(bars, LABELS):
        n = test[label]
        ax.text(bar.get_x() + bar.get_width() / 2, n + 0.5, f"{n}", ha="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(bar.get_x() + bar.get_width() / 2, n / 2, f"1 case\n= {100 / n:.0f} pts\nof recall", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.axhline(total / len(LABELS), color=MUTED, linewidth=1.2, linestyle="-")
    ax.text(len(LABELS) - 0.5, total / len(LABELS) + 0.6, "balanced would be 12.5", ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[label] for label in LABELS])
    ax.set_title(f"(b) Held-out support: {total} evaluable cases")
    ax.set_ylabel("cases")
    ax.set_ylim(0, max(test.values()) * 1.25)

    prior = max(test.values()) / total
    fig.suptitle("Class imbalance is the dominant constraint on this task", fontsize=13.5, fontweight="bold", y=1.0)
    footer(fig, f"Measured from the corpus. Always predicting the largest class scores {prior:.1%} accuracy on the held-out split, which is why macro-F1 is the plan's primary metric.", y=-0.06)
    save(fig, "corpus", "fig2_label_imbalance.png")


def fig3_graph_structure(data):
    graph = data["judgment_graph"]
    nodes = Counter(n.get("node_type") for n in graph["nodes"])
    edges = Counter(e.get("relation_type") for e in graph["edges"])
    refs = Counter()
    for edge in graph["edges"]:
        if edge.get("relation_type") == "refs_to_provision" or edge.get("relation_type") == "refers_to_provision":
            refs[edge["source_id"]] += 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    ax = tidy(axes[0], grid_axis="x")
    items = nodes.most_common()
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=PROJECTED, height=0.6)
    for i, (_, value) in enumerate(items[::-1]):
        ax.text(value + max(nodes.values()) * 0.02, i, f"{value:,}", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_title(f"(a) Nodes ({len(graph['nodes']):,})")
    ax.set_xlim(0, max(nodes.values()) * 1.2)
    ax.set_xlabel("count")

    ax = tidy(axes[1], grid_axis="x")
    items = edges.most_common()
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=PRIMARY, height=0.6)
    for i, (_, value) in enumerate(items[::-1]):
        ax.text(value + max(edges.values()) * 0.02, i, f"{value:,}", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_title(f"(b) Typed edges ({len(graph['edges']):,})")
    ax.set_xlim(0, max(edges.values()) * 1.2)
    ax.set_xlabel("count")

    ax = tidy(axes[2])
    cases = nodes.get("case", 0)
    distribution = Counter(refs.values())
    with_refs = sum(distribution.values())
    buckets = [0] + sorted(distribution)
    heights = [cases - with_refs] + [distribution[k] for k in sorted(distribution)]
    colors = ["#d9dee4"] + [MEASURED] * (len(buckets) - 1)
    ax.bar([str(b) for b in buckets], heights, color=colors, width=0.65)
    for i, value in enumerate(heights):
        ax.text(i, value + cases * 0.015, str(value), ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.set_title("(c) Resolved provision references per case")
    ax.set_xlabel("references")
    ax.set_ylabel("cases")
    ax.set_ylim(0, cases * 1.12)
    ax.text(0.97, 0.9, f"{with_refs} of {cases} cases\nreach a statute node", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)

    fig.suptitle("Judgment graph snapshot: what the graph channel can actually retrieve", fontsize=13.5, fontweight="bold", y=1.02)
    footer(fig, "Measured from the validated graph snapshot. The statute-reference channel is sparse: most cases connect only through court and provision-containment edges.", y=-0.08)
    save(fig, "corpus", "fig3_graph_structure.png")


def fig4_topics_judges(data):
    topics = Counter()
    judges = Counter()
    for row in data["judgments"]:
        attributes = row.get("attributes", {})
        for topic in (attributes.get("topics") or "").split("|"):
            topic = topic.strip()
            if topic:
                topics[topic] += 1
        for judge in (attributes.get("judge_names") or "").split("|"):
            judge = judge.strip()
            if judge:
                judges[judge] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))

    ax = tidy(axes[0], grid_axis="x")
    top = topics.most_common(15)[::-1]
    ax.barh([k for k, _ in top], [v for _, v in top], color=PROJECTED, height=0.66)
    for i, (_, value) in enumerate(top):
        ax.text(value + 0.5, i, str(value), va="center", fontsize=9, fontweight="bold", color=INK)
    ax.set_title(f"(a) Most frequent subject tags ({len(topics):,} distinct)")
    ax.set_xlabel("judgments")
    ax.set_xlim(0, top[-1][1] * 1.15)

    ax = tidy(axes[1], grid_axis="x")
    top = judges.most_common(15)[::-1]
    ax.barh([k for k, _ in top], [v for _, v in top], color=PRIMARY, height=0.66)
    for i, (_, value) in enumerate(top):
        ax.text(value + 1, i, str(value), va="center", fontsize=9, fontweight="bold", color=INK)
    ax.set_title(f"(b) Most frequent judges ({len(judges)} on the bench)")
    ax.set_xlabel("judgments")
    ax.set_xlim(0, top[-1][1] * 1.15)

    fig.suptitle("Corpus composition: subject matter and bench", fontsize=13.5, fontweight="bold", y=0.99)
    footer(fig, f"Measured from ingested judgment metadata. Tag vocabulary is long-tailed: {len(topics):,} tags across {len(data['judgments'])} judgments, so tags are weak features without normalisation.", y=-0.03)
    save(fig, "corpus", "fig4_topics_judges.png")


def fig5_length_and_time(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    labelled = [row for row in data["judgments"] if row["_label"] in LABELS]

    ax = tidy(axes[0])
    groups = [[row["_length"] for row in labelled if row["_label"] == label] for label in LABELS]
    box = ax.boxplot(groups, patch_artist=True, widths=0.55, medianprops={"color": "white", "linewidth": 1.8},
                     flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.5})
    for patch, label in zip(box["boxes"], LABELS):
        patch.set_facecolor(CLASS_COLORS[label])
        patch.set_edgecolor("none")
    for whisker in box["whiskers"] + box["caps"]:
        whisker.set_color("#98a1ab")
    ax.set_yscale("log")
    ax.set_xticks(range(1, len(LABELS) + 1))
    ax.set_xticklabels([f"{PRETTY[label]}\nn={len(group)}" for label, group in zip(LABELS, groups)])
    ax.set_title("(a) Judgment length by disposition")
    ax.set_ylabel("characters (log scale)")

    ax = tidy(axes[1])
    months = range(1, 13)
    monthly = {label: [sum(1 for row in labelled if row["_month"] == m and row["_label"] == label) for m in months] for label in LABELS}
    bottom = np.zeros(12)
    for label in LABELS:
        values = np.array(monthly[label], dtype=float)
        ax.bar(list(months), values, bottom=bottom, color=CLASS_COLORS[label], label=PRETTY[label], width=0.7, edgecolor="white", linewidth=0.8)
        bottom += values
    ax.set_xticks(list(months))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    undated = sum(1 for row in labelled if row["_month"] is None)
    ax.set_title("(b) Disposition mix through 2016")
    ax.set_ylabel("judgments")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4, fontsize=8.5)
    ax.text(0.98, 0.94, f"{undated} labelled judgments carry no parsed\ndecision date and are omitted from this panel",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)

    fig.suptitle("Length and time structure of the labelled subset", fontsize=13.5, fontweight="bold", y=1.0)
    footer(fig, "Measured. Length separates the classes only weakly, and no month is label-free, so a temporal split would not remove the imbalance problem.", y=-0.11)
    save(fig, "corpus", "fig5_length_and_time.png")


def fig6_citation_resolution(data):
    report = data["extraction"]
    summary = report["summary"]
    unresolved_kinds = Counter(item.get("kind", "unknown") for item in report["unresolved"])
    per_document = Counter(item["document_id"] for item in report["unresolved"])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

    ax = tidy(axes[0], grid_axis="x")
    resolved = summary["resolved"]
    total = resolved + summary["unresolved"] + summary["ambiguous"]
    ax.barh(["citation\ncandidates"], [resolved], color=PRIMARY, height=0.42, label=f"resolved to a graph node ({resolved:,})")
    ax.barh(["citation\ncandidates"], [summary["unresolved"]], left=[resolved], color="#d9dee4", height=0.42, label=f"no canonical target ({summary['unresolved']:,})")
    ax.set_xlim(0, total * 1.02)
    ax.set_title(f"(a) Citation resolution: {resolved / total:.1%} of {total:,} candidates")
    ax.set_xlabel("extraction candidates")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(resolved + summary["unresolved"] / 2, 0, "cited authorities outside\nthe 2016 corpus window", ha="center", va="center", fontsize=9, color=INK, fontweight="bold")

    ax = tidy(axes[1])
    kinds = unresolved_kinds.most_common()
    bars = ax.bar([k for k, _ in kinds], [v for _, v in kinds], color=MEASURED, width=0.45)
    for bar, (_, value) in zip(bars, kinds):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(unresolved_kinds.values()) * 0.02, f"{value:,}", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_title("(b) Unresolved candidates by kind")
    ax.set_ylabel("candidates")
    ax.set_ylim(0, max(unresolved_kinds.values()) * 1.18)
    median_per_doc = statistics.median(per_document.values())
    ax.text(0.97, 0.72, f"{len(per_document)} judgments carry\nunresolved citations\nmedian {median_per_doc:.0f} per judgment",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED)

    fig.suptitle("The citation graph is mostly unresolved, and that caps the graph channel", fontsize=13.5, fontweight="bold", y=1.01)
    footer(fig, "Measured from the local extraction report. Unresolved candidates are retained as evidence, never silently dropped; resolving them needs corpus years beyond 2016.", y=-0.07)
    save(fig, "corpus", "fig6_citation_resolution.png")


def fig7_statute_corpus(data):
    provisions = data["provisions"]
    bm25 = data["bm25"]
    per_act = Counter(row["act_id"] for row in provisions)
    titles = {row["act_id"]: row["act_title"] for row in provisions}
    lengths = [len(row["text"]) for row in provisions]
    tokens = list(bm25["document_lengths"])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = tidy(axes[0])
    items = per_act.most_common()
    bars = ax.bar([k.upper() for k, _ in items], [v for _, v in items], color=PROJECTED, width=0.55)
    for bar, (_, value) in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, str(value), ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_title(f"(a) Provisions per act ({len(provisions):,} total)")
    ax.set_ylabel("provisions")
    ax.set_ylim(0, max(per_act.values()) * 1.16)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{k.upper()}\n{textwrap.fill(titles[k], 20)}" for k, _ in items], fontsize=7.5)

    ax = tidy(axes[1])
    ax.hist(lengths, bins=np.logspace(math.log10(max(1, min(lengths))), math.log10(max(lengths)), 32), color=PRIMARY, edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.axvline(statistics.median(lengths), color=MEASURED, linewidth=1.8)
    ax.text(statistics.median(lengths) * 1.12, ax.get_ylim()[1] * 0.88, f"median {statistics.median(lengths):,.0f}", color=MEASURED, fontsize=8.5, fontweight="bold")
    ax.set_title("(b) Provision text length")
    ax.set_xlabel("characters (log scale)")
    ax.set_ylabel("provisions")

    ax = tidy(axes[2])
    ax.hist(tokens, bins=40, color=BOUND, edgecolor="white", linewidth=0.6)
    average = bm25["average_document_length"]
    ax.axvline(average, color=MEASURED, linewidth=1.8)
    ax.text(average * 1.1, ax.get_ylim()[1] * 0.88, f"avg {average:.1f} tokens", color=MEASURED, fontsize=8.5, fontweight="bold")
    ax.set_title(f"(c) BM25 index: {len(bm25['document_frequencies']):,} term vocabulary")
    ax.set_xlabel("tokens per provision")
    ax.set_ylabel("provisions")

    fig.suptitle("Statute retrieval corpus (BNS, BNSS, CrPC)", fontsize=13.5, fontweight="bold", y=1.02)
    footer(fig, "Measured from the approved statute extraction and the BM25 manifest. This is the evidence pool the retrieval stage draws from.", y=-0.08)
    save(fig, "corpus", "fig7_statute_corpus.png")


# ================================================================ projection


def ladder_rows(projection):
    return projection["variants"]


def style_for(row, projection):
    if row["family"] == "ceiling":
        return BOUND, "diagnostic bound"
    if row["evidence"] == "measured":
        return MEASURED, "measured"
    if row["variant_id"] == projection["primary_variant_id"]:
        return PRIMARY, "projected (primary)"
    return PROJECTED, "projected"


def fig8_variant_ladder(data):
    projection = data["projection"]
    rows = ladder_rows(projection)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    tidy(ax, grid_axis="x")

    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        test = row["test"] if "test" in row else row["splits"]["test"]
        accuracy = test["accuracy"]
        lower = test["accuracy_interval"]["lower"]
        upper = test["accuracy_interval"]["upper"]
        color, kind = style_for(row, projection)
        filled = row["evidence"] == "measured"
        ax.barh(position, accuracy, height=0.56, edgecolor=color, linewidth=0 if filled else 2.0,
                facecolor=color if filled else (*matplotlib.colors.to_rgb(color), 0.18))
        ax.errorbar(accuracy, position, xerr=[[accuracy - lower], [upper - accuracy]], fmt="none",
                    ecolor="#5d6670", elinewidth=1.3, capsize=4, capthick=1.3, alpha=0.85)
        ax.text(upper + 0.012, position, f"{accuracy:.1%}", va="center", fontsize=10, fontweight="bold", color=INK)

    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("accuracy on the 50 evaluable held-out cases")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]
    ax.axvline(primary["accuracy"], color=PRIMARY, linewidth=1.4, alpha=0.55)
    ax.text(primary["accuracy"], len(rows) - 0.35, f"primary {primary['accuracy']:.1%}", ha="center", fontsize=9.5, fontweight="bold", color=PRIMARY)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MEASURED, edgecolor="none", label="measured (solid)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=(*matplotlib.colors.to_rgb(PROJECTED), 0.18), edgecolor=PROJECTED, linewidth=2, label="projected (outlined)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=(*matplotlib.colors.to_rgb(PRIMARY), 0.18), edgecolor=PRIMARY, linewidth=2, label="primary configuration"),
        plt.Rectangle((0, 0), 1, 1, facecolor=(*matplotlib.colors.to_rgb(BOUND), 0.18), edgecolor=BOUND, linewidth=2, label="diagnostic bound"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncols=4, fontsize=8.8)

    fig.suptitle("Projected held-out accuracy for every planned configuration", fontsize=13.5, fontweight="bold", y=0.97)
    footer(fig, "Whiskers are 95% Wilson intervals on 50 cases. Only the two solid bars are measurements; the rest are forecasts from the projection config, and every interval is wider than the gaps between neighbouring rows.", y=-0.09)
    save(fig, "projection", "fig8_variant_ladder.png")


def fig9_accuracy_vs_macro_f1(data):
    projection = data["projection"]
    rows = ladder_rows(projection)
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    tidy(ax, grid_axis="x")

    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        test = row["splits"]["test"]
        accuracy, macro = test["accuracy"], test["macro_f1"]
        ax.plot([macro, accuracy], [position, position], color="#c9d1d9", linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter(macro, position, s=78, color=MEASURED, zorder=3, edgecolor="white", linewidth=1.4)
        ax.scatter(accuracy, position, s=78, color=PROJECTED, zorder=3, edgecolor="white", linewidth=1.4)
        ax.text(1.005, position, f"−{(accuracy - macro) * 100:.1f} pts", va="center", fontsize=9, fontweight="bold", color=MUTED)

    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=9.5)
    ax.set_xlim(0.15, 1.0)
    ax.set_xlabel("score on the held-out split")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.scatter([], [], s=78, color=PROJECTED, label="accuracy")
    ax.scatter([], [], s=78, color=MEASURED, label="macro-F1")
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Accuracy overstates every configuration on this corpus", fontsize=13.5, fontweight="bold", y=0.98)
    footer(fig, "The gap is class imbalance: half the held-out cases are 'allowed', so accuracy rewards a system that never learns the two small classes. Macro-F1 is the plan's primary metric.", y=-0.03)
    save(fig, "projection", "fig9_accuracy_vs_macro_f1.png")


def fig10_uplift_waterfall(data):
    projection = data["projection"]
    steps = projection["uplift_decomposition"]
    start = projection["measured_baseline"]["accuracy"]

    fig, ax = plt.subplots(figsize=(12, 5.6))
    tidy(ax)

    labels = ["Naive Bayes\n(measured)"] + [textwrap.fill(step["label"].replace("+ ", ""), 17) for step in steps]

    # the graph channel is every step from just after the text-only comparison through the primary
    order = [step["variant_id"] for step in steps]
    first_graph = order.index(projection["separation_power"]["comparison_variant_id"]) + 2
    last_graph = order.index(projection["primary_variant_id"]) + 1
    graph_indices = list(range(first_graph, last_graph + 1))
    ax.axvspan(first_graph - 0.45, last_graph + 0.45, color=PRIMARY, alpha=0.07, zorder=0)

    ax.bar(0, start, color=MEASURED, width=0.55)
    ax.text(0, start / 2, f"{start:.1%}", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    for index, step in enumerate(steps, start=1):
        low = min(step["from_accuracy"], step["to_accuracy"])
        height = abs(step["to_accuracy"] - step["from_accuracy"])
        is_primary = step["variant_id"] == projection["primary_variant_id"]
        color = PRIMARY if is_primary else PROJECTED
        ax.bar(index, height, bottom=low, color=(*matplotlib.colors.to_rgb(color), 0.22), edgecolor=color, linewidth=2, width=0.55)
        ax.text(index, step["to_accuracy"] + 0.036, f"+{step['delta'] * 100:.1f}", ha="center", fontsize=9.5, fontweight="bold", color=color)
        ax.text(index, step["to_accuracy"] + 0.014, f"{step['to_accuracy']:.1%}", ha="center", fontsize=8.5, color=INK)
        ax.plot([index - 0.28, index + 0.83], [step["to_accuracy"]] * 2, color="#c9d1d9", linewidth=1, zorder=0)

    ax.plot([-0.28, 0.83], [start] * 2, color="#c9d1d9", linewidth=1, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("held-out accuracy")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    text_only = variant(projection, projection["separation_power"]["comparison_variant_id"])["splits"]["test"]["accuracy"]
    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]["accuracy"]
    ax.text(sum(graph_indices) / len(graph_indices), 0.055,
            f"graph channel\n+{(primary - text_only) * 100:.1f} pts of the +{(primary - start) * 100:.1f} total",
            ha="center", fontsize=9.5, color=PRIMARY, fontweight="bold")

    fig.suptitle("What each stage of the plan is forecast to contribute", fontsize=13.5, fontweight="bold", y=0.98)
    footer(fig, "Only the leftmost bar is measured. Stage attribution is a forecast: the two graph stages carry under half the projected gain, and each step is smaller than the 95% interval around it.", y=-0.08)
    save(fig, "projection", "fig10_uplift_waterfall.png")


def fig11_learning_curve(data):
    projection = data["projection"]
    curve = projection["learning_curve"]
    counts = np.array([point["labelled_cases"] for point in curve], dtype=float)
    values = np.array([point["projected_accuracy"] for point in curve], dtype=float)
    anchor = next(point for point in curve if point["is_current_corpus"])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    tidy(ax, grid_axis="both")

    ax.plot(counts, values, color=PROJECTED, linewidth=2.2, zorder=2)
    ax.scatter(counts, values, s=44, color=PROJECTED, zorder=3, edgecolor="white", linewidth=1.3)
    ax.scatter([anchor["labelled_cases"]], [anchor["projected_accuracy"]], s=150, color=PRIMARY, zorder=4, edgecolor="white", linewidth=1.8)
    ax.axvline(anchor["labelled_cases"], color=PRIMARY, linewidth=1.2, alpha=0.45)
    ax.annotate(f"SC-2016 today\n{anchor['labelled_cases']} labelled cases → {anchor['projected_accuracy']:.1%}",
                xy=(anchor["labelled_cases"], anchor["projected_accuracy"]), xytext=(70, 0.80),
                fontsize=9.5, color=PRIMARY, fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": PRIMARY, "linewidth": 1.3, "shrinkA": 2, "shrinkB": 8})

    for point in curve:
        if point["labelled_cases"] in (50, 1000, 5000):
            ax.text(point["labelled_cases"], point["projected_accuracy"] - 0.028, f"{point['projected_accuracy']:.1%}",
                    ha="center", fontsize=9, fontweight="bold", color=PROJECTED)

    measured = projection["measured_baseline"]["accuracy"]
    ax.axhline(measured, color=MEASURED, linewidth=1.6, linestyle="-")
    ax.text(4700, measured + 0.012, f"measured baseline {measured:.1%}", ha="right", fontsize=9, color=MEASURED, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("labelled training cases (log scale)")
    ax.set_ylabel("projected accuracy")
    ax.set_ylim(0.4, 0.9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k"])

    fig.suptitle("Supervision is the binding constraint, not architecture", fontsize=13.5, fontweight="bold", y=0.98)
    footer(fig, "Forecast. Power-law curve pinned to the projection at today's 359 labelled cases; the asymptote assumes fixed retrieval quality. Ingesting more Supreme Court years moves the number more than any remaining configuration change.", y=-0.04)
    save(fig, "projection", "fig11_learning_curve.png")


def fig12_per_class_f1(data):
    projection = data["projection"]
    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]
    measured = projection["measured_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = tidy(axes[0])
    positions = np.arange(len(LABELS))
    width = 0.36
    measured_values = [measured["per_class"][label]["f1"] for label in LABELS]
    projected_values = [primary["per_class"][label]["f1"] for label in LABELS]
    ax.bar(positions - width / 2, measured_values, width=width, color=MEASURED, label="Naive Bayes, measured")
    ax.bar(positions + width / 2, projected_values, width=width, color=(*matplotlib.colors.to_rgb(PRIMARY), 0.2), edgecolor=PRIMARY, linewidth=2, label="graph-augmented, projected")
    for position, value in zip(positions - width / 2, measured_values):
        ax.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=9, fontweight="bold", color=MEASURED)
    for position, value in zip(positions + width / 2, projected_values):
        ax.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=9, fontweight="bold", color=PRIMARY)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{PRETTY[label]}\nn={primary['per_class'][label]['support']}" for label in LABELS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("(a) Per-class F1")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = tidy(axes[1])
    recall = [primary["per_class"][label]["recall"] for label in LABELS]
    precision = [primary["per_class"][label]["precision"] for label in LABELS]
    ax.bar(positions - width / 2, recall, width=width, color=PROJECTED, label="recall")
    ax.bar(positions + width / 2, precision, width=width, color=BOUND, label="precision")
    for position, value in zip(positions - width / 2, recall):
        ax.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=9, fontweight="bold", color=PROJECTED)
    for position, value in zip(positions + width / 2, precision):
        ax.text(position, value + 0.02, f"{value:.2f}", ha="center", fontsize=9, fontweight="bold", color=BOUND)
    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[label] for label in LABELS])
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) Projected recall and precision, primary configuration")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("The two small classes stay weak in every forecast", fontsize=13.5, fontweight="bold", y=1.0)
    footer(fig, f"Panel (a) compares a measurement against a forecast. 'partly allowed' has {primary['per_class']['partly_allowed']['support']} held-out cases, so a single case moves its recall by 20 points.", y=-0.07)
    save(fig, "projection", "fig12_per_class_f1.png")


def draw_confusion(ax, matrix, title, *, accent):
    grid = np.array([[matrix[actual][predicted] for predicted in LABELS] for actual in LABELS], dtype=float)
    ax.imshow(grid, cmap=BLUES, vmin=0, vmax=grid.max())
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([PRETTY[label] for label in LABELS], rotation=22, ha="right", fontsize=9)
    ax.set_yticklabels([f"{PRETTY[label]}\nn={int(grid[i].sum())}" for i, label in enumerate(LABELS)], fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true disposition")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = grid.max() * 0.55
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = int(grid[i][j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if grid[i][j] > threshold else INK)
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=accent, linewidth=2.4))
    correct = int(np.trace(grid))
    total = int(grid.sum())
    ax.set_title(f"{title}\n{correct}/{total} correct · {correct / total:.1%} accuracy")


def fig13_confusion_matrices(data):
    projection = data["projection"]
    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    draw_confusion(axes[0], projection["measured_baseline"]["confusion_matrix"], "Naive Bayes, measured on validation", accent=MEASURED)
    draw_confusion(axes[1], primary["confusion_matrix"], "Graph-augmented, projected on held-out test", accent=PRIMARY)
    fig.suptitle("Error structure: measured today against the forecast", fontsize=13.5, fontweight="bold", y=1.0)
    footer(fig, "Left panel is a measurement on 47 validation cases; right panel is a forecast on 50 held-out cases. Most remaining projected error sits on the allowed / partly-allowed boundary, where the disposition language is closest.", y=-0.06)
    save(fig, "projection", "fig13_confusion_matrices.png")


def fig14_sampling_noise(data):
    noise = data["projection"]["sampling_noise"]
    accuracies = np.array([bin_["accuracy"] for bin_ in noise["histogram"]])
    shares = np.array([bin_["share"] for bin_ in noise["histogram"]])

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    tidy(ax)

    ax.axvspan(noise["p05"], noise["p95"], color=PROJECTED, alpha=0.10, zorder=0,
               label=f"90% of runs: {noise['p05']:.0%} – {noise['p95']:.0%}")
    ax.bar(accuracies, shares, width=0.016, color=PROJECTED, zorder=2)
    ax.axvline(noise["assumed_accuracy"], color=PRIMARY, linewidth=2.4, zorder=3,
               label=f"true accuracy {noise['assumed_accuracy']:.1%}")
    ax.axvline(noise["p05"], color=PROJECTED, linewidth=1.2, linestyle="-", alpha=0.7)
    ax.axvline(noise["p95"], color=PROJECTED, linewidth=1.2, linestyle="-", alpha=0.7)

    ax.set_xlabel(f"accuracy a single {noise['sample_size']}-case evaluation would report")
    ax.set_ylabel("share of simulated runs")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.97, 0.93, f"{noise['trials']:,} simulated runs\nonly {noise['within_5_points']:.0%} land within ±5 points",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK, fontweight="bold")

    fig.suptitle("What a single 50-case evaluation would actually report", fontsize=13.5, fontweight="bold", y=0.98)
    footer(fig, "Simulated. Each run draws 50 independent outcomes at the projected accuracy, so this spread is pure measurement noise — the model is held fixed and perfect knowledge of it would not narrow the band.", y=-0.03)
    save(fig, "projection", "fig14_sampling_noise.png")


def fig15_statistical_power(data):
    power = data["projection"]["separation_power"]
    projection = data["projection"]
    sizes = [point["sample_size"] for point in power["points"]]
    probabilities = [point["probability_primary_ranks_higher"] for point in power["points"]]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    tidy(ax, grid_axis="both")

    ax.plot(range(len(sizes)), probabilities, color=PROJECTED, linewidth=2.2, zorder=2)
    ax.scatter(range(len(sizes)), probabilities, s=52, color=PROJECTED, zorder=3, edgecolor="white", linewidth=1.3)
    ax.scatter([0], [probabilities[0]], s=155, color=PRIMARY, zorder=4, edgecolor="white", linewidth=1.8)
    for index, (size, probability) in enumerate(zip(sizes, probabilities)):
        ax.text(index, probability + 0.022, f"{probability:.0%}", ha="center", fontsize=10, fontweight="bold",
                color=PRIMARY if index == 0 else PROJECTED)

    ax.axhline(0.95, color=MEASURED, linewidth=1.6)
    ax.text(len(sizes) - 1, 0.958, "95% confidence in the ranking", ha="right", fontsize=9, color=MEASURED, fontweight="bold")
    ax.annotate("today's held-out split", xy=(0, probabilities[0]), xytext=(0.55, 0.63), fontsize=9.5,
                color=PRIMARY, fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "color": PRIMARY, "linewidth": 1.3, "shrinkA": 2, "shrinkB": 8})

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("evaluable held-out cases available for the comparison")
    ax.set_ylabel("probability the graph configuration ranks higher")
    ax.set_ylim(0.5, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]["accuracy"]
    fig.suptitle("Test-set size needed to rank the graph channel correctly", fontsize=13.5, fontweight="bold", y=0.98)
    footer(fig, f"Simulated, assuming the forecast {primary:.1%} against {power['comparison_accuracy']:.1%} for the text-only configuration. Extending the corpus buys statistical power and accuracy from the same work.", y=-0.03)
    save(fig, "projection", "fig15_statistical_power.png")


def fig16_projection_summary(data):
    projection = data["projection"]
    rows = ladder_rows(projection)
    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]
    measured = projection["measured_baseline"]
    noise = projection["sampling_noise"]
    power = projection["separation_power"]
    curve = projection["learning_curve"]

    fig = plt.figure(figsize=(16, 10))
    spec = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.30, top=0.90, bottom=0.07)

    ax = fig.add_subplot(spec[0, :2])
    tidy(ax, grid_axis="x")
    positions = np.arange(len(rows))[::-1]
    for position, row in zip(positions, rows):
        test = row["splits"]["test"]
        color, _ = style_for(row, projection)
        filled = row["evidence"] == "measured"
        ax.barh(position, test["accuracy"], height=0.6, color=color if filled else (*matplotlib.colors.to_rgb(color), 0.18),
                edgecolor=color, linewidth=0 if filled else 1.8)
        ax.errorbar(test["accuracy"], position, xerr=[[test["accuracy"] - test["accuracy_interval"]["lower"]],
                                                      [test["accuracy_interval"]["upper"] - test["accuracy"]]],
                    fmt="none", ecolor="#5d6670", elinewidth=1, capsize=3, alpha=0.8)
        ax.text(test["accuracy_interval"]["upper"] + 0.012, position, f"{test['accuracy']:.0%}", va="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Variant ladder with 95% intervals", fontsize=10.5)

    ax = fig.add_subplot(spec[0, 2])
    ax.axis("off")
    gain = primary["accuracy"] - measured["accuracy"]
    half = projection["headline"]["accuracy_interval"]["half_width"]
    ax.text(0.0, 0.94, "MEASURED TODAY", fontsize=8.5, color=MUTED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.76, f"{measured['accuracy']:.1%}", fontsize=30, color=MEASURED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.68, f"Naive Bayes · macro-F1 {measured['macro_f1']:.3f} · n={measured['eligible_case_count']}", fontsize=8.5, color=MUTED, transform=ax.transAxes)
    ax.text(0.0, 0.50, "PROJECTED · PRIMARY", fontsize=8.5, color=MUTED, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.32, f"{primary['accuracy']:.1%}", fontsize=30, color=PRIMARY, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.24, f"typed R-GCN + PCST · macro-F1 {primary['macro_f1']:.3f} · n={primary['eligible_case_count']}", fontsize=8.5, color=MUTED, transform=ax.transAxes)
    ax.text(0.0, 0.09, f"forecast gain +{gain * 100:.1f} pts, on a 95% interval\nspanning {half * 200:.0f} points", fontsize=9, color=INK, transform=ax.transAxes, fontweight="bold")

    ax = fig.add_subplot(spec[1, 0])
    tidy(ax)
    width = 0.36
    positions = np.arange(len(LABELS))
    ax.bar(positions - width / 2, [measured["per_class"][label]["f1"] for label in LABELS], width=width, color=MEASURED, label="measured")
    ax.bar(positions + width / 2, [primary["per_class"][label]["f1"] for label in LABELS], width=width,
           color=(*matplotlib.colors.to_rgb(PRIMARY), 0.2), edgecolor=PRIMARY, linewidth=1.8, label="projected")
    ax.set_xticks(positions)
    ax.set_xticklabels([PRETTY[label].replace(" ", "\n") for label in LABELS], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")

    ax = fig.add_subplot(spec[1, 1])
    draw_confusion(ax, primary["confusion_matrix"], "Projected confusion, held-out", accent=PRIMARY)
    ax.title.set_fontsize(10.5)

    ax = fig.add_subplot(spec[1, 2])
    tidy(ax, grid_axis="both")
    counts = [point["labelled_cases"] for point in curve]
    values = [point["projected_accuracy"] for point in curve]
    anchor = next(point for point in curve if point["is_current_corpus"])
    ax.plot(counts, values, color=PROJECTED, linewidth=2)
    ax.scatter([anchor["labelled_cases"]], [anchor["projected_accuracy"]], s=95, color=PRIMARY, zorder=3, edgecolor="white", linewidth=1.5)
    ax.axhline(measured["accuracy"], color=MEASURED, linewidth=1.3)
    ax.set_xscale("log")
    ax.set_ylim(0.4, 0.9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xticks([50, 200, 1000, 5000])
    ax.set_xticklabels(["50", "200", "1k", "5k"], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("labelled cases", fontsize=8.5)
    ax.set_title("Data-scaling forecast", fontsize=10.5)

    ax = fig.add_subplot(spec[2, 0])
    tidy(ax)
    steps = projection["uplift_decomposition"]
    ax.bar(0, measured["accuracy"], color=MEASURED, width=0.6)
    for index, step in enumerate(steps, start=1):
        low = min(step["from_accuracy"], step["to_accuracy"])
        is_primary = step["variant_id"] == projection["primary_variant_id"]
        color = PRIMARY if is_primary else PROJECTED
        ax.bar(index, abs(step["delta"]), bottom=low, color=(*matplotlib.colors.to_rgb(color), 0.22), edgecolor=color, linewidth=1.6, width=0.6)
    ax.set_xticks(range(len(steps) + 1))
    ax.set_xticklabels(["base"] + [f"+{i}" for i in range(1, len(steps) + 1)], fontsize=8)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Stage attribution", fontsize=10.5)

    ax = fig.add_subplot(spec[2, 1])
    tidy(ax)
    ax.axvspan(noise["p05"], noise["p95"], color=PROJECTED, alpha=0.10)
    ax.bar([b["accuracy"] for b in noise["histogram"]], [b["share"] for b in noise["histogram"]], width=0.016, color=PROJECTED)
    ax.axvline(noise["assumed_accuracy"], color=PRIMARY, linewidth=2)
    ax.set_xlim(0.5, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"Single-run spread (n={noise['sample_size']})", fontsize=10.5)

    ax = fig.add_subplot(spec[2, 2])
    tidy(ax, grid_axis="both")
    sizes = [point["sample_size"] for point in power["points"]]
    probabilities = [point["probability_primary_ranks_higher"] for point in power["points"]]
    ax.plot(range(len(sizes)), probabilities, color=PROJECTED, linewidth=2)
    ax.scatter(range(len(sizes)), probabilities, s=38, color=PROJECTED, zorder=3, edgecolor="white", linewidth=1.1)
    ax.scatter([0], [probabilities[0]], s=110, color=PRIMARY, zorder=4, edgecolor="white", linewidth=1.5)
    ax.axhline(0.95, color=MEASURED, linewidth=1.4)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(size) for size in sizes], fontsize=8)
    ax.set_ylim(0.5, 1.04)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("held-out cases", fontsize=8.5)
    ax.set_title("Power to rank variants", fontsize=10.5)

    fig.suptitle("SC-2016 outcome forecast — complete summary", fontsize=16, fontweight="bold", y=0.965)
    fig.text(0.5, 0.935, "Orange is measured. Blue and green are forecasts from the projection config, produced before any model has been trained.",
             ha="center", fontsize=10, color=MUTED)
    footer(fig, "No held-out label has been read by a fitted model. Regenerate with: py -3 -m legal_graph.cli project-outcomes && py -3 scripts/render_figures.py", y=0.015)
    save(fig, "projection", "fig16_projection_summary.png")


# ================================================================ diagram


def fig17_pipeline_diagram(data):
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 63)
    ax.axis("off")

    counts = data["manifest"]["split_counts"]
    provisions = len(data["provisions"])
    graph = data["judgment_graph"]
    projection = data["projection"]
    primary = variant(projection, projection["primary_variant_id"])["splits"]["test"]

    stages = [
        (3, 44, 20, 9, "1 · SOURCES", f"3 statute PDFs (BNS, BNSS, CrPC)\n{sum(counts.values())} SC-2016 judgments\nchecksum-pinned, local only", PROJECTED, "done"),
        (28, 44, 20, 9, "2 · EXTRACTION", f"{provisions:,} provisions\nmonotonic section sequence\nnon-destructive audit report", PROJECTED, "done"),
        (53, 44, 20, 9, "3 · GRAPH", f"{len(graph['nodes']):,} nodes · {len(graph['edges']):,} edges\ncase / court / act / provision\nschema-validated snapshot", PROJECTED, "done"),
        (78, 44, 19, 9, "4 · INDEX", "BM25 over provisions\n6,136-term vocabulary\npolicy-hash manifest", PROJECTED, "done"),
        (3, 28, 20, 9, "5 · SPLIT", f"train {counts['train']} · val {counts['validation']}\ntest {counts['test']} held out\nsha256-prefix modulo 10", PROJECTED, "done"),
        (28, 28, 20, 9, "6 · BASELINE", f"train-only multinomial NB\nmeasured {projection['measured_baseline']['accuracy']:.1%} accuracy\nmacro-F1 {projection['measured_baseline']['macro_f1']:.3f}", MEASURED, "measured"),
        (53, 28, 20, 9, "7 · RETRIEVAL", "PCST subgraph, N=80, k=2\nsplit-safe + temporal filter\ndeterministic tie-breaking", BOUND, "planned"),
        (78, 28, 19, 9, "8 · GRAPH ENCODER", "2-layer typed R-GCN\nsoft-token prefix\nrelation masks for ablation", BOUND, "planned"),
        (53, 12, 20, 9, "9 · QLORA", "Qwen2.5-7B / SaulLM-7B\n4-bit base, rank-16 LoRA\n≤2% trainable cap", BOUND, "planned"),
        (78, 12, 19, 9, "10 · EVALUATION", f"forecast {primary['accuracy']:.1%} accuracy\nmacro-F1 {primary['macro_f1']:.3f}\nvs stage 6, 95% CI ±{projection['headline']['accuracy_interval']['half_width'] * 100:.0f} pts", PRIMARY, "projected"),
    ]

    for x, y, w, h, title, body, color, state in stages:
        dashed = state == "planned"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.7",
                                    linewidth=1.8, edgecolor=color, linestyle="--" if dashed else "-",
                                    facecolor=(*matplotlib.colors.to_rgb(color), 0.09 if dashed else 0.14)))
        ax.text(x + 0.9, y + h - 1.9, title, fontsize=9, fontweight="bold", color=color)
        ax.text(x + 0.9, y + h - 3.6, body, fontsize=8, color=INK, va="top", linespacing=1.45)
        ax.text(x + w - 0.9, y + h - 1.9, state, fontsize=7.5, color=color, ha="right", fontweight="bold")

    def arrow(x1, y1, x2, y2, *, style="arc3,rad=0", color="#98a1ab"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.4, color=color, connectionstyle=style, shrinkA=2, shrinkB=2))

    arrow(23.5, 48.5, 27.5, 48.5)
    arrow(48.5, 48.5, 52.5, 48.5)
    arrow(73.5, 48.5, 77.5, 48.5)
    arrow(13, 43.5, 13, 37.5)
    arrow(23.5, 32.5, 27.5, 32.5)
    arrow(58, 43.5, 58, 37.5)
    arrow(87, 43.5, 69, 37.5, style="arc3,rad=0.18")
    arrow(73.5, 32.5, 77.5, 32.5)
    arrow(82, 27.5, 68, 21.5, style="arc3,rad=0.16")
    arrow(13, 27.5, 52.5, 16.5, style="angle,angleA=-90,angleB=180,rad=3")
    arrow(73.5, 16.5, 77.5, 16.5)

    ax.text(3, 6.4, "Solid boxes are implemented and produce checked local artifacts. Dashed boxes are planned; no model weights have been downloaded and no training run exists.",
            fontsize=9, color=MUTED)
    ax.text(3, 3.6, "Stage 6 is the only measured accuracy on this diagram. Stage 10 is a forecast produced by the projection config, not a result.",
            fontsize=9, color=MUTED)
    ax.text(3, 59.4, "Graph-augmented legal LLM pipeline", fontsize=17, fontweight="bold", color=INK)
    ax.text(3, 56.4, "Local-first: every stage reads and writes checksum-pinned artifacts on this machine.", fontsize=10, color=MUTED)

    save(fig, "diagrams", "fig17_pipeline_diagram.png")


# ================================================================ main


def main():
    print("loading local artifacts...")
    data = load_everything()
    print(f"  {len(data['judgments'])} judgments, {len(data['provisions'])} provisions, "
          f"{len(data['judgment_graph']['nodes'])} graph nodes")
    print("\ncorpus figures")
    fig1_corpus_overview(data)
    fig2_label_imbalance(data)
    fig3_graph_structure(data)
    fig4_topics_judges(data)
    fig5_length_and_time(data)
    fig6_citation_resolution(data)
    fig7_statute_corpus(data)
    print("\nprojection figures")
    fig8_variant_ladder(data)
    fig9_accuracy_vs_macro_f1(data)
    fig10_uplift_waterfall(data)
    fig11_learning_curve(data)
    fig12_per_class_f1(data)
    fig13_confusion_matrices(data)
    fig14_sampling_noise(data)
    fig15_statistical_power(data)
    fig16_projection_summary(data)
    print("\ndiagram")
    fig17_pipeline_diagram(data)
    print(f"\n17 figures written under {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
