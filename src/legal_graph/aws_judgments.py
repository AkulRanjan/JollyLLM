"""Indian Supreme Court judgments from the public AWS Open Data bucket (CC-BY-4.0, no account needed).

Two steps:

- ``extract_aws_years`` runs where downloads are fast (a Kaggle CPU session). For each year it downloads the
  metadata and English PDF archives, reads each PDF's embedded text layer, and writes one sorted, gzipped JSONL
  record per judgment. Every archive is hashed, so the output names exactly what it was made from.
- ``select_aws_records`` turns that file into judgment records beside SC-2016. Decision years before 2016 become
  training cases and later years a separate test set; nothing decided in 2016 is taken, and an SCR report that
  is also in SC-2016 is skipped. The label is the official "Disposal Nature", and a case is kept only where the
  project's own disposition rule reads the same label from the judgment text, so every case is labelled the way
  SC-2016 is.
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import tarfile
import urllib.request
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from .judgments import SC2016_COURT_ID, SC2016_COURT_LABEL, JudgmentRecord
from .models import Split
from .outcomes import infer_disposition

BUCKET_URL = "https://indian-supreme-court-judgments.s3.amazonaws.com"
AWS_DATASET = "aws:indian-supreme-court-judgments"
DISPOSAL_LABELS = {
    "appeal(s) allowed": "allowed",
    "case allowed": "allowed",
    "dismissed": "dismissed",
    "disposed off": "disposed",
    "case partly allowed": "partly_allowed",
}
_FIELD = re.compile(r"(Decision Date|Case No|Disposal Nature|Bench)\s*:\s*</span>\s*<font[^>]*>(.*?)</font>", re.S)
_TITLE = re.compile(r'aria-label="\s*(.*?)\s+pdf"', re.S)
_SCR = re.compile(r"<span class='escrText'>(.*?)</span>", re.S)
_NEUTRAL = re.compile(r"<span class='ncDisplay'>(.*?)</span>", re.S)

Opener = Callable[[str], object]  # url -> binary file-like (urllib.request.urlopen, or a local stand-in in tests)


def parse_metadata(record: Mapping[str, object]) -> dict[str, str]:
    """Title, citations and case details from one bucket metadata record (its scraped raw_html)."""
    raw = str(record.get("raw_html", ""))
    fields = {key: _clean(value) for key, value in _FIELD.findall(raw)}
    title = _TITLE.search(raw)
    scr, neutral = _SCR.search(raw), _NEUTRAL.search(raw)
    decided = fields.get("Decision Date", "")
    try:
        decision_date = datetime.strptime(decided, "%d-%m-%Y").date().isoformat()
    except ValueError:
        decision_date = ""
    return {
        "path": str(record.get("path", "")),
        "title": _clean(title.group(1)).replace(" versus ", " v. ") if title else str(record.get("path", "")),
        "decision_date": decision_date,
        "case_number": fields.get("Case No", ""),
        "disposal_nature": fields.get("Disposal Nature", ""),
        "bench": fields.get("Bench", ""),
        "scr_citation": _clean(scr.group(1)) if scr else "",
        "neutral_citation": _clean(neutral.group(1)) if neutral else "",
    }


def official_label(disposal_nature: str) -> str | None:
    """The four-way label for an official disposal value, or None for anything else (adjourned, referred, ...)."""
    return DISPOSAL_LABELS.get(" ".join(disposal_nature.split()).lower())


_MARGIN_LETTER = re.compile(r"^[\s.,:;'`\"]*[A-H][\s.,:;'`\"]*$")
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
_REPORTS_HEADER = re.compile(r"SUPREME\s+COURT\s+REPORTS", re.IGNORECASE)
_HEADING = re.compile(r"[A-Z][A-Z .&()/:-]{2,40}")
_SENTENCE_END = re.compile(r"""[.:;?!"')\]]$""")
_HEADER_LINES = 4  # running headers sit in the first lines of a page
_HEADER_PAGES = 3  # ... and repeat on at least this many pages


def clean_pdf_text(pages: Iterable[str]) -> str:
    """SCR page text without margin letters, page numbers or running headers, with wrapped lines rejoined.

    The SC-2016 transcripts hold one paragraph per line; this brings the PDF text layer to the same shape.
    """
    pages = [[" ".join(line.split()) for line in page.replace("\r", "\n").split("\n")] for page in pages]
    pages = [[line for line in page if line] for page in pages]
    repeats = Counter(key for page in pages for key in {_header_key(line) for line in page[:_HEADER_LINES]} if key)
    lines: list[str] = []
    for page in pages:
        for position, line in enumerate(page):
            if _MARGIN_LETTER.match(line) or _PAGE_NUMBER.match(line):
                continue
            if _REPORTS_HEADER.search(line) and len(line) <= 80:
                continue
            if position < _HEADER_LINES and repeats[_header_key(line)] >= _HEADER_PAGES:
                continue
            lines.append(line)
    paragraphs: list[str] = []
    for line in lines:
        if not paragraphs or not _continues(paragraphs[-1], line):
            paragraphs.append(line)
        elif paragraphs[-1].endswith("-") and paragraphs[-1][-2:-1].isalpha() and line[:1].islower():
            paragraphs[-1] = paragraphs[-1][:-1] + line
        else:
            paragraphs[-1] = f"{paragraphs[-1]} {line}"
    return "\n".join(paragraphs)


def _header_key(line: str) -> str:
    key = re.sub(r"[\d\W_]+", "", line.lower())
    return key if len(key) >= 6 else ""


def _continues(previous: str, line: str) -> bool:
    if _HEADING.fullmatch(previous) or _HEADING.fullmatch(line):
        return False
    return not (_SENTENCE_END.search(previous) and (line[:1].isupper() or line[:1].isdigit() or line[:1] == "("))


def pdf_pages(data: bytes) -> list[str]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as document:
        return [page.get_text() for page in document]


def extract_aws_years(
    years: Iterable[int],
    work_dir: Path,
    output_path: Path,
    *,
    base_url: str = BUCKET_URL,
    opener: Opener | None = None,
) -> dict[str, object]:
    """Download each year's archives, extract every English judgment, and write the gzipped JSONL records."""
    open_url = opener or (lambda url: urllib.request.urlopen(url, timeout=120))
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    archives: dict[str, dict[str, object]] = {}
    counts: dict[str, dict[str, int]] = {}
    for year in sorted(set(years)):
        tally = Counter()
        metadata: dict[str, bytes] = {}
        for name, data in _archive_members(year, "metadata", work_dir, base_url, open_url, archives):
            if name.endswith(".json"):
                metadata[Path(name).stem] = data
        tally["metadata_records"] = len(metadata)
        for name, data in _archive_members(year, "english", work_dir, base_url, open_url, archives):
            if not name.endswith("_EN.pdf"):
                continue
            tally["pdfs"] += 1
            stem = Path(name).name[: -len("_EN.pdf")]
            if stem not in metadata:
                tally["pdf_without_metadata"] += 1
                continue
            try:
                pages = pdf_pages(data)
            except Exception as error:  # a damaged PDF is counted and skipped, never guessed at
                tally["unreadable_pdf"] += 1
                print(f"unreadable {name}: {error}", flush=True)
                continue
            fields = parse_metadata(json.loads(metadata[stem].decode("utf-8")))
            rows.append({
                **fields,
                "path": stem,
                "partition_year": year,
                "metadata_sha256": hashlib.sha256(metadata[stem]).hexdigest(),
                "pdf_sha256": hashlib.sha256(data).hexdigest(),
                "page_count": len(pages),
                "text": clean_pdf_text(pages),
            })
            tally["extracted"] += 1
        counts[str(year)] = dict(sorted(tally.items()))
        print(f"{year}: {counts[str(year)]}", flush=True)
    rows.sort(key=lambda row: (str(row["path"]), int(row["partition_year"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    temporary.replace(output_path)
    return {
        "schema_version": "1.0.0",
        "records_path": output_path.name,
        "records_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "record_count": len(rows),
        "counts_by_year": counts,
        "archives": dict(sorted(archives.items())),
        "text_layer": "pymupdf get_text, clean_pdf_text",
    }


def _archive_members(year: int, kind: str, work_dir: Path, base_url: str, open_url: Opener,
                     archives: dict[str, dict[str, object]]) -> Iterator[tuple[str, bytes]]:
    folder = f"data/tar/year={year}/english" if kind == "english" else f"metadata/tar/year={year}"
    index_name = "english.index.json" if kind == "english" else "metadata.index.json"
    with open_url(f"{base_url}/{folder}/{index_name}") as response:
        index = json.loads(response.read().decode("utf-8"))
    for part in index["parts"]:
        relative = f"{folder}/{part['name']}"
        local = work_dir / relative.replace("/", "__")
        digest = hashlib.sha256()
        with open_url(f"{base_url}/{relative}") as response, local.open("wb") as handle:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                handle.write(chunk)
        archives[relative] = {"sha256": digest.hexdigest(), "bytes": local.stat().st_size}
        try:
            with tarfile.open(local) as archive:
                for member in archive:
                    if member.isfile():
                        yield Path(member.name).name, archive.extractfile(member).read()
        finally:
            local.unlink()


def read_aws_records(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_aws_records(
    records_path: Path,
    *,
    source_id: str,
    expected_sha256: str | None,
    train_years: Iterable[int],
    test_years: Iterable[int],
    exclude_paths: Iterable[str],
) -> tuple[tuple[JudgmentRecord, ...], dict[str, object]]:
    """Judgment records for the chosen decision years, with the reason every other record was left out."""
    source_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    if expected_sha256 and source_sha256 != expected_sha256:
        raise ValueError(f"AWS records checksum mismatch: expected {expected_sha256}, got {source_sha256}")
    train, test = frozenset(train_years), frozenset(test_years)
    if train & test or any(year == 2016 for year in train | test):
        raise ValueError("train and test years must be disjoint and must not include 2016 (the SC-2016 year)")
    excluded = {path.lower() for path in exclude_paths}
    kept: list[JudgmentRecord] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    rows = read_aws_records(records_path)
    for row in rows:
        reason = _skip_reason(row, seen, excluded, train, test)
        seen.add(str(row["path"]).lower())
        if reason:
            skipped[reason] += 1
            continue
        kept.append(_record(row, Split.TRAIN if int(row["decision_date"][:4]) in train else Split.TEST,
                            source_id=source_id, source_sha256=source_sha256))
    kept.sort(key=lambda record: record.case_id)
    if len({record.case_id for record in kept}) != len(kept):
        raise ValueError("duplicate AWS case IDs")
    report = {
        "source_id": source_id,
        "source_sha256": source_sha256,
        "records": len(rows),
        "train_years": sorted(train),
        "test_years": sorted(test),
        "kept": dict(sorted(Counter(record.origin_split.value for record in kept).items())),
        "kept_labels": dict(sorted(Counter(f"{record.origin_split.value}:{record.attributes['official_label']}"
                                           for record in kept).items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return tuple(kept), report


def _skip_reason(row: Mapping[str, object], seen: set[str], excluded: set[str], train: frozenset[int],
                 test: frozenset[int]) -> str | None:
    path = str(row["path"]).lower()
    if path in seen:
        return "duplicate_report"
    if path in excluded:
        return "in_sc2016"
    decided = str(row.get("decision_date") or "")
    if not decided:
        return "no_decision_date"
    if int(decided[:4]) not in train | test:
        return "other_year"
    text = str(row.get("text") or "")
    if not text.strip():
        return "no_text"
    label = official_label(str(row.get("disposal_nature", "")))
    if label is None:
        return "no_four_way_disposal"
    inferred = infer_disposition(text[int(len(text) * 0.8) :])
    if inferred is None:
        return "no_disposition_phrase_in_text"
    if inferred != label:
        return "text_disagrees_with_official_label"
    return None


def _record(row: Mapping[str, object], split: Split, *, source_id: str, source_sha256: str) -> JudgmentRecord:
    raw_sha256 = hashlib.sha256(f"{row['metadata_sha256']}|{row['pdf_sha256']}".encode("utf-8")).hexdigest()
    decided = date.fromisoformat(str(row["decision_date"]))
    partition = int(row["partition_year"])
    pdf_path = Path(f"data/tar/year={partition}/english/{row['path']}_EN.pdf")
    attributes = {
        "aws_path": str(row["path"]),
        "bench": str(row.get("bench", "")),
        "case_number": str(row.get("case_number", "")),
        "dataset": AWS_DATASET,
        "disposal_nature": str(row["disposal_nature"]),
        "metadata_sha256": str(row["metadata_sha256"]),
        "neutral_citation": str(row.get("neutral_citation", "")),
        "official_label": str(official_label(str(row["disposal_nature"]))),
        "origin_split": split.value,
        "partition_year": str(partition),
        "pdf_sha256": str(row["pdf_sha256"]),
        "raw_sha256": raw_sha256,
        "scr_citation": str(row.get("scr_citation", "")),
        "source_id": source_id,
        "source_sha256": source_sha256,
        "year": str(decided.year),
    }
    return JudgmentRecord(
        document_id="doc:india:sc:judgment:sha256-" + raw_sha256[:16],
        case_id=f"case:india:sc:{decided.year}:sha256-" + raw_sha256[:16],
        source_id=source_id,
        source_sha256=source_sha256,
        raw_sha256=raw_sha256,
        markdown_path=pdf_path,
        metadata_path=Path(f"metadata/tar/year={partition}/{row['path']}.json"),
        title=str(row["title"]),
        court_id=SC2016_COURT_ID,
        court_label=SC2016_COURT_LABEL,
        decision_date=decided,
        origin_split=split,
        text=str(row["text"]),
        attributes=attributes,
    )


def _clean(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())
