"""Transparent quality audits for locally ingested statute PDFs."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pypdf import PdfReader

from .statutes import (
    Provision,
    StatuteIngestion,
    _crop_first_body_page,
    _find_body_start,
    extract_provision_candidates_from_pages,
)

AUDIT_SCHEMA_VERSION = "1.0.0"
_STRUCTURAL_HEADING = re.compile(
    r"(?im)^\s*(?:the\s+)?(?:(?:first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth)\s+)?(?:schedule|appendix|state\s+amendments?)\b"
)


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """A reviewable extraction anomaly anchored to a source PDF range."""

    severity: str
    category: str
    act_id: str
    section: str | None
    source_page_start: int | None
    source_page_end: int | None
    message: str


def read_provision_candidates(path: Path) -> tuple[Provision, ...]:
    """Re-read a statute body and preserve every candidate heading for audit."""

    reader = PdfReader(str(path))
    pages = tuple(reader_page.extract_text() or "" for reader_page in reader.pages)
    body_start_index = _find_body_start(pages)
    body_pages = tuple(
        (page_index + 1, _crop_first_body_page(text, page_index == body_start_index))
        for page_index, text in enumerate(pages[body_start_index:], start=body_start_index)
    )
    return extract_provision_candidates_from_pages(body_pages)


def audit_provision_candidates(
    *, act_id: str, candidates: Sequence[Provision]
) -> tuple[AuditFinding, ...]:
    """Find duplicate headings, number resets, and structural boundaries.

    Findings are conservative: they request review instead of treating a later
    heading as an amendment, schedule, or parser error.
    """

    findings: list[AuditFinding] = []
    by_section: dict[str, list[Provision]] = defaultdict(list)
    previous_number: int | None = None

    for provision in candidates:
        by_section[provision.section].append(provision)
        number = _section_number(provision.section)
        if number is not None and previous_number is not None and number < previous_number:
            findings.append(
                AuditFinding(
                    severity="review",
                    category="section_number_reset",
                    act_id=act_id,
                    section=provision.section,
                    source_page_start=provision.source_page_start,
                    source_page_end=provision.source_page_end,
                    message=(
                        f"Section sequence drops from {previous_number} to {provision.section}; "
                        "inspect this page for a schedule, appendix, or inserted amendment."
                    ),
                )
            )
        if number is not None:
            previous_number = number

        heading_window = re.sub(r"^\s*\d+[A-Z]?\.\s*", "", provision.text[:350])
        if _STRUCTURAL_HEADING.search(heading_window):
            findings.append(
                AuditFinding(
                    severity="review",
                    category="structural_heading",
                    act_id=act_id,
                    section=provision.section,
                    source_page_start=provision.source_page_start,
                    source_page_end=provision.source_page_end,
                    message="A schedule, appendix, or state-amendment heading appears at the start of this candidate.",
                )
            )

    for section, occurrences in sorted(by_section.items(), key=lambda item: _section_key(item[0])):
        if len(occurrences) > 1:
            page_ranges = ", ".join(
                f"{item.source_page_start}-{item.source_page_end}" for item in occurrences
            )
            findings.append(
                AuditFinding(
                    severity="review",
                    category="duplicate_section",
                    act_id=act_id,
                    section=section,
                    source_page_start=occurrences[-1].source_page_start,
                    source_page_end=occurrences[-1].source_page_end,
                    message=(
                        f"Section {section} occurs {len(occurrences)} times on PDF pages {page_ranges}. "
                        "The searchable corpus currently retains the earliest occurrence."
                    ),
                )
            )
    return tuple(sorted(findings, key=_finding_key))


def audit_statute_ingestions(ingestions: Iterable[StatuteIngestion]) -> dict[str, object]:
    """Create a deterministic, source-provenanced audit report without changing data."""

    acts: list[dict[str, object]] = []
    all_findings: list[AuditFinding] = []
    for ingestion in sorted(ingestions, key=lambda item: item.spec.act_id):
        candidates = read_provision_candidates(ingestion.source_path)
        findings = audit_provision_candidates(act_id=ingestion.spec.act_id, candidates=candidates)
        all_findings.extend(findings)
        acts.append(
            {
                "act_id": ingestion.spec.act_id,
                "title": ingestion.spec.title,
                "source_pdf": ingestion.source_path.as_posix(),
                "source_sha256": ingestion.source_sha256,
                "page_count": ingestion.page_count,
                "body_start_page": ingestion.body_start_page,
                "candidate_heading_count": len(candidates),
                "searchable_provision_count": len(ingestion.provisions),
                "duplicate_sections": list(ingestion.duplicate_sections),
            }
        )
    sorted_findings = tuple(sorted(all_findings, key=_finding_key))
    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_policy": "statute-extraction-audit-v1",
        "status": "needs_review" if sorted_findings else "passed_no_findings",
        "acts": acts,
        "finding_count": len(sorted_findings),
        "findings": [asdict(finding) for finding in sorted_findings],
    }
    payload["content_sha256"] = _payload_sha256(payload)
    return payload


def write_audit_report(payload: dict[str, object], output_path: Path) -> tuple[Path, str]:
    """Persist a canonical local audit report using an atomic replacement."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(canonical, encoding="utf-8", newline="\n")
    temporary.replace(output_path)
    return output_path, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _section_number(section: str) -> int | None:
    match = re.fullmatch(r"(\d+)[A-Z]?", section)
    return int(match.group(1)) if match else None


def _section_key(section: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)([A-Z]?)", section)
    return (int(match.group(1)), match.group(2)) if match else (10**9, section)


def _finding_key(finding: AuditFinding) -> tuple[str, int, str, str, int]:
    return (
        finding.act_id,
        finding.source_page_start or 0,
        finding.section or "",
        finding.category,
        finding.source_page_end or 0,
    )


def _payload_sha256(payload: dict[str, object]) -> str:
    without_digest = {key: value for key, value in payload.items() if key != "content_sha256"}
    text = json.dumps(without_digest, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
