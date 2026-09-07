"""Rule-based Indian legal reference extraction for the initial graph-construction path.

The extractor does not assert that a reference resolves. It preserves the original
span and a canonical resolution hint so unresolved and ambiguous references remain
measurable quality signals instead of being silently discarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Iterable, Mapping

from .models import RelationType


class ReferenceKind(StrEnum):
    CASE = "case"
    PROVISION = "provision"


class ExtractionMethod(StrEnum):
    INDIAN_REGEX_V1 = "indian_regex_v1"


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    extraction_id: str
    document_id: str
    kind: ReferenceKind
    relation_type: RelationType
    span_start: int
    span_end: int
    raw_text: str
    canonical_hint: str
    method: ExtractionMethod = ExtractionMethod.INDIAN_REGEX_V1
    confidence: float = 0.95


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    candidate: ReferenceCandidate
    resolved_target_id: str | None
    unresolved_reason: str | None


_PROVISION_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.)\s*"
    r"(?P<section>\d+[A-Za-z]?(?:\s*\(\s*\d+[A-Za-z]?\s*\))?)"
    r"\s*(?:of\s+(?:the\s+)?)?"
    r"(?P<act>Indian\s+Penal\s+Code|IPC|Bharatiya\s+Nyaya\s+Sanhita|BNS|"
    r"Code\s+of\s+Criminal\s+Procedure|CrPC)\b",
    re.IGNORECASE,
)
_SCC_PATTERN = re.compile(
    r"\(\s*(?P<year>\d{4})\s*\)\s*(?P<volume>\d+)\s*SCC\s*(?P<page>\d+)\b",
    re.IGNORECASE,
)
_SCC_ONLINE_PATTERN = re.compile(
    r"\b(?P<year>\d{4})\s+SCC\s+OnLine\s+SC\s+(?P<number>\d+)\b",
    re.IGNORECASE,
)

_ACT_IDENTIFIERS = {
    "indian penal code": "ipc",
    "ipc": "ipc",
    "bharatiya nyaya sanhita": "bns",
    "bns": "bns",
    "code of criminal procedure": "crpc",
    "crpc": "crpc",
}


def extract_indian_references(text: str, document_id: str) -> tuple[ReferenceCandidate, ...]:
    """Extract Indian statute and reporter references in deterministic text order."""

    matches: list[ReferenceCandidate] = []
    for match in _PROVISION_PATTERN.finditer(text):
        section = re.sub(r"\s+", "", match.group("section")).lower()
        act_id = _ACT_IDENTIFIERS[_normalise_spaces(match.group("act")).lower()]
        matches.append(
            _candidate(
                document_id=document_id,
                kind=ReferenceKind.PROVISION,
                relation_type=RelationType.REFERS_TO_PROVISION,
                span_start=match.start(),
                span_end=match.end(),
                raw_text=match.group(0),
                canonical_hint=f"provision:india:{act_id}:{section}:",
            )
        )

    for match in _SCC_PATTERN.finditer(text):
        matches.append(
            _candidate(
                document_id=document_id,
                kind=ReferenceKind.CASE,
                relation_type=RelationType.CITES,
                span_start=match.start(),
                span_end=match.end(),
                raw_text=match.group(0),
                canonical_hint=(
                    "case:india:reporter:scc:"
                    f"{match.group('year')}:{match.group('volume')}:{match.group('page')}"
                ),
            )
        )

    for match in _SCC_ONLINE_PATTERN.finditer(text):
        matches.append(
            _candidate(
                document_id=document_id,
                kind=ReferenceKind.CASE,
                relation_type=RelationType.CITES,
                span_start=match.start(),
                span_end=match.end(),
                raw_text=match.group(0),
                canonical_hint=f"case:india:reporter:scc-online-sc:{match.group('year')}:{match.group('number')}",
            )
        )

    return tuple(sorted(matches, key=lambda candidate: (candidate.span_start, candidate.span_end, candidate.kind.value)))


def resolve_candidates(
    candidates: Iterable[ReferenceCandidate],
    known_target_ids: Iterable[str],
) -> tuple[ResolutionResult, ...]:
    """Resolve exact case IDs and uniquely versioned provision hints against local IDs."""

    target_ids = tuple(sorted(set(known_target_ids)))
    results: list[ResolutionResult] = []
    for candidate in candidates:
        if candidate.kind is ReferenceKind.CASE:
            matches = [target_id for target_id in target_ids if target_id == candidate.canonical_hint]
        else:
            matches = [target_id for target_id in target_ids if target_id.startswith(candidate.canonical_hint)]

        if len(matches) == 1:
            results.append(ResolutionResult(candidate, matches[0], None))
        elif not matches:
            results.append(ResolutionResult(candidate, None, "no_canonical_target"))
        else:
            results.append(ResolutionResult(candidate, None, "ambiguous_canonical_target"))
    return tuple(results)


def resolution_summary(results: Iterable[ResolutionResult]) -> Mapping[str, int]:
    """Return stable counts suitable for an extraction-quality report."""

    counts = {"resolved": 0, "unresolved": 0, "ambiguous": 0}
    for result in results:
        if result.resolved_target_id:
            counts["resolved"] += 1
        elif result.unresolved_reason == "ambiguous_canonical_target":
            counts["ambiguous"] += 1
        else:
            counts["unresolved"] += 1
    return counts


def _candidate(
    *,
    document_id: str,
    kind: ReferenceKind,
    relation_type: RelationType,
    span_start: int,
    span_end: int,
    raw_text: str,
    canonical_hint: str,
) -> ReferenceCandidate:
    stable_fields = "|".join((document_id, kind.value, str(span_start), str(span_end), canonical_hint))
    return ReferenceCandidate(
        extraction_id="extract:" + sha256(stable_fields.encode("utf-8")).hexdigest()[:16],
        document_id=document_id,
        kind=kind,
        relation_type=relation_type,
        span_start=span_start,
        span_end=span_end,
        raw_text=raw_text,
        canonical_hint=canonical_hint,
    )


def _normalise_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
