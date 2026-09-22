"""Deterministic local ingestion of the SC-2016 judgment corpus.

The source dataset provides a JSON metadata record and a corresponding Markdown
transcript for every judgment. This module treats the transcript as the searchable
primary text, retains the source-content checksums, and never converts an
unresolved citation into a graph edge.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .builder import build_reference_edges
from .extraction import ResolutionResult, extract_indian_references, resolution_summary, resolve_candidates
from .manifests import SourceRecord, sha256_payload, validate_source_records
from .models import EdgeRecord, GraphSnapshot, NodeRecord, NodeType, RelationType, Split
from .validation import validate_graph


SC2016_SOURCE_ID = "source:huggingface:sc-judgments-2016:e928c72019d62098"
SC2016_COURT_ID = "court:india:supreme-court"
SC2016_COURT_LABEL = "Supreme Court of India"
_MONTH_DATE_PATTERN = re.compile(
    r"\b(?P<month>JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
    r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_DATE_MONTH_PATTERN = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|"
    r"SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER),?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class JudgmentRecord:
    """A content-addressed judgment ready for corpus and graph materialisation."""

    document_id: str
    case_id: str
    source_id: str
    source_sha256: str
    raw_sha256: str
    markdown_path: Path
    metadata_path: Path
    title: str
    court_id: str
    court_label: str
    decision_date: date | None
    origin_split: Split
    text: str
    attributes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class JudgmentIngestion:
    """All locally parsed records for one content-verified SC-2016 source."""

    source_id: str
    source_root: Path
    source_sha256: str
    judgments: tuple[JudgmentRecord, ...]


@dataclass(frozen=True, slots=True)
class JudgmentGraphAssembly:
    """Graph result plus retained unresolved evidence for quality review."""

    graph: GraphSnapshot
    unresolved: tuple[ResolutionResult, ...]
    resolution_counts: Mapping[str, int]


def load_source_record(registry_path: Path, source_id: str = SC2016_SOURCE_ID) -> SourceRecord:
    """Load one approved local source record from the versioned registry JSON."""

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError(f"source registry {registry_path} must contain a sources list")
    records = tuple(_source_record(item) for item in raw_sources)
    validate_source_records(records)
    matches = [record for record in records if record.source_id == source_id]
    if len(matches) != 1:
        raise ValueError(f"source registry {registry_path} does not contain exactly one record for {source_id!r}")
    record = matches[0]
    if record.status != "approved":
        raise ValueError(f"source {source_id!r} is {record.status!r}, not approved for ingestion")
    return record


def ingest_sc2016_directory(
    source_root: Path,
    *,
    source_id: str = SC2016_SOURCE_ID,
    expected_source_sha256: str | None = None,
) -> JudgmentIngestion:
    """Load SC-2016 JSON/Markdown pairs and verify their deterministic source hash."""

    metadata_root = source_root / "extracted_jsons"
    markdown_root = source_root / "extracted_mds"
    metadata_paths = tuple(sorted(metadata_root.glob("*.json")))
    if not metadata_paths:
        raise ValueError(f"no SC-2016 metadata JSON files found under {metadata_root}")
    if not markdown_root.is_dir():
        raise FileNotFoundError(markdown_root)

    loaded: list[tuple[Path, Path, Mapping[str, Any], bytes, bytes]] = []
    source_digest = hashlib.sha256()
    for metadata_path in metadata_paths:
        metadata_bytes = metadata_path.read_bytes()
        payload = json.loads(metadata_bytes.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"judgment metadata {metadata_path} is not an object")
        markdown_path = _markdown_path(payload, markdown_root, metadata_path)
        markdown_bytes = markdown_path.read_bytes()
        source_digest.update(metadata_path.name.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(hashlib.sha256(metadata_bytes).digest())
        source_digest.update(b"\0")
        source_digest.update(markdown_path.name.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(hashlib.sha256(markdown_bytes).digest())
        source_digest.update(b"\n")
        loaded.append((metadata_path, markdown_path, payload, metadata_bytes, markdown_bytes))

    source_sha256 = source_digest.hexdigest()
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            "SC-2016 source checksum mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )

    judgments = tuple(
        _judgment_record(
            metadata_path,
            markdown_path,
            payload,
            metadata_bytes,
            markdown_bytes,
            source_id=source_id,
            source_sha256=source_sha256,
        )
        for metadata_path, markdown_path, payload, metadata_bytes, markdown_bytes in loaded
    )
    _validate_unique_case_ids(judgments)
    return JudgmentIngestion(
        source_id=source_id,
        source_root=source_root,
        source_sha256=source_sha256,
        judgments=tuple(sorted(judgments, key=lambda record: record.case_id)),
    )


def write_judgment_corpus(
    ingestion: JudgmentIngestion,
    output_path: Path,
) -> tuple[Path, str, int]:
    """Write a deterministic local JSONL corpus with full searchable judgment text."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in ingestion.judgments:
            payload = {
                "attributes": dict(sorted(record.attributes.items())),
                "case_id": record.case_id,
                "court_id": record.court_id,
                "decision_date": record.decision_date.isoformat() if record.decision_date else None,
                "document_id": record.document_id,
                "markdown_path": record.markdown_path.as_posix(),
                "origin_split": record.origin_split.value,
                "raw_sha256": record.raw_sha256,
                "source_id": record.source_id,
                "source_sha256": record.source_sha256,
                "text": record.text,
                "title": record.title,
            }
            line = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    temporary_path.replace(output_path)
    return output_path, digest.hexdigest(), len(ingestion.judgments)


def build_judgment_graph(
    base_graph: GraphSnapshot,
    ingestion: JudgmentIngestion,
    *,
    version: str = "india-statutes-sc2016-0.1.0",
) -> JudgmentGraphAssembly:
    """Extend a validated statute graph with cases, court edges, and resolved provisions."""

    existing_nodes = base_graph.nodes_by_id
    added_nodes: list[NodeRecord] = []
    added_edges: list[EdgeRecord] = []
    unresolved: list[ResolutionResult] = []
    resolution_counts = {"resolved": 0, "unresolved": 0, "ambiguous": 0}
    known_target_ids = tuple(existing_nodes)

    if SC2016_COURT_ID not in existing_nodes:
        added_nodes.append(
            NodeRecord(
                node_id=SC2016_COURT_ID,
                node_type=NodeType.COURT,
                label=SC2016_COURT_LABEL,
                source_document_ids=(ingestion.source_id,),
                attributes={"source_id": ingestion.source_id, "source_sha256": ingestion.source_sha256},
            )
        )

    for record in ingestion.judgments:
        if record.case_id in existing_nodes:
            raise ValueError(f"base graph already contains judgment case_id {record.case_id!r}")
        added_nodes.append(
            NodeRecord(
                node_id=record.case_id,
                node_type=NodeType.CASE,
                label=record.title,
                source_document_ids=(record.document_id,),
                decision_date=record.decision_date,
                attributes=dict(sorted(record.attributes.items())),
            )
        )
        court_evidence = _evidence_id("decided-by", record.raw_sha256)
        added_edges.append(
            EdgeRecord(
                edge_id=_edge_id(RelationType.DECIDED_BY, record.case_id, SC2016_COURT_ID, (court_evidence,)),
                relation_type=RelationType.DECIDED_BY,
                source_id=record.case_id,
                target_id=SC2016_COURT_ID,
                evidence_extraction_ids=(court_evidence,),
                origin_split=record.origin_split,
                source_decision_date=record.decision_date,
            )
        )
        resolutions = resolve_candidates(
            extract_indian_references(record.text, record.document_id),
            known_target_ids,
        )
        for key, value in resolution_summary(resolutions).items():
            resolution_counts[key] += value
        references = build_reference_edges(
            record.case_id,
            resolutions,
            origin_split=record.origin_split,
            source_decision_date=record.decision_date,
        )
        added_edges.extend(references.edges)
        unresolved.extend(references.unresolved)

    graph = GraphSnapshot(
        version=version,
        schema_version=base_graph.schema_version,
        nodes=base_graph.nodes + tuple(added_nodes),
        edges=base_graph.edges + tuple(added_edges),
    )
    validate_graph(graph)
    return JudgmentGraphAssembly(
        graph=graph,
        unresolved=tuple(unresolved),
        resolution_counts=dict(sorted(resolution_counts.items())),
    )


def extraction_report(assembly: JudgmentGraphAssembly) -> dict[str, object]:
    """Return a compact, deterministic quality report without dropping evidence."""

    unresolved = tuple(sorted(assembly.unresolved, key=lambda item: item.candidate.extraction_id))
    return {
        "schema_version": "1.0.0",
        "summary": dict(assembly.resolution_counts),
        "unresolved_count": len(unresolved),
        "unresolved": [
            {
                "extraction_id": result.candidate.extraction_id,
                "document_id": result.candidate.document_id,
                "kind": result.candidate.kind.value,
                "relation_type": result.candidate.relation_type.value,
                "canonical_hint": result.candidate.canonical_hint,
                "unresolved_reason": result.unresolved_reason,
            }
            for result in unresolved
        ],
    }


def write_extraction_report(assembly: JudgmentGraphAssembly, output_path: Path) -> tuple[Path, str]:
    """Persist a content-addressed local unresolved-reference report."""

    payload = extraction_report(assembly)
    content = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(output_path)
    return output_path, hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_training_manifest(
    ingestion: JudgmentIngestion,
    *,
    corpus_sha256: str,
    graph_content_sha256: str,
) -> dict[str, object]:
    """Describe split-safe local judgment supervision without embedding legal text."""

    split_counts = {
        split.value: sum(record.origin_split is split for record in ingestion.judgments)
        for split in Split
        if any(record.origin_split is split for record in ingestion.judgments)
    }
    payload: dict[str, object] = {
        "artifact_type": "judgment_training_corpus",
        "artifact_version": "sc2016-training-0.1.0",
        "schema_version": "1.0.0",
        "status": "validated",
        "source_id": ingestion.source_id,
        "source_sha256": ingestion.source_sha256,
        "corpus_sha256": corpus_sha256,
        "graph_content_sha256": graph_content_sha256,
        "document_count": len(ingestion.judgments),
        "split_counts": split_counts,
        "training_splits": [Split.TRAIN.value],
        "validation_splits": [Split.VALIDATION.value],
        "held_out_splits": [Split.TEST.value],
        "split_strategy": "sha256-prefix-modulo-10-v1",
    }
    payload["content_sha256"] = sha256_payload(payload)
    return payload


def write_training_manifest(
    ingestion: JudgmentIngestion,
    output_path: Path,
    *,
    corpus_sha256: str,
    graph_content_sha256: str,
) -> tuple[Path, str]:
    """Write the local supervision lineage manifest required by model preflight."""

    payload = build_training_manifest(
        ingestion,
        corpus_sha256=corpus_sha256,
        graph_content_sha256=graph_content_sha256,
    )
    content = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="\n")
    temporary_path.replace(output_path)
    return output_path, hashlib.sha256(content.encode("utf-8")).hexdigest()


def _source_record(item: object) -> SourceRecord:
    if not isinstance(item, Mapping):
        raise ValueError("source registry entries must be objects")
    try:
        return SourceRecord(
            source_id=str(item["source_id"]),
            display_name=str(item["display_name"]),
            source_locator=str(item["source_locator"]),
            acquired_on=date.fromisoformat(str(item["acquired_on"])),
            raw_sha256=str(item["raw_sha256"]),
            license_label=str(item["license_label"]),
            status=str(item["status"]),
            permitted_uses=tuple(str(value) for value in item.get("permitted_uses", ())),
            redistribution_allowed=bool(item["redistribution_allowed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid source registry entry: {error}") from error


def _markdown_path(payload: Mapping[str, Any], markdown_root: Path, metadata_path: Path) -> Path:
    filename = payload.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".md"):
        raise ValueError(f"judgment metadata {metadata_path} has an unsafe or missing Markdown filename")
    markdown_path = markdown_root / filename
    if not markdown_path.is_file():
        raise FileNotFoundError(f"missing Markdown transcript for {metadata_path.name}: {markdown_path}")
    return markdown_path


def _judgment_record(
    metadata_path: Path,
    markdown_path: Path,
    payload: Mapping[str, Any],
    metadata_bytes: bytes,
    markdown_bytes: bytes,
    *,
    source_id: str,
    source_sha256: str,
) -> JudgmentRecord:
    text = markdown_bytes.decode("utf-8").replace("\r\n", "\n").strip()
    if not text:
        raise ValueError(f"judgment transcript {markdown_path} is empty")
    raw_sha256 = hashlib.sha256(metadata_bytes + b"\0" + markdown_bytes).hexdigest()
    entities = payload.get("entities")
    entity_map = entities if isinstance(entities, Mapping) else {}
    title = _title(entity_map, text, metadata_path)
    decision_date = _decision_date(text)
    metadata = payload.get("metadata")
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    attributes = {
        "dataset": "Shreyasrao/Indian-law-supreme-court-judgements-2016",
        "markdown_path": markdown_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "origin_split": _split_for_digest(raw_sha256).value,
        "raw_sha256": raw_sha256,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "year": str(metadata_map.get("year", "")),
        "judge_names": _entity_names(entity_map.get("judges")),
        "party_names": _entity_names(entity_map.get("parties")),
        "topics": _topic_names(entity_map.get("topics")),
    }
    return JudgmentRecord(
        document_id="doc:india:sc:judgment:sha256-" + raw_sha256[:16],
        case_id="case:india:sc:2016:sha256-" + raw_sha256[:16],
        source_id=source_id,
        source_sha256=source_sha256,
        raw_sha256=raw_sha256,
        markdown_path=markdown_path,
        metadata_path=metadata_path,
        title=title,
        court_id=SC2016_COURT_ID,
        court_label=SC2016_COURT_LABEL,
        decision_date=decision_date,
        origin_split=_split_for_digest(raw_sha256),
        text=text,
        attributes=attributes,
    )


def _title(entities: Mapping[str, Any], text: str, metadata_path: Path) -> str:
    case_title = entities.get("case_title")
    if isinstance(case_title, Mapping) and isinstance(case_title.get("title"), str):
        title = _normalise_label(case_title["title"])
        if title:
            return title
    for line in text.splitlines():
        if line.startswith("#"):
            title = _normalise_label(line.lstrip("#"))
            if title:
                return title
    return metadata_path.stem


def _decision_date(text: str) -> date | None:
    header = text[:4000]
    for pattern in (_MONTH_DATE_PATTERN, _DATE_MONTH_PATTERN):
        match = pattern.search(header)
        if not match:
            continue
        try:
            return datetime.strptime(
                f"{match.group('month').title()} {match.group('day')} {match.group('year')}",
                "%B %d %Y",
            ).date()
        except ValueError:
            continue
    return None


def _entity_names(value: object) -> str:
    if not isinstance(value, list):
        return ""
    names = [
        _normalise_label(item.get("name", ""))
        for item in value
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    ]
    return " | ".join(name for name in names if name)


def _topic_names(value: object) -> str:
    if not isinstance(value, list):
        return ""
    topics = [
        _normalise_label(item.get("text", ""))
        for item in value
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    ]
    return " | ".join(topic for topic in topics if topic)


def _normalise_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _split_for_digest(digest: str) -> Split:
    bucket = int(digest[:8], 16) % 10
    if bucket < 8:
        return Split.TRAIN
    if bucket == 8:
        return Split.VALIDATION
    return Split.TEST


def _validate_unique_case_ids(records: Iterable[JudgmentRecord]) -> None:
    seen: set[str] = set()
    duplicates = {record.case_id for record in records if record.case_id in seen or seen.add(record.case_id)}
    if duplicates:
        raise ValueError(f"duplicate SC-2016 case IDs: {sorted(duplicates)}")


def _evidence_id(kind: str, raw_sha256: str) -> str:
    return "extract:judgment:" + hashlib.sha256(f"{kind}|{raw_sha256}".encode("utf-8")).hexdigest()[:16]


def _edge_id(
    relation_type: RelationType,
    source_id: str,
    target_id: str,
    evidence_ids: tuple[str, ...],
) -> str:
    seed = "|".join((relation_type.value, source_id, target_id, *evidence_ids))
    return "edge:" + relation_type.value + ":" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
