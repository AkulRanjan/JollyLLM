"""Local provenance, serialization, and content-addressed artifact manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import EdgeRecord, GraphSnapshot, NodeRecord, NodeType, RelationType, Split


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """A reviewable intake record; real corpus ingestion requires status=approved."""

    source_id: str
    display_name: str
    source_locator: str
    acquired_on: date
    raw_sha256: str
    license_label: str
    status: str
    permitted_uses: tuple[str, ...]
    redistribution_allowed: bool


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    artifact_type: str
    artifact_version: str
    content_sha256: str
    created_at: str
    code_revision: str
    config_sha256: str
    parents: tuple[str, ...]
    schema_version: str
    status: str
    storage_uri: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegistryValidationError(ValueError):
    """Raised when a local source registry is unfit for use."""


def validate_source_records(records: Sequence[SourceRecord]) -> None:
    """Validate source provenance before any real-data pipeline may consume it."""

    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not record.source_id:
            errors.append("source has an empty source_id")
        elif record.source_id in seen:
            errors.append(f"duplicate source_id {record.source_id!r}")
        seen.add(record.source_id)

        if not record.display_name or not record.source_locator:
            errors.append(f"source {record.source_id!r} needs a display_name and source_locator")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", record.raw_sha256):
            errors.append(f"source {record.source_id!r} raw_sha256 must be a 64-character hexadecimal digest")
        if record.status not in {"approved", "pending", "excluded"}:
            errors.append(f"source {record.source_id!r} has unsupported status {record.status!r}")
        if record.status == "approved" and not record.permitted_uses:
            errors.append(f"approved source {record.source_id!r} has no permitted_uses")
        if record.status == "excluded" and record.redistribution_allowed:
            errors.append(f"excluded source {record.source_id!r} cannot be redistributable")
        if not record.license_label:
            errors.append(f"source {record.source_id!r} has no license_label")
    if errors:
        raise RegistryValidationError("Source registry validation failed:\n- " + "\n- ".join(errors))


def canonical_json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    """Stable JSON representation used for content addressing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_payload(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def graph_to_dict(graph: GraphSnapshot) -> dict[str, Any]:
    """Create a portable graph payload that preserves provenance and dates."""

    return {
        "version": graph.version,
        "schema_version": graph.schema_version,
        "nodes": [
            {
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "label": node.label,
                "source_document_ids": list(node.source_document_ids),
                "available_from": _date_text(node.available_from),
                "available_to": _date_text(node.available_to),
                "decision_date": _date_text(node.decision_date),
                "attributes": dict(sorted(node.attributes.items())),
            }
            for node in graph.nodes
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "relation_type": edge.relation_type.value,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "evidence_extraction_ids": list(edge.evidence_extraction_ids),
                "origin_split": edge.origin_split.value,
                "confidence": edge.confidence,
                "available_from": _date_text(edge.available_from),
                "available_to": _date_text(edge.available_to),
                "source_decision_date": _date_text(edge.source_decision_date),
            }
            for edge in graph.edges
        ],
    }


def graph_from_dict(payload: Mapping[str, Any]) -> GraphSnapshot:
    """Load a graph payload without accepting untyped/ambiguous relation values."""

    try:
        nodes = tuple(
            NodeRecord(
                node_id=item["node_id"],
                node_type=NodeType(item["node_type"]),
                label=item["label"],
                source_document_ids=tuple(item["source_document_ids"]),
                available_from=_parse_date(item.get("available_from")),
                available_to=_parse_date(item.get("available_to")),
                decision_date=_parse_date(item.get("decision_date")),
                attributes=item.get("attributes", {}),
            )
            for item in payload["nodes"]
        )
        edges = tuple(
            EdgeRecord(
                edge_id=item["edge_id"],
                relation_type=RelationType(item["relation_type"]),
                source_id=item["source_id"],
                target_id=item["target_id"],
                evidence_extraction_ids=tuple(item["evidence_extraction_ids"]),
                origin_split=Split(item["origin_split"]),
                confidence=float(item.get("confidence", 1.0)),
                available_from=_parse_date(item.get("available_from")),
                available_to=_parse_date(item.get("available_to")),
                source_decision_date=_parse_date(item.get("source_decision_date")),
            )
            for item in payload["edges"]
        )
        return GraphSnapshot(
            version=payload["version"],
            schema_version=payload["schema_version"],
            nodes=nodes,
            edges=edges,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid graph snapshot payload: {error}") from error


def create_graph_manifest(
    graph: GraphSnapshot,
    *,
    code_revision: str,
    config_payload: Mapping[str, Any],
    parents: Sequence[str] = (),
    storage_uri: str = "local://artifacts",
    created_at: datetime | None = None,
    status: str = "validated",
) -> ArtifactManifest:
    """Create local lineage metadata without embedding legal-source text."""

    timestamp = created_at or datetime.now(timezone.utc)
    return ArtifactManifest(
        artifact_type="graph_snapshot",
        artifact_version=graph.version,
        content_sha256=sha256_payload(graph_to_dict(graph)),
        created_at=timestamp.isoformat().replace("+00:00", "Z"),
        code_revision=code_revision,
        config_sha256=sha256_payload(config_payload),
        parents=tuple(parents),
        schema_version=graph.schema_version,
        status=status,
        storage_uri=storage_uri,
    )


def write_graph_and_manifest(
    graph: GraphSnapshot,
    manifest: ArtifactManifest,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Export caller-approved artifacts locally using atomic replacements."""

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph.json"
    manifest_path = output_dir / "graph.manifest.json"
    _atomic_write(graph_path, canonical_json_bytes(graph_to_dict(graph)) + b"\n")
    _atomic_write(manifest_path, canonical_json_bytes(manifest.to_dict()) + b"\n")
    return graph_path, manifest_path


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def _date_text(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None
