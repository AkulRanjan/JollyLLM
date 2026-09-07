"""Offline ingestion of local Indian statute PDFs into provenance-backed provision graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from pypdf import PdfReader

from .approvals import load_approval_policy, validate_policy_sources
from .manifests import create_graph_manifest, write_graph_and_manifest
from .models import EdgeRecord, GraphSnapshot, NodeRecord, NodeType, RelationType, Split
from .validation import validate_graph


_BODY_MARKER = re.compile(r"\bBE\s+it\s+enacted\b", re.IGNORECASE)
_SECTION_START = re.compile(
    r"(?m)^\s*(?P<section>[1-9]\d{0,2}[A-Z]?)\.(?=\s|\(|[A-Za-z\"“])"
)

_EMBEDDED_MATERIAL = re.compile(
    r"(?im)^\s*(?:the\s+)?(?:(?:first|second|third|fourth|fifth|sixth|"
    r"seventh|eighth|ninth|tenth)\s+)?(?:schedule|appendix|state\s+amendments?)\b"
)

_PAGE_MARKER = re.compile(r"\n?<<PAGE:\d+>>\n?")


@dataclass(frozen=True, slots=True)
class StatuteSpec:
    act_id: str
    title: str
    enactment_date: date
    filename_markers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Provision:
    section: str
    text: str
    source_page_start: int
    source_page_end: int


@dataclass(frozen=True, slots=True)
class StatuteIngestion:
    spec: StatuteSpec
    document_id: str
    source_path: Path
    source_sha256: str
    page_count: int
    body_start_page: int
    provisions: tuple[Provision, ...]
    duplicate_sections: tuple[str, ...]


KNOWN_STATUTES: tuple[StatuteSpec, ...] = (
    StatuteSpec(
        act_id="bns",
        title="Bharatiya Nyaya Sanhita, 2023",
        enactment_date=date(2023, 12, 25),
        filename_markers=("bharatiya-nyaya-sanhita", "bns"),
    ),
    StatuteSpec(
        act_id="bnss",
        title="Bharatiya Nagarik Suraksha Sanhita, 2023",
        enactment_date=date(2023, 12, 25),
        filename_markers=("bharatiya_nagarik_suraksha_sanhita", "bnss"),
    ),
    StatuteSpec(
        act_id="crpc",
        title="Code of Criminal Procedure, 1973",
        enactment_date=date(1974, 1, 25),
        filename_markers=("code_of_criminal_procedure", "crpc"),
    ),
)


def infer_statute_spec(path: Path) -> StatuteSpec:
    """Infer the known act from a local filename; require an explicit spec if unknown."""

    normalized_name = path.name.lower().replace(" ", "_")
    matches = [
        spec
        for spec in KNOWN_STATUTES
        if any(marker in normalized_name for marker in spec.filename_markers)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot infer a unique statute spec from {path.name!r}; "
            "provide a recognised BNS, BNSS, or CrPC filename"
        )
    return matches[0]


def ingest_statute_pdf(path: Path, spec: StatuteSpec | None = None) -> StatuteIngestion:
    """Extract numbered provisions from a text-readable local statute PDF."""

    if not path.is_file():
        raise FileNotFoundError(path)
    statute = spec or infer_statute_spec(path)
    reader = PdfReader(str(path))
    pages = tuple(reader_page.extract_text() or "" for reader_page in reader.pages)
    body_start_index = _find_body_start(pages)
    body_pages = tuple(
        (page_index + 1, _crop_first_body_page(text, page_index == body_start_index))
        for page_index, text in enumerate(pages[body_start_index:], start=body_start_index)
    )
    provisions, duplicates = extract_provisions_from_pages(body_pages)
    if not provisions:
        raise ValueError(f"no numbered provisions found in {path.name!r}")
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    document_id = f"doc:india:statute:{statute.act_id}:{source_hash[:16]}"
    return StatuteIngestion(
        spec=statute,
        document_id=document_id,
        source_path=path,
        source_sha256=source_hash,
        page_count=len(pages),
        body_start_page=body_start_index + 1,
        provisions=provisions,
        duplicate_sections=duplicates,
    )


def extract_provisions_from_pages(
    pages: Sequence[tuple[int, str]],
) -> tuple[tuple[Provision, ...], tuple[str, ...]]:
    """Split act-body page text into de-duplicated, page-provenanced provisions."""

    candidates = extract_provision_candidates_from_pages(pages)
    selected: list[Provision] = []
    excluded_sections: list[str] = []
    highest_key: tuple[int, str] | None = None
    for provision in candidates:
        section_key = _section_sort_key(provision.section)
        if highest_key is not None and section_key <= highest_key:
            # Consolidated PDFs can embed state amendments, schedules, and forms
            # between or after the Act's provisions. Their local numbering often
            # restarts at 1, so accepting it would corrupt the statute corpus.
            excluded_sections.append(provision.section)
            continue
        selected.append(provision)
        highest_key = section_key

    return tuple(selected), tuple(sorted(set(excluded_sections), key=_section_sort_key))


def extract_provision_candidates_from_pages(pages: Sequence[tuple[int, str]]) -> tuple[Provision, ...]:
    """Return every numbered heading, including duplicate section identifiers.

    This preserves the evidence needed by the quality-audit step. Consumers that
    need a searchable corpus must continue to use extract_provisions_from_pages,
    which keeps the earliest occurrence and reports duplicates separately.
    """

    combined_parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_number, text in pages:
        page_offsets.append((cursor, page_number))
        combined_parts.append(text)
        cursor += len(text)
        combined_parts.append(f"\n<<PAGE:{page_number}>>\n")
        cursor += len(combined_parts[-1])
    body_text = "".join(combined_parts)
    matches = tuple(_SECTION_START.finditer(body_text))
    candidates: list[Provision] = []

    for index, match in enumerate(matches):
        section = match.group("section")
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body_text)
        raw_text = body_text[start:end]
        embedded_material = _EMBEDDED_MATERIAL.search(raw_text)
        content_end = start + embedded_material.start() if embedded_material else end
        text = _PAGE_MARKER.sub("", body_text[start:content_end]).strip()
        start_page = _page_for_offset(start, page_offsets)
        end_page = _page_for_offset(max(start, content_end - 1), page_offsets)
        candidates.append(
            Provision(
                section=section,
                text=text,
                source_page_start=start_page,
                source_page_end=end_page,
            )
        )
    return tuple(candidates)


def provision_records(ingestions: Iterable[StatuteIngestion]) -> Iterable[dict[str, object]]:
    """Yield provision text with stable identifiers and source-page provenance."""

    for ingestion in sorted(ingestions, key=lambda item: item.spec.act_id):
        version_id = f"sha256-{ingestion.source_sha256[:16]}"
        for provision in ingestion.provisions:
            node_id = f"provision:india:{ingestion.spec.act_id}:{provision.section}:{version_id}"
            yield {
                "node_id": node_id,
                "document_id": ingestion.document_id,
                "act_id": ingestion.spec.act_id,
                "act_title": ingestion.spec.title,
                "section": provision.section,
                "text": provision.text,
                "source_pdf": ingestion.source_path.as_posix(),
                "source_sha256": ingestion.source_sha256,
                "source_page_start": provision.source_page_start,
                "source_page_end": provision.source_page_end,
            }


def write_provision_corpus(
    ingestions: Iterable[StatuteIngestion],
    output_path: Path,
) -> tuple[Path, str, int]:
    """Write a deterministic local JSONL corpus and return its checksum and count."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in provision_records(ingestions):
            line = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    temporary_path.replace(output_path)
    return output_path, digest.hexdigest(), count


def build_statute_graph(
    ingestions: Iterable[StatuteIngestion],
    *,
    version: str = "india-statutes-0.1.0",
) -> GraphSnapshot:
    """Materialise local act/provision nodes and authoritative containment edges."""

    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    for ingestion in sorted(ingestions, key=lambda item: item.spec.act_id):
        act_node_id = f"act:india:{ingestion.spec.act_id}"
        nodes.append(
            NodeRecord(
                node_id=act_node_id,
                node_type=NodeType.ACT,
                label=ingestion.spec.title,
                source_document_ids=(ingestion.document_id,),
                available_from=ingestion.spec.enactment_date,
                attributes={
                    "source_pdf": ingestion.source_path.as_posix(),
                    "source_sha256": ingestion.source_sha256,
                    "body_start_page": str(ingestion.body_start_page),
                },
            )
        )
        version_id = f"sha256-{ingestion.source_sha256[:16]}"
        for provision in ingestion.provisions:
            node_id = f"provision:india:{ingestion.spec.act_id}:{provision.section}:{version_id}"
            nodes.append(
                NodeRecord(
                    node_id=node_id,
                    node_type=NodeType.PROVISION,
                    label=f"{ingestion.spec.title} section {provision.section}",
                    source_document_ids=(ingestion.document_id,),
                    available_from=ingestion.spec.enactment_date,
                    attributes={
                        "act_id": ingestion.spec.act_id,
                        "section": provision.section,
                        "source_pdf": ingestion.source_path.as_posix(),
                        "source_sha256": ingestion.source_sha256,
                        "source_page_start": str(provision.source_page_start),
                        "source_page_end": str(provision.source_page_end),
                        "text_sha256": hashlib.sha256(provision.text.encode("utf-8")).hexdigest(),
                    },
                )
            )
            evidence_seed = f"{ingestion.document_id}|{provision.section}|{provision.source_page_start}"
            evidence_id = "extract:statute:" + hashlib.sha256(evidence_seed.encode("utf-8")).hexdigest()[:16]
            edge_seed = f"{act_node_id}|{node_id}|{evidence_id}"
            edges.append(
                EdgeRecord(
                    edge_id="edge:contains:" + hashlib.sha256(edge_seed.encode("utf-8")).hexdigest()[:20],
                    relation_type=RelationType.CONTAINS,
                    source_id=act_node_id,
                    target_id=node_id,
                    evidence_extraction_ids=(evidence_id,),
                    origin_split=Split.EXTERNAL,
                    available_from=ingestion.spec.enactment_date,
                )
            )

    graph = GraphSnapshot(
        version=version,
        schema_version="1.0.0",
        nodes=tuple(nodes),
        edges=tuple(edges),
    )
    validate_graph(graph)
    return graph


def _find_body_start(pages: Sequence[str]) -> int:
    for index, text in enumerate(pages):
        if _BODY_MARKER.search(text):
            return index
    raise ValueError("could not find the act body marker 'BE it enacted'")


def _crop_first_body_page(text: str, is_first_body_page: bool) -> str:
    if not is_first_body_page:
        return text
    marker = _BODY_MARKER.search(text)
    return text[marker.start():] if marker else text


def _page_for_offset(offset: int, page_offsets: Sequence[tuple[int, int]]) -> int:
    starts = [item[0] for item in page_offsets]
    index = bisect_right(starts, offset) - 1
    return page_offsets[max(index, 0)][1]


def _section_sort_key(section: str) -> tuple[int, str]:
    match = re.fullmatch(r"(?P<number>\d+)(?P<suffix>[A-Z]?)", section)
    if not match:
        return (10**9, section)
    return (int(match.group("number")), match.group("suffix"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local statute PDFs into a validated provision graph")
    parser.add_argument("pdf_paths", nargs="+", type=Path, help="local BNS, BNSS, or CrPC PDF paths")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/graphs/india_statutes"),
        help="local graph and manifest output directory",
    )
    parser.add_argument("--code-revision", default="local-uncommitted")
    parser.add_argument(
        "--approval-policy",
        type=Path,
        help="approved local extraction-policy JSON bound to these source PDFs",
    )
    parser.add_argument(
        "--corpus-output",
        type=Path,
        default=Path("data/clean/india_statutes/provisions.jsonl"),
        help="local JSONL output containing provision text and provenance",
    )
    arguments = parser.parse_args()

    ingestions = tuple(ingest_statute_pdf(path) for path in arguments.pdf_paths)
    approval_policy = load_approval_policy(arguments.approval_policy) if arguments.approval_policy else None
    if approval_policy:
        validate_policy_sources(
            approval_policy,
            {ingestion.spec.act_id: ingestion.source_sha256 for ingestion in ingestions},
        )
    corpus_path, corpus_sha256, corpus_count = write_provision_corpus(ingestions, arguments.corpus_output)
    graph = build_statute_graph(ingestions)
    manifest = create_graph_manifest(
        graph,
        code_revision=arguments.code_revision,
        config_payload={
            "ingester": "statute-pdf-v1",
            "acts": [ingestion.spec.act_id for ingestion in ingestions],
            "source_hashes": [ingestion.source_sha256 for ingestion in ingestions],
            "approval_policy_id": approval_policy.policy_id if approval_policy else "unapproved",
            "approval_policy_sha256": approval_policy.policy_sha256 if approval_policy else None,
        },
        parents=tuple(ingestion.document_id for ingestion in ingestions)
        + ((f"approval-policy:sha256:{approval_policy.policy_sha256}",) if approval_policy else ()),
        storage_uri=f"local://{arguments.output_dir.as_posix()}",
        status="validated" if approval_policy else "provisional",
    )
    graph_path, manifest_path = write_graph_and_manifest(graph, manifest, arguments.output_dir)
    print(
        json.dumps(
            {
                "graph_version": graph.version,
                "graph_path": str(graph_path),
                "manifest_path": str(manifest_path),
                "provision_corpus_path": str(corpus_path),
                "provision_corpus_sha256": corpus_sha256,
                "provision_corpus_count": corpus_count,
                "acts": {
                    ingestion.spec.act_id: {
                        "page_count": ingestion.page_count,
                        "body_start_page": ingestion.body_start_page,
                        "provision_count": len(ingestion.provisions),
                        "duplicate_sections": list(ingestion.duplicate_sections),
                        "source_sha256": ingestion.source_sha256,
                    }
                    for ingestion in ingestions
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
