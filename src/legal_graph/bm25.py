"""Deterministic, dependency-free BM25 retrieval over local provision corpora."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .manifests import ArtifactManifest, canonical_json_bytes, sha256_payload


INDEX_SCHEMA_VERSION = "1.0.0"
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Bm25Config:
    """Pinned BM25 policy; changing a value requires a new index version."""

    strategy: str = "bm25_v1"
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between zero and one")


@dataclass(frozen=True, slots=True)
class Bm25Document:
    node_id: str
    text: str
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Bm25Index:
    index_version: str
    corpus_sha256: str
    config: Bm25Config
    documents: tuple[Bm25Document, ...]
    document_lengths: tuple[int, ...]
    document_frequencies: Mapping[str, int]
    postings: Mapping[str, tuple[tuple[int, int], ...]]
    average_document_length: float


@dataclass(frozen=True, slots=True)
class Bm25SearchResult:
    node_id: str
    score: float
    metadata: Mapping[str, str]


def read_provision_corpus(path: Path) -> tuple[Bm25Document, ...]:
    """Read a locally generated provision JSONL file with strict required fields."""

    documents: list[Bm25Document] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            node_id = record["node_id"]
            text = record["text"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid corpus JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(node_id, str) or not node_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid node_id/text at {path}:{line_number}")
        metadata = {
            key: str(record[key])
            for key in ("act_id", "act_title", "section", "source_pdf", "source_page_start", "source_page_end")
            if key in record
        }
        documents.append(Bm25Document(node_id=node_id, text=text, metadata=metadata))
    if not documents:
        raise ValueError(f"no provision documents found in {path}")
    return tuple(documents)


def corpus_sha256(path: Path) -> str:
    """Return the checksum of the exact input corpus bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_bm25_index(
    documents: Sequence[Bm25Document],
    *,
    corpus_digest: str,
    config: Bm25Config | None = None,
    index_version: str = "bm25-india-statutes-0.1.0",
) -> Bm25Index:
    """Build a deterministic in-memory BM25 index from one immutable corpus."""

    if not re.fullmatch(r"[0-9a-f]{64}", corpus_digest):
        raise ValueError("corpus_digest must be a lowercase SHA-256 hex digest")
    policy = config or Bm25Config()
    ordered = tuple(sorted(documents, key=lambda item: item.node_id))
    if not ordered:
        raise ValueError("BM25 index requires at least one document")
    if len({document.node_id for document in ordered}) != len(ordered):
        raise ValueError("BM25 documents require unique node IDs")

    lengths: list[int] = []
    frequencies: Counter[str] = Counter()
    posting_builder: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for document_index, document in enumerate(ordered):
        terms = tokenize(document.text)
        if not terms:
            raise ValueError(f"BM25 document {document.node_id!r} contains no indexable tokens")
        lengths.append(len(terms))
        counts = Counter(terms)
        frequencies.update(counts.keys())
        for term, term_frequency in sorted(counts.items()):
            posting_builder[term].append((document_index, term_frequency))

    return Bm25Index(
        index_version=index_version,
        corpus_sha256=corpus_digest,
        config=policy,
        documents=ordered,
        document_lengths=tuple(lengths),
        document_frequencies=dict(sorted(frequencies.items())),
        postings={term: tuple(items) for term, items in sorted(posting_builder.items())},
        average_document_length=sum(lengths) / len(lengths),
    )


def search_bm25(index: Bm25Index, query: str, *, limit: int = 10) -> tuple[Bm25SearchResult, ...]:
    """Score a query against the exact corpus used to build the index."""

    if limit < 1:
        raise ValueError("limit must be at least one")
    query_terms = tuple(dict.fromkeys(tokenize(query)))
    if not query_terms:
        return ()

    scores: defaultdict[int, float] = defaultdict(float)
    document_count = len(index.documents)
    for term in query_terms:
        frequency = index.document_frequencies.get(term)
        if not frequency:
            continue
        inverse_frequency = math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for document_index, term_frequency in index.postings[term]:
            document_length = index.document_lengths[document_index]
            normalizer = term_frequency + index.config.k1 * (
                1 - index.config.b + index.config.b * document_length / index.average_document_length
            )
            scores[document_index] += inverse_frequency * (
                term_frequency * (index.config.k1 + 1) / normalizer
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], index.documents[item[0]].node_id),
    )[:limit]
    return tuple(
        Bm25SearchResult(
            node_id=index.documents[document_index].node_id,
            score=round(score, 12),
            metadata=index.documents[document_index].metadata,
        )
        for document_index, score in ranked
    )


def write_bm25_index(
    index: Bm25Index,
    output_dir: Path,
    *,
    code_revision: str = "local-uncommitted",
    status: str = "provisional",
    approval_policy_sha256: str | None = None,
) -> tuple[Path, Path, ArtifactManifest]:
    """Write a canonical local index plus a lineage manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "bm25.index.json"
    manifest_path = output_dir / "bm25.manifest.json"
    if status == "validated" and not _is_sha256(approval_policy_sha256):
        raise ValueError("validated BM25 indexes require an approved policy SHA-256")
    payload = index_to_dict(index)
    _atomic_write(index_path, canonical_json_bytes(payload) + b"\n")
    manifest = ArtifactManifest(
        artifact_type="bm25_index",
        artifact_version=index.index_version,
        content_sha256=sha256_payload(payload),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        code_revision=code_revision,
        config_sha256=sha256_payload(asdict(index.config)),
        parents=(f"sha256:{index.corpus_sha256}",)
        + ((f"approval-policy:sha256:{approval_policy_sha256}",) if approval_policy_sha256 else ()),
        schema_version=INDEX_SCHEMA_VERSION,
        status=status,
        storage_uri=f"local://{output_dir.as_posix()}",
    )
    _atomic_write(manifest_path, canonical_json_bytes(manifest.to_dict()) + b"\n")
    return index_path, manifest_path, manifest


def load_bm25_index(path: Path) -> Bm25Index:
    """Load a serialized index and reject mismatched posting/doc references."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["schema_version"] != INDEX_SCHEMA_VERSION:
            raise ValueError(f"unsupported index schema {payload['schema_version']!r}")
        config = Bm25Config(**payload["config"])
        documents = tuple(
            Bm25Document(
                node_id=item["node_id"],
                text="",
                metadata=item.get("metadata", {}),
            )
            for item in payload["documents"]
        )
        index = Bm25Index(
            index_version=payload["index_version"],
            corpus_sha256=payload["corpus_sha256"],
            config=config,
            documents=documents,
            document_lengths=tuple(payload["document_lengths"]),
            document_frequencies=payload["document_frequencies"],
            postings={term: tuple(tuple(item) for item in values) for term, values in payload["postings"].items()},
            average_document_length=float(payload["average_document_length"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid BM25 index {path}: {error}") from error
    _validate_index(index)
    return index


def index_to_dict(index: Bm25Index) -> dict[str, object]:
    """Return the portable representation; provision text is not duplicated."""

    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "index_version": index.index_version,
        "corpus_sha256": index.corpus_sha256,
        "config": asdict(index.config),
        "documents": [
            {"node_id": document.node_id, "metadata": dict(sorted(document.metadata.items()))}
            for document in index.documents
        ],
        "document_lengths": list(index.document_lengths),
        "document_frequencies": dict(sorted(index.document_frequencies.items())),
        "postings": {
            term: [list(item) for item in postings]
            for term, postings in sorted(index.postings.items())
        },
        "average_document_length": index.average_document_length,
    }


def tokenize(text: str) -> tuple[str, ...]:
    """Use a small documented tokenizer that is stable across local environments."""

    return tuple(_TOKEN.findall(text.lower()))


def _validate_index(index: Bm25Index) -> None:
    document_count = len(index.documents)
    if document_count == 0 or len(index.document_lengths) != document_count:
        raise ValueError("BM25 index document lengths do not match documents")
    if len({document.node_id for document in index.documents}) != document_count:
        raise ValueError("BM25 index has duplicate document IDs")
    if any(length < 1 for length in index.document_lengths) or index.average_document_length <= 0:
        raise ValueError("BM25 index has invalid document lengths")
    for term, postings in index.postings.items():
        if index.document_frequencies.get(term) != len(postings):
            raise ValueError(f"BM25 index has inconsistent document frequency for {term!r}")
        if any(
            document_index < 0 or document_index >= document_count or frequency < 1
            for document_index, frequency in postings
        ):
            raise ValueError(f"BM25 index has an invalid posting for {term!r}")



def _is_sha256(value: str | None) -> bool:
    return value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is not None

def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
