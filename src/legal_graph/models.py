"""Core immutable records shared by graph, retrieval, and model layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Mapping


class NodeType(StrEnum):
    CASE = "case"
    RHETORICAL_UNIT = "rhetorical_unit"
    PROVISION = "provision"
    CONTRACT_CLAUSE = "contract_clause"
    LEGAL_CONCEPT = "legal_concept"
    COURT = "court"
    PARTY = "party"
    DATE_ANCHOR = "date_anchor"
    ACT = "act"


class RelationType(StrEnum):
    CITES = "cites"
    IS_CITED_BY = "is_cited_by"
    OVERRULES = "overrules"
    DISTINGUISHES = "distinguishes"
    REFERS_TO_PROVISION = "refers_to_provision"
    CONTAINS = "contains"
    DEFINES = "defines"
    AMENDS = "amends"
    DECIDED_BY = "decided_by"
    CO_OCCURS_WITH = "co_occurs_with"
    TEMPORALLY_PRECEDES = "temporally_precedes"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class NodeRecord:
    """A canonical graph node with source and temporal provenance."""

    node_id: str
    node_type: NodeType
    label: str
    source_document_ids: tuple[str, ...]
    available_from: date | None = None
    available_to: date | None = None
    decision_date: date | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """A typed directed edge backed by at least one extraction record."""

    edge_id: str
    relation_type: RelationType
    source_id: str
    target_id: str
    evidence_extraction_ids: tuple[str, ...]
    origin_split: Split
    confidence: float = 1.0
    available_from: date | None = None
    available_to: date | None = None
    source_decision_date: date | None = None


@dataclass(frozen=True, slots=True)
class GraphSnapshot:
    """An immutable, versioned collection of legal nodes and relations."""

    version: str
    schema_version: str
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]

    @property
    def nodes_by_id(self) -> dict[str, NodeRecord]:
        return {node.node_id: node for node in self.nodes}


@dataclass(frozen=True, slots=True)
class GraphValidationReport:
    """Validation output that can be saved beside a graph release manifest."""

    graph_version: str
    node_count: int
    edge_count: int
    relation_counts: Mapping[str, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """All fields that materially affect a deterministic retrieval result."""

    query_id: str
    seed_node_ids: tuple[str, ...]
    node_prizes: Mapping[str, float] = field(default_factory=dict)
    as_of_date: date | None = None
    allowed_origin_splits: frozenset[Split] = frozenset({Split.TRAIN, Split.VALIDATION})
    allowed_relation_types: frozenset[RelationType] | None = None


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Versioned retrieval policy for the vertical-slice selector."""

    strategy: str = "prize_guided_connected_expansion_v1"
    node_budget: int = 80
    expansion_hops: int = 2
    require_connected: bool = True


@dataclass(frozen=True, slots=True)
class SubgraphBundle:
    """A deterministic, traceable input bundle for the future graph encoder."""

    graph_version: str
    query_id: str
    retrieval_strategy: str
    selected_node_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    cache_key: str
    status: str
    diagnostics: Mapping[str, str] = field(default_factory=dict)
