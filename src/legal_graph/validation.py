"""Graph schema and temporal validation with explicit, actionable failures."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import EdgeRecord, GraphSnapshot, GraphValidationReport, NodeType, RelationType


@dataclass(frozen=True, slots=True)
class GraphValidationError(ValueError):
    """Raised when a graph snapshot violates its publishable contract."""

    errors: tuple[str, ...]

    def __str__(self) -> str:
        return "Graph validation failed:\n- " + "\n- ".join(self.errors)


RELATION_DOMAINS: dict[RelationType, frozenset[NodeType]] = {
    RelationType.CITES: frozenset({NodeType.CASE}),
    RelationType.IS_CITED_BY: frozenset({NodeType.CASE}),
    RelationType.OVERRULES: frozenset({NodeType.CASE}),
    RelationType.DISTINGUISHES: frozenset({NodeType.CASE}),
    RelationType.REFERS_TO_PROVISION: frozenset({NodeType.CASE, NodeType.CONTRACT_CLAUSE}),
    RelationType.CONTAINS: frozenset({NodeType.ACT, NodeType.CASE, NodeType.CONTRACT_CLAUSE}),
    RelationType.DEFINES: frozenset({NodeType.PROVISION, NodeType.CONTRACT_CLAUSE}),
    RelationType.AMENDS: frozenset({NodeType.PROVISION}),
    RelationType.DECIDED_BY: frozenset({NodeType.CASE}),
    RelationType.CO_OCCURS_WITH: frozenset({NodeType.PROVISION}),
    RelationType.TEMPORALLY_PRECEDES: frozenset({NodeType.CASE}),
}

RELATION_RANGES: dict[RelationType, frozenset[NodeType]] = {
    RelationType.CITES: frozenset({NodeType.CASE}),
    RelationType.IS_CITED_BY: frozenset({NodeType.CASE}),
    RelationType.OVERRULES: frozenset({NodeType.CASE}),
    RelationType.DISTINGUISHES: frozenset({NodeType.CASE}),
    RelationType.REFERS_TO_PROVISION: frozenset({NodeType.PROVISION}),
    RelationType.CONTAINS: frozenset({NodeType.PROVISION, NodeType.RHETORICAL_UNIT, NodeType.CONTRACT_CLAUSE}),
    RelationType.DEFINES: frozenset({NodeType.LEGAL_CONCEPT}),
    RelationType.AMENDS: frozenset({NodeType.PROVISION}),
    RelationType.DECIDED_BY: frozenset({NodeType.COURT}),
    RelationType.CO_OCCURS_WITH: frozenset({NodeType.PROVISION}),
    RelationType.TEMPORALLY_PRECEDES: frozenset({NodeType.CASE}),
}


def validate_graph(graph: GraphSnapshot) -> GraphValidationReport:
    """Validate a graph before it can be used to build an index or train a model."""

    errors: list[str] = []
    warnings: list[str] = []
    node_ids = [node.node_id for node in graph.nodes]
    edge_ids = [edge.edge_id for edge in graph.edges]

    _append_duplicate_errors("node", node_ids, errors)
    _append_duplicate_errors("edge", edge_ids, errors)

    nodes = graph.nodes_by_id
    for node in graph.nodes:
        if not node.node_id:
            errors.append("node has an empty node_id")
        if not node.label.strip():
            errors.append(f"node {node.node_id!r} has an empty label")
        if not node.source_document_ids:
            errors.append(f"node {node.node_id!r} has no source_document_ids")
        if node.available_from and node.available_to and node.available_from > node.available_to:
            errors.append(f"node {node.node_id!r} has an inverted validity window")

    for edge in graph.edges:
        _validate_edge(edge, nodes, errors, warnings)

    if errors:
        raise GraphValidationError(tuple(errors))

    relation_counts = Counter(edge.relation_type.value for edge in graph.edges)
    return GraphValidationReport(
        graph_version=graph.version,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        relation_counts=dict(sorted(relation_counts.items())),
        warnings=tuple(warnings),
    )


def _append_duplicate_errors(record_type: str, identifiers: list[str], errors: list[str]) -> None:
    for identifier, count in Counter(identifiers).items():
        if not identifier:
            errors.append(f"{record_type} has an empty identifier")
        elif count > 1:
            errors.append(f"duplicate {record_type}_id {identifier!r}")


def _validate_edge(
    edge: EdgeRecord,
    nodes: dict[str, object],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not edge.evidence_extraction_ids:
        errors.append(f"edge {edge.edge_id!r} has no evidence_extraction_ids")
    if not 0.0 <= edge.confidence <= 1.0:
        errors.append(f"edge {edge.edge_id!r} confidence must be between 0 and 1")
    if edge.available_from and edge.available_to and edge.available_from > edge.available_to:
        errors.append(f"edge {edge.edge_id!r} has an inverted validity window")

    source = nodes.get(edge.source_id)
    target = nodes.get(edge.target_id)
    if source is None:
        errors.append(f"edge {edge.edge_id!r} source {edge.source_id!r} does not resolve")
        return
    if target is None:
        errors.append(f"edge {edge.edge_id!r} target {edge.target_id!r} does not resolve")
        return

    # `nodes` is built from NodeRecord values; the guards above make these attributes safe.
    if source.node_type not in RELATION_DOMAINS[edge.relation_type]:  # type: ignore[attr-defined]
        errors.append(
            f"edge {edge.edge_id!r} relation {edge.relation_type.value!r} "
            f"cannot start at {source.node_type.value!r}"  # type: ignore[attr-defined]
        )
    if target.node_type not in RELATION_RANGES[edge.relation_type]:  # type: ignore[attr-defined]
        errors.append(
            f"edge {edge.edge_id!r} relation {edge.relation_type.value!r} "
            f"cannot end at {target.node_type.value!r}"  # type: ignore[attr-defined]
        )

    _validate_relation_time(edge, source, target, errors, warnings)


def _validate_relation_time(edge: EdgeRecord, source: object, target: object, errors: list[str], warnings: list[str]) -> None:
    source_date = getattr(source, "decision_date")
    target_date = getattr(target, "decision_date")
    if edge.relation_type is RelationType.TEMPORALLY_PRECEDES and source_date and target_date:
        if source_date >= target_date:
            errors.append(f"edge {edge.edge_id!r} violates temporal-precedes ordering")
    if edge.relation_type is RelationType.OVERRULES and source_date and target_date:
        if source_date <= target_date:
            errors.append(f"edge {edge.edge_id!r} overrules a same-or-later decision")
    if edge.relation_type is RelationType.CITES and source_date and target_date and source_date < target_date:
        warnings.append(f"edge {edge.edge_id!r} cites a later decision; check source dates or extraction")
