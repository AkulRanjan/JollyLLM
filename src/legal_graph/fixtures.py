"""Small synthetic records for deterministic local tests and demos."""

from __future__ import annotations

from datetime import date

from .models import EdgeRecord, GraphSnapshot, NodeRecord, NodeType, RelationType, Split


def fixture_graph() -> GraphSnapshot:
    """Return a valid graph with temporal and split-safety edge cases."""

    nodes = (
        NodeRecord(
            node_id="case:india:sc:alpha-2020",
            node_type=NodeType.CASE,
            label="Alpha v State",
            source_document_ids=("doc:india:sc:alpha-2020",),
            decision_date=date(2020, 5, 10),
        ),
        NodeRecord(
            node_id="case:india:sc:beta-2022",
            node_type=NodeType.CASE,
            label="Beta v State",
            source_document_ids=("doc:india:sc:beta-2022",),
            decision_date=date(2022, 7, 12),
        ),
        NodeRecord(
            node_id="case:india:sc:query-2025",
            node_type=NodeType.CASE,
            label="Query v State",
            source_document_ids=("doc:india:sc:query-2025",),
            decision_date=date(2025, 2, 1),
        ),
        NodeRecord(
            node_id="act:india:penal-code",
            node_type=NodeType.ACT,
            label="Indian Penal Code",
            source_document_ids=("doc:india:statute:ipc",),
            available_from=date(1860, 1, 1),
        ),
        NodeRecord(
            node_id="provision:india:ipc:302:v1",
            node_type=NodeType.PROVISION,
            label="IPC section 302 (historic version)",
            source_document_ids=("doc:india:statute:ipc",),
            available_from=date(1860, 1, 1),
            available_to=date(2023, 6, 30),
        ),
        NodeRecord(
            node_id="provision:india:bns:103:v1",
            node_type=NodeType.PROVISION,
            label="BNS section 103",
            source_document_ids=("doc:india:statute:bns",),
            available_from=date(2024, 7, 1),
        ),
        NodeRecord(
            node_id="court:india:supreme-court",
            node_type=NodeType.COURT,
            label="Supreme Court of India",
            source_document_ids=("doc:india:courts",),
        ),
    )
    edges = (
        EdgeRecord(
            edge_id="edge:cites:beta-alpha",
            relation_type=RelationType.CITES,
            source_id="case:india:sc:beta-2022",
            target_id="case:india:sc:alpha-2020",
            evidence_extraction_ids=("extract:beta:1",),
            origin_split=Split.TRAIN,
            source_decision_date=date(2022, 7, 12),
        ),
        EdgeRecord(
            edge_id="edge:overrules:beta-alpha",
            relation_type=RelationType.OVERRULES,
            source_id="case:india:sc:beta-2022",
            target_id="case:india:sc:alpha-2020",
            evidence_extraction_ids=("extract:beta:2",),
            origin_split=Split.TRAIN,
            source_decision_date=date(2022, 7, 12),
        ),
        EdgeRecord(
            edge_id="edge:refers:beta-ipc302",
            relation_type=RelationType.REFERS_TO_PROVISION,
            source_id="case:india:sc:beta-2022",
            target_id="provision:india:ipc:302:v1",
            evidence_extraction_ids=("extract:beta:3",),
            origin_split=Split.TRAIN,
            source_decision_date=date(2022, 7, 12),
        ),
        EdgeRecord(
            edge_id="edge:contains:ipc-ipc302",
            relation_type=RelationType.CONTAINS,
            source_id="act:india:penal-code",
            target_id="provision:india:ipc:302:v1",
            evidence_extraction_ids=("extract:ipc:1",),
            origin_split=Split.TRAIN,
        ),
        EdgeRecord(
            edge_id="edge:amends:bns-ipc",
            relation_type=RelationType.AMENDS,
            source_id="provision:india:bns:103:v1",
            target_id="provision:india:ipc:302:v1",
            evidence_extraction_ids=("extract:bns:1",),
            origin_split=Split.TRAIN,
            available_from=date(2024, 7, 1),
        ),
        EdgeRecord(
            edge_id="edge:decided-by:beta-sc",
            relation_type=RelationType.DECIDED_BY,
            source_id="case:india:sc:beta-2022",
            target_id="court:india:supreme-court",
            evidence_extraction_ids=("extract:beta:4",),
            origin_split=Split.TRAIN,
        ),
        EdgeRecord(
            edge_id="edge:temporal:alpha-beta",
            relation_type=RelationType.TEMPORALLY_PRECEDES,
            source_id="case:india:sc:alpha-2020",
            target_id="case:india:sc:beta-2022",
            evidence_extraction_ids=("extract:dates:1",),
            origin_split=Split.TRAIN,
        ),
        # This edge demonstrates that a test-origin source cannot enter a train-safe retrieval result.
        EdgeRecord(
            edge_id="edge:cites:query-beta-test",
            relation_type=RelationType.CITES,
            source_id="case:india:sc:query-2025",
            target_id="case:india:sc:beta-2022",
            evidence_extraction_ids=("extract:query:1",),
            origin_split=Split.TEST,
            source_decision_date=date(2025, 2, 1),
        ),
    )
    return GraphSnapshot(
        version="graph-fixture-0.1.0",
        schema_version="1.0.0",
        nodes=nodes,
        edges=edges,
    )
