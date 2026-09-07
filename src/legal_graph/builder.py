"""Turn resolved citation candidates into provenance-backed graph edges."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import Iterable

from .extraction import ResolutionResult
from .models import EdgeRecord, Split


@dataclass(frozen=True, slots=True)
class GraphAssemblyResult:
    """Resolved edges and the unresolved evidence that must be reviewed or reported."""

    edges: tuple[EdgeRecord, ...]
    unresolved: tuple[ResolutionResult, ...]


def build_reference_edges(
    source_node_id: str,
    resolutions: Iterable[ResolutionResult],
    *,
    origin_split: Split,
    source_decision_date: date | None = None,
) -> GraphAssemblyResult:
    """Create deterministic edges, grouping repeated references by target and relation.

    Unresolved or ambiguous records remain in the returned result and are never
    converted into anonymous edges. Callers persist them in their review queue
    and include their counts in extraction-quality reports.
    """

    grouped_evidence: dict[tuple[str, str], list[str]] = defaultdict(list)
    unresolved: list[ResolutionResult] = []

    for resolution in resolutions:
        if resolution.resolved_target_id is None:
            unresolved.append(resolution)
            continue
        key = (resolution.candidate.relation_type.value, resolution.resolved_target_id)
        grouped_evidence[key].append(resolution.candidate.extraction_id)

    edges: list[EdgeRecord] = []
    for (relation_value, target_id), evidence_ids in sorted(grouped_evidence.items()):
        evidence = tuple(sorted(set(evidence_ids)))
        edge_identity = "|".join((relation_value, source_node_id, target_id, *evidence))
        edge_id = "edge:" + sha256(edge_identity.encode("utf-8")).hexdigest()[:20]
        from .models import RelationType  # avoids a cyclic import at module-load time

        edges.append(
            EdgeRecord(
                edge_id=edge_id,
                relation_type=RelationType(relation_value),
                source_id=source_node_id,
                target_id=target_id,
                evidence_extraction_ids=evidence,
                origin_split=origin_split,
                source_decision_date=source_decision_date,
            )
        )

    return GraphAssemblyResult(edges=tuple(edges), unresolved=tuple(unresolved))
