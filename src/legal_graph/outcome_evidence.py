"""BM25 statute evidence for the flat-text retrieval rung (no graph structure).

The index is rebuilt in memory from the pinned provision corpus, so the evidence is a
pure function of the corpus bytes. Provisions of Acts not yet in force on the query's
decision date are never returned.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .bm25 import build_bm25_index, corpus_sha256, read_provision_corpus, search_bm25
from .outcome_examples import OutcomeExample
from .statutes import KNOWN_STATUTES

EVIDENCE_HEADER = "Relevant statutory provisions:\n"
_ENACTED = {spec.act_id: spec.enactment_date for spec in KNOWN_STATUTES}


class Bm25Evidence:
    def __init__(self, provisions_path: Path, top_k: int, max_chars_per_provision: int = 500) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        documents = read_provision_corpus(provisions_path)
        self.corpus_sha256 = corpus_sha256(provisions_path)
        self.index = build_bm25_index(documents, corpus_digest=self.corpus_sha256, index_version="bm25-outcome-evidence")
        self.text = {document.node_id: document.text for document in documents}
        self.top_k = top_k
        self.max_chars = max_chars_per_provision

    def provisions(self, view_text: str, decision_date: date | None) -> list[str]:
        """Node ids of the top-k provisions in force on the decision date."""
        chosen: list[str] = []
        for result in search_bm25(self.index, view_text, limit=self.top_k * 20):
            enacted = _ENACTED.get(result.metadata.get("act_id", ""))
            if decision_date is not None and enacted is not None and enacted > decision_date:
                continue
            chosen.append(result.node_id)
            if len(chosen) == self.top_k:
                break
        return chosen

    def __call__(self, example: OutcomeExample) -> str:
        lines = []
        for node_id in self.provisions(example.view_text, example.as_of):
            body = " ".join(self.text[node_id].split())[: self.max_chars]
            act = node_id.split(":")[2].upper()
            lines.append(f"- {act}: {body}")
        return EVIDENCE_HEADER + "\n".join(lines) + "\n\n" if lines else ""
