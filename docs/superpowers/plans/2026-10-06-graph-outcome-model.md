# Graph-Augmented Outcome Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, test, and smoke-run the trainable graph-augmented outcome classifier (text-only control, untyped graph, typed R-GCN, random-graph control) so the projected variants in `papers2/` can be measured on Kaggle under the pre-registered protocol.

**Architecture:** Pure-Python modules select a split- and time-safe query subgraph (budgeted expansion or a greedy PCST), map it to index tensors, and load leakage-controlled outcome examples. A small PyTorch R-GCN encoder pools the subgraph into K soft tokens that are prepended to the prompt embeddings of a 4-bit, LoRA-adapted causal LM. Training supervises only the verbalised label tokens, and the label is chosen by scoring the four candidate continuations. A freeze-then-evaluate-once step and a paired bootstrap complete the protocol.

**Tech Stack:** Python ≥ 3.11, unittest, PyTorch, transformers, peft, bitsandbytes (CUDA only), Kaggle CLI 2.x. Pure-Python modules have no third-party dependency.

**Spec:** `papers2/paper.html`, Sections 5–6 (retrieval, model) and Section 10 (protocol). Decisions recorded in the 2026-10-06 conversation: Kaggle is driven through the user's CLI token (account `akxran`); the journal datasets are SC-2016 + ILDC + IL-TUR; the GPU budget is 8B headline runs plus small-model seeds.

## Roadmap (this plan is Plan 1 of 3)

1. **Plan 1 (this file): model stack.** Everything needed to train and evaluate every variant on SC-2016, verified locally on the RTX 3060 and once on Kaggle.
2. **Plan 2: journal data.** Multi-year Supreme Court corpus from the AWS Open Data bucket `s3://indian-supreme-court-judgments` (CC-BY-4.0, no AWS account needed); IPC 1860, Indian Evidence Act 1872 and the Constitution in the statute layer; loaders for IL-TUR CJPE (ILDC; HF dataset `Exploration-Lab/IL-TUR`, gated, CC-BY-NC-SA-4.0) and LSI/PCR; a binary label schema for CJPE beside the four-label SC-2016 schema.
3. **Plan 3: experiment campaign.** The Kaggle run ledger and quota scheduling; Qwen3-8B and SaulLM-7B headline runs (1 seed each); Qwen3-4B or 1.7B 3-seed ablations; an InLegalBERT baseline; the frozen test evaluation and the forecast comparison.

Plan 2 and Plan 3 are written after Plan 1 ships, because their task boundaries depend on the interfaces fixed here.

## Global Constraints

- Python `>=3.11`; tests use `unittest` and run with `py -3 -m unittest discover -s tests -v` and `PYTHONPATH=src`.
- Modules that need torch must import it lazily. Their tests use `@unittest.skipUnless(_has("torch"), ...)` so the pure suite still passes without ML packages.
- Retrieval for every query uses `allowed_origin_splits = {train}` only, `as_of_date = query decision date`, and excludes the query's own case node.
- Query seeds come only from the masked 6,000-character input view (`build_input_view`), never from the full judgment.
- Test labels are loadable only through `load_outcome_examples(..., allow_test_labels=True)`, which only `evaluate_frozen` calls.
- Trainable parameters (LoRA + encoder + projector) must be `<= 0.02` of total parameters; training aborts otherwise.
- Every run directory is write-once: refuse a non-empty output directory, as `training.py` does.
- Label schema and evaluator are reused unchanged: `legal_graph.outcomes.LABELS` and `legal_graph.outcomes.evaluate`.
- Nothing is downloaded or pushed without the configured `local_files_only`/CLI step the user approved. Kaggle kernels are always private.

## Review Focus

1. **A query whose view cites no resolvable provision.** This is the common case: only 75 of 589 cases reach a statute. It must yield an empty subgraph and the learned null tokens, never a crash or a silent fallback to the full graph. Pinned in Task 1 and Task 3.
2. **A 2016 query seeded with a BNS/BNSS provision**, which is only valid from 2023-12-25. That seed must be dropped by the time filter. Pinned in Task 1.
3. **A test-split case reaching training code.** Its label must be `None`, and training must refuse unlabelled examples. Pinned in Task 4 and Task 6.
4. **Re-running the test evaluation after a change to the adapter or encoder.** Freeze hashes must mismatch and the evaluation must be refused. Pinned in Task 7.
5. **A multi-token verbaliser ("partly allowed") versus single-token ones.** Scoring must sum over all tokens of each candidate, and ties must break deterministically. Pinned in Task 5.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/legal_graph/query_graph.py` | Seeds from the masked view; split- and time-safe query subgraph; index mapping (pure Python) |
| `src/legal_graph/pcst.py` | Greedy prize-collecting Steiner tree selector `pcst_greedy_v1` (pure Python) |
| `src/legal_graph/graph_encoder.py` | `RGCNLayer`, `GraphPrefixEncoder`, node-feature embedding (torch) |
| `src/legal_graph/outcome_examples.py` | Leakage-controlled `OutcomeExample` loading per split (pure Python) |
| `src/legal_graph/outcome_prompt.py` | Instruction template, verbalisers, candidate scoring helpers |
| `src/legal_graph/outcome_training.py` | Config, model assembly, training loop, validation, run manifest |
| `src/legal_graph/outcome_eval.py` | Freeze manifest, evaluate-once on test, graph-token ablation |
| `src/legal_graph/comparison.py` | Paired bootstrap, seed aggregation, forecast comparison (pure Python) |
| `src/legal_graph/cli.py` | New subcommands `train-outcome`, `freeze-outcome`, `evaluate-outcome`, `compare-outcomes` |
| `configs/training/outcome/*.json` | One config per variant and model |
| `scripts/kaggle_run.py` | Push a private GPU kernel, poll status, pull outputs |
| `scripts/build_kaggle_notebook.py` | Gains outcome `RUN` options |
| `tests/test_query_graph.py`, `test_pcst.py`, `test_graph_encoder.py`, `test_outcome_examples.py`, `test_outcome_prompt.py`, `test_outcome_training.py`, `test_outcome_eval.py`, `test_comparison.py`, `test_kaggle_run.py` | One test module per source module |

Shared test helper (put at the top of each torch-dependent test module):

```python
import importlib.util
def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
```

---

### Task 0: Local ML environment and smoke model

**Files:**
- Create: `requirements-ml.txt`
- Modify: `.gitignore` (ensure `.venv-ml/` is ignored)

**Interfaces:**
- Produces: a Python 3.13 virtual environment `.venv-ml` with CUDA torch, used by every torch task; local weights at `models/qwen2.5-0.5b-instruct`.

- [ ] **Step 1: Write the requirements file**

```text
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.6.0
transformers>=4.51,<5
peft>=0.15
accelerate>=1.6
bitsandbytes>=0.45
huggingface_hub>=0.30
pypdf>=6.0
```

- [ ] **Step 2: Create the environment** (asks the user before installing, about 3 GB)

```powershell
py -3.13 -m venv .venv-ml
.venv-ml\Scripts\python -m pip install -r requirements-ml.txt
.venv-ml\Scripts\python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.6.0+cu124 True NVIDIA GeForce RTX 3060 Laptop GPU`

- [ ] **Step 3: Download the smoke model at a pinned revision** (asks the user first, about 1 GB)

```powershell
.venv-ml\Scripts\python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/qwen2.5-0.5b-instruct'))"
```

Record the resolved commit hash printed in `models/qwen2.5-0.5b-instruct/.cache` into the smoke config in Task 8.

- [ ] **Step 4: Confirm that the existing suite passes in the new environment**

Run: `$env:PYTHONPATH="src"; .venv-ml\Scripts\python -m unittest discover -s tests -v`
Expected: all 64 existing tests pass, including the torch collator tests, which skip today.

- [ ] **Step 5: Commit**

```bash
git add requirements-ml.txt .gitignore
git commit -m "chore: pin local CUDA ML environment for outcome models"
```

---

### Task 1: Split- and time-safe query subgraph

**Files:**
- Create: `src/legal_graph/query_graph.py`
- Test: `tests/test_query_graph.py`

**Interfaces:**
- Consumes: `extract_indian_references(text, document_id)` and `resolve_candidates(candidates, known_target_ids)` from `extraction.py`; `retrieve_subgraph`, `RetrievalRequest`, `RetrievalConfig`, `GraphSnapshot`, `RelationType`, `Split` from existing modules.
- Produces:
  - `RELATIONS: tuple[RelationType, ...]` (fixed order = `tuple(RelationType)`)
  - `NUM_RELATION_TYPES: int` (= `2 * len(RELATIONS)`; inverse edges use `index + len(RELATIONS)`)
  - `@dataclass(frozen=True) QuerySubgraph(query_id: str, node_ids: tuple[str, ...], edges: tuple[tuple[int, int, int], ...], seed_node_ids: tuple[str, ...], status: str)`; `status` is `"ok"` or `"empty"`
  - `view_seed_node_ids(view_text: str, document_id: str, graph: GraphSnapshot) -> tuple[str, ...]`
  - `build_query_subgraph(graph: GraphSnapshot, *, query_id: str, document_id: str, view_text: str, as_of_date: date | None, exclude_node_ids: frozenset[str], config: RetrievalConfig, selector: str = "expansion", random_seed: int | None = None) -> QuerySubgraph`; `selector` is `"expansion"`, `"pcst"`, or `"random"`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from datetime import date
import unittest

from legal_graph.models import EdgeRecord, GraphSnapshot, NodeRecord, NodeType, RelationType, RetrievalConfig, Split
from legal_graph.query_graph import NUM_RELATION_TYPES, RELATIONS, build_query_subgraph, view_seed_node_ids


def _graph() -> GraphSnapshot:
    crpc = NodeRecord("act:india:crpc", NodeType.ACT, "CrPC", ("d",), available_from=date(1974, 4, 1))
    bns = NodeRecord("act:india:bns", NodeType.ACT, "BNS", ("d",), available_from=date(2023, 12, 25))
    s482 = NodeRecord("provision:india:crpc:482:sha256-x", NodeType.PROVISION, "CrPC section 482", ("d",), available_from=date(1974, 4, 1))
    s103 = NodeRecord("provision:india:bns:103:sha256-y", NodeType.PROVISION, "BNS section 103", ("d",), available_from=date(2023, 12, 25))
    train_case = NodeRecord("case:train", NodeType.CASE, "A v B", ("d",), decision_date=date(2016, 1, 5))
    query_case = NodeRecord("case:query", NodeType.CASE, "C v D", ("d",), decision_date=date(2016, 6, 1))
    test_case = NodeRecord("case:test", NodeType.CASE, "E v F", ("d",), decision_date=date(2016, 2, 1))
    later_case = NodeRecord("case:later", NodeType.CASE, "G v H", ("d",), decision_date=date(2016, 9, 1))
    edges = (
        EdgeRecord("e1", RelationType.CONTAINS, "act:india:crpc", s482.node_id, ("x",), Split.TRAIN),
        EdgeRecord("e2", RelationType.CONTAINS, "act:india:bns", s103.node_id, ("x",), Split.TRAIN),
        EdgeRecord("e3", RelationType.REFERS_TO_PROVISION, "case:train", s482.node_id, ("x",), Split.TRAIN, source_decision_date=date(2016, 1, 5)),
        EdgeRecord("e4", RelationType.REFERS_TO_PROVISION, "case:query", s482.node_id, ("x",), Split.VALIDATION, source_decision_date=date(2016, 6, 1)),
        EdgeRecord("e5", RelationType.REFERS_TO_PROVISION, "case:test", s482.node_id, ("x",), Split.TEST, source_decision_date=date(2016, 2, 1)),
        EdgeRecord("e6", RelationType.REFERS_TO_PROVISION, "case:later", s482.node_id, ("x",), Split.TRAIN, source_decision_date=date(2016, 9, 1)),
    )
    return GraphSnapshot("g", "1.0.0", (crpc, bns, s482, s103, train_case, query_case, test_case, later_case), edges)


VIEW = "The petition under Section 482 of the CrPC and s. 103 BNS was filed."


class SeedTests(unittest.TestCase):
    def test_seeds_resolve_from_view_text(self) -> None:
        seeds = view_seed_node_ids(VIEW, "doc:query", _graph())
        self.assertEqual(seeds, ("provision:india:crpc:482:sha256-x", "provision:india:bns:103:sha256-y"))

    def test_view_without_citations_has_no_seeds(self) -> None:
        self.assertEqual(view_seed_node_ids("No statute is named here.", "doc:q", _graph()), ())


class SubgraphTests(unittest.TestCase):
    def _build(self, view: str = VIEW, selector: str = "expansion"):
        return build_query_subgraph(
            _graph(), query_id="q", document_id="doc:query", view_text=view,
            as_of_date=date(2016, 6, 1), exclude_node_ids=frozenset({"case:query"}),
            config=RetrievalConfig(node_budget=10, expansion_hops=2), selector=selector, random_seed=3,
        )

    def test_only_train_split_and_past_nodes_are_reachable(self) -> None:
        sub = self._build()
        self.assertEqual(sub.status, "ok")
        self.assertIn("case:train", sub.node_ids)
        self.assertNotIn("case:query", sub.node_ids)   # own node excluded
        self.assertNotIn("case:test", sub.node_ids)    # test-split edge excluded
        self.assertNotIn("case:later", sub.node_ids)   # decided after the query
        self.assertNotIn("provision:india:bns:103:sha256-y", sub.node_ids)  # not in force in 2016

    def test_edges_are_index_triples_with_inverse_relations(self) -> None:
        sub = self._build()
        forward = RELATIONS.index(RelationType.REFERS_TO_PROVISION)
        types = {rel for _, _, rel in sub.edges}
        self.assertIn(forward, types)
        self.assertIn(forward + len(RELATIONS), types)
        self.assertTrue(all(0 <= rel < NUM_RELATION_TYPES for rel in types))
        self.assertTrue(all(0 <= s < len(sub.node_ids) and 0 <= t < len(sub.node_ids) for s, t, _ in sub.edges))

    def test_no_citation_gives_empty_subgraph(self) -> None:
        sub = self._build(view="Nothing to resolve.")
        self.assertEqual((sub.status, sub.node_ids, sub.edges), ("empty", (), ()))

    def test_random_selector_matches_size_and_is_seeded(self) -> None:
        a, b = self._build(selector="random"), self._build(selector="random")
        self.assertEqual(a.node_ids, b.node_ids)
        self.assertEqual(len(a.node_ids), len(self._build().node_ids))
        self.assertNotIn("case:test", a.node_ids)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_query_graph -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legal_graph.query_graph'`

- [ ] **Step 3: Implement**

```python
"""Split- and time-safe query subgraphs for the graph-augmented outcome model.

Seeds come only from the masked input view. Retrieval sees training-split edges only,
nothing decided or in force after the query's decision date, and never the query's
own case node, whose edges were extracted from the full judgment text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from .extraction import extract_indian_references, resolve_candidates
from .models import GraphSnapshot, RelationType, RetrievalConfig, RetrievalRequest, Split
from .retrieval import retrieve_subgraph, _edge_is_eligible, _node_is_eligible

RELATIONS: tuple[RelationType, ...] = tuple(RelationType)
NUM_RELATION_TYPES = 2 * len(RELATIONS)
_ALLOWED = frozenset({Split.TRAIN})


@dataclass(frozen=True, slots=True)
class QuerySubgraph:
    query_id: str
    node_ids: tuple[str, ...]
    edges: tuple[tuple[int, int, int], ...]
    seed_node_ids: tuple[str, ...]
    status: str


def view_seed_node_ids(view_text: str, document_id: str, graph: GraphSnapshot) -> tuple[str, ...]:
    candidates = extract_indian_references(view_text, document_id)
    results = resolve_candidates(candidates, (node.node_id for node in graph.nodes))
    seen: list[str] = []
    for result in results:
        if result.resolved_target_id and result.resolved_target_id not in seen:
            seen.append(result.resolved_target_id)
    return tuple(seen)


def build_query_subgraph(
    graph: GraphSnapshot,
    *,
    query_id: str,
    document_id: str,
    view_text: str,
    as_of_date: date | None,
    exclude_node_ids: frozenset[str],
    config: RetrievalConfig,
    selector: str = "expansion",
    random_seed: int | None = None,
) -> QuerySubgraph:
    if selector not in {"expansion", "pcst", "random"}:
        raise ValueError(f"unknown selector {selector!r}")
    view = _without(graph, exclude_node_ids)
    nodes = view.nodes_by_id
    seeds = tuple(
        seed for seed in view_seed_node_ids(view_text, document_id, view)
        if seed in nodes and _node_is_eligible(nodes[seed], as_of_date)
    )
    if not seeds:
        return QuerySubgraph(query_id, (), (), (), "empty")
    request = RetrievalRequest(
        query_id=query_id,
        seed_node_ids=seeds,
        node_prizes={seed: 1.0 for seed in seeds},
        as_of_date=as_of_date,
        allowed_origin_splits=_ALLOWED,
    )
    eligible_edges = tuple(edge for edge in view.edges if _edge_is_eligible(edge, nodes, request))
    selected = _expand(view, request, config, seeds)
    if selector == "pcst":
        from .pcst import select_pcst

        selected = select_pcst(seeds, eligible_edges, nodes, budget=config.node_budget, max_hops=config.expansion_hops)
    elif selector == "random":
        selected = _random_nodes(view, as_of_date, len(selected), seeds[0], random_seed if random_seed is not None else 0)
    index = {node_id: position for position, node_id in enumerate(selected)}
    triples: list[tuple[int, int, int]] = []
    for edge in eligible_edges:
        if edge.source_id in index and edge.target_id in index:
            rel = RELATIONS.index(edge.relation_type)
            triples.append((index[edge.source_id], index[edge.target_id], rel))
            triples.append((index[edge.target_id], index[edge.source_id], rel + len(RELATIONS)))
    return QuerySubgraph(query_id, tuple(selected), tuple(sorted(set(triples))), seeds, "ok")


def _without(graph: GraphSnapshot, excluded: frozenset[str]) -> GraphSnapshot:
    if not excluded:
        return graph
    return GraphSnapshot(
        graph.version,
        graph.schema_version,
        tuple(node for node in graph.nodes if node.node_id not in excluded),
        tuple(edge for edge in graph.edges if edge.source_id not in excluded and edge.target_id not in excluded),
    )


def _expand(graph: GraphSnapshot, request: RetrievalRequest, config: RetrievalConfig, seeds: tuple[str, ...]) -> list[str]:
    """Union of per-seed connected expansions, sharing the node budget across seeds."""
    per_seed = max(1, config.node_budget // len(seeds))
    selected: list[str] = []
    for seed in seeds:
        single = RetrievalRequest(
            query_id=request.query_id, seed_node_ids=(seed,), node_prizes=request.node_prizes,
            as_of_date=request.as_of_date, allowed_origin_splits=request.allowed_origin_splits,
        )
        bundle = retrieve_subgraph(graph, single, RetrievalConfig(node_budget=per_seed, expansion_hops=config.expansion_hops))
        for node_id in bundle.selected_node_ids or (seed,):
            if node_id not in selected and len(selected) < config.node_budget:
                selected.append(node_id)
    return selected


def _random_nodes(graph: GraphSnapshot, as_of_date: date | None, count: int, query_key: str, seed: int) -> list[str]:
    """Random-retrieval control: same node count, eligible nodes, no relevance signal."""
    reachable = {edge.source_id for edge in graph.edges if edge.origin_split in _ALLOWED}
    reachable |= {edge.target_id for edge in graph.edges if edge.origin_split in _ALLOWED}
    pool = sorted(
        node.node_id for node in graph.nodes
        if node.node_id in reachable and _node_is_eligible(node, as_of_date)
    )
    generator = random.Random(f"{seed}|{query_key}")
    return generator.sample(pool, min(count, len(pool)))
```

- [ ] **Step 4: Run the tests to verify they pass** (the `pcst` path is exercised in Task 2)

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_query_graph -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/query_graph.py tests/test_query_graph.py
git commit -m "feat: split- and time-safe query subgraphs seeded from the masked view"
```

---

### Task 2: Greedy prize-collecting Steiner tree selector

**Files:**
- Create: `src/legal_graph/pcst.py`
- Test: `tests/test_pcst.py`

**Interfaces:**
- Consumes: `EdgeRecord`, `NodeRecord`.
- Produces: `select_pcst(seeds: tuple[str, ...], edges: tuple[EdgeRecord, ...], nodes: Mapping[str, NodeRecord], *, budget: int, max_hops: int, edge_cost: float = 0.1, neighbour_prize: float = 0.25) -> list[str]`, which returns a connected node list starting at `seeds[0]`. Strategy name: `pcst_greedy_v1`.

The algorithm, which the paper must describe as a greedy approximation and not an exact PCST: start the tree at `seeds[0]`. Repeat: among the seeds not yet in the tree, find the shortest path (in hops, at most `2 * max_hops`) from the tree to each, and attach the one with the best net gain `prize(seed) - edge_cost * path_length` while it is positive and the budget allows. Then add single neighbours with a positive net gain, where `prize(v) = neighbour_prize * (number of seeds adjacent to v)`, best first, until the budget is reached. Finally, repeatedly remove leaves that are not seeds and whose prize is below `edge_cost`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import unittest

from legal_graph.models import EdgeRecord, NodeRecord, NodeType, RelationType, Split
from legal_graph.pcst import select_pcst


def _n(node_id: str) -> NodeRecord:
    return NodeRecord(node_id, NodeType.PROVISION, node_id, ("d",))


def _e(i: int, a: str, b: str) -> EdgeRecord:
    return EdgeRecord(f"e{i}", RelationType.REFERS_TO_PROVISION, a, b, ("x",), Split.TRAIN)


NODES = {k: _n(k) for k in ("s1", "s2", "hub", "c1", "c2", "c3", "far")}
EDGES = (_e(1, "c1", "s1"), _e(2, "c1", "s2"), _e(3, "hub", "s1"), _e(4, "c2", "s1"), _e(5, "c3", "far"))


class PcstTests(unittest.TestCase):
    def test_connects_seeds_through_shared_case(self) -> None:
        tree = select_pcst(("s1", "s2"), EDGES, NODES, budget=10, max_hops=2)
        self.assertEqual(tree[0], "s1")
        self.assertTrue({"s1", "s2", "c1"} <= set(tree))

    def test_prunes_nodes_without_prize(self) -> None:
        tree = select_pcst(("s1", "s2"), EDGES, NODES, budget=10, max_hops=2)
        self.assertNotIn("far", tree)
        self.assertNotIn("c3", tree)

    def test_respects_budget_and_is_deterministic(self) -> None:
        a = select_pcst(("s1", "s2"), EDGES, NODES, budget=2, max_hops=2)
        b = select_pcst(("s1", "s2"), EDGES, NODES, budget=2, max_hops=2)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 2)

    def test_single_isolated_seed(self) -> None:
        self.assertEqual(select_pcst(("hub",), (), NODES, budget=5, max_hops=2), ["hub"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_pcst -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Greedy prize-collecting Steiner tree selection (pcst_greedy_v1).

An approximation, not an exact PCST solver: seeds are joined to the growing tree along
shortest paths while their prize exceeds the path cost, profitable neighbours are
added best-first, and unprofitable non-seed leaves are pruned.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Mapping

from .models import EdgeRecord, NodeRecord


def select_pcst(
    seeds: tuple[str, ...],
    edges: tuple[EdgeRecord, ...],
    nodes: Mapping[str, NodeRecord],
    *,
    budget: int,
    max_hops: int,
    edge_cost: float = 0.1,
    neighbour_prize: float = 0.25,
) -> list[str]:
    if not seeds or budget < 1:
        return []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.source_id in nodes and edge.target_id in nodes:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
    seed_set = set(seeds)
    prize = defaultdict(float, {seed: 1.0 for seed in seeds})
    for node_id in adjacency:
        if node_id not in seed_set:
            prize[node_id] = neighbour_prize * len(adjacency[node_id] & seed_set)

    tree = [seeds[0]]
    in_tree = {seeds[0]}
    while len(tree) < budget:
        best: tuple[float, str, list[str]] | None = None
        for seed in sorted(seed_set - in_tree):
            path = _shortest_path(in_tree, seed, adjacency, 2 * max_hops)
            if path is None:
                continue
            new = [node for node in path if node not in in_tree]
            gain = sum(prize[node] for node in new) - edge_cost * len(new)
            if gain > 0 and len(tree) + len(new) <= budget and (best is None or gain > best[0]):
                best = (gain, seed, new)
        if best is None:
            break
        for node in best[2]:
            tree.append(node)
            in_tree.add(node)

    frontier = sorted(
        {n for node in tree for n in adjacency[node]} - in_tree,
        key=lambda node: (-(prize[node] - edge_cost), node),
    )
    for node in frontier:
        if len(tree) >= budget or prize[node] - edge_cost <= 0:
            break
        tree.append(node)
        in_tree.add(node)

    pruned = True
    while pruned:
        pruned = False
        for node in list(tree):
            degree = len(adjacency[node] & in_tree)
            if node not in seed_set and degree <= 1 and prize[node] < edge_cost:
                tree.remove(node)
                in_tree.discard(node)
                pruned = True
    return tree


def _shortest_path(sources: set[str], target: str, adjacency: Mapping[str, set[str]], limit: int) -> list[str] | None:
    parents: dict[str, str | None] = {source: None for source in sorted(sources)}
    queue = deque((source, 0) for source in sorted(sources))
    while queue:
        node, depth = queue.popleft()
        if node == target:
            path = []
            while node is not None and node not in sources:
                path.append(node)
                node = parents[node]
            return list(reversed(path))
        if depth >= limit:
            continue
        for neighbour in sorted(adjacency.get(node, ())):
            if neighbour not in parents:
                parents[neighbour] = node
                queue.append((neighbour, depth + 1))
    return None
```

- [ ] **Step 4: Add a query-graph test for the `pcst` selector to `tests/test_query_graph.py`**

```python
    def test_pcst_selector_respects_eligibility(self) -> None:
        sub = self._build(selector="pcst")
        self.assertEqual(sub.status, "ok")
        self.assertNotIn("case:test", sub.node_ids)
        self.assertNotIn("case:later", sub.node_ids)
        self.assertEqual(sub.node_ids[0], "provision:india:crpc:482:sha256-x")
```

Note: `select_pcst` receives only `eligible_edges` from `build_query_subgraph`, so it inherits the split and time filters.

- [ ] **Step 5: Run both modules**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_pcst tests.test_query_graph -v`
Expected: 11 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/legal_graph/pcst.py tests/test_pcst.py tests/test_query_graph.py
git commit -m "feat: greedy PCST subgraph selector over eligible edges"
```

---

### Task 3: R-GCN prefix encoder (torch)

**Files:**
- Create: `src/legal_graph/graph_encoder.py`
- Test: `tests/test_graph_encoder.py`

**Interfaces:**
- Consumes: `QuerySubgraph` and `NUM_RELATION_TYPES` (Task 1).
- Produces:
  - `RGCNLayer(in_dim: int, out_dim: int, num_relations: int)`, with `forward(h: Tensor[N, in], edge_index: LongTensor[2, E], edge_type: LongTensor[E]) -> Tensor[N, out]`
  - `GraphPrefixEncoder(node_dim: int, hidden_dim: int, model_dim: int, num_tokens: int, num_relations: int, typed: bool, num_layers: int = 2, output_rms: float = 0.02)`, with `forward(node_features: Tensor[N, node_dim], edge_index, edge_type, query: Tensor[model_dim]) -> Tensor[num_tokens, model_dim]`
  - `subgraph_tensors(sub: QuerySubgraph, feature_lookup: Mapping[str, Tensor], device) -> tuple[Tensor, LongTensor, LongTensor]`
  - `embed_node_texts(texts: Mapping[str, str], tokenizer, embedding: nn.Embedding, max_tokens: int, device) -> dict[str, Tensor]`

Design notes:
- Message normalisation is $1/c_{v,r}$, where $c_{v,r}$ counts the incoming edges of relation $r$ at node $v$.
- `typed=False` maps every edge type to 0 and allocates a single relation weight. That is the untyped ablation, at a matched architecture and a parameter count that is smaller only by the relation weights.
- An empty subgraph returns the learned `null_tokens`.
- Pooling uses $K$ learned queries, plus a projection of the query vector (the mean prompt embedding), attending over the node states.
- A LayerNorm then a learned scale initialised to `output_rms` keeps soft tokens at the language model's embedding magnitude.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import importlib.util
import unittest


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@unittest.skipUnless(_has("torch"), "torch not installed")
class EncoderTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from legal_graph.graph_encoder import GraphPrefixEncoder

        torch.manual_seed(0)
        self.torch = torch
        self.typed = GraphPrefixEncoder(node_dim=6, hidden_dim=8, model_dim=10, num_tokens=3, num_relations=4, typed=True)
        self.x = torch.randn(4, 6)
        self.query = torch.randn(10)
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    def test_output_shape(self) -> None:
        out = self.typed(self.x, self.edge_index, self.torch.tensor([0, 1, 2]), self.query)
        self.assertEqual(tuple(out.shape), (3, 10))

    def test_relation_types_change_typed_output(self) -> None:
        a = self.typed(self.x, self.edge_index, self.torch.tensor([0, 0, 0]), self.query)
        b = self.typed(self.x, self.edge_index, self.torch.tensor([0, 1, 2]), self.query)
        self.assertFalse(self.torch.allclose(a, b))

    def test_untyped_ignores_relation_types(self) -> None:
        from legal_graph.graph_encoder import GraphPrefixEncoder

        untyped = GraphPrefixEncoder(node_dim=6, hidden_dim=8, model_dim=10, num_tokens=3, num_relations=4, typed=False)
        a = untyped(self.x, self.edge_index, self.torch.tensor([0, 0, 0]), self.query)
        b = untyped(self.x, self.edge_index, self.torch.tensor([0, 1, 3]), self.query)
        self.assertTrue(self.torch.allclose(a, b))

    def test_empty_graph_returns_learned_null_tokens(self) -> None:
        empty = self.typed(self.torch.zeros(0, 6), self.torch.zeros(2, 0, dtype=self.torch.long), self.torch.zeros(0, dtype=self.torch.long), self.query)
        self.assertTrue(self.torch.equal(empty, self.typed.null_tokens))

    def test_gradients_reach_relation_weights(self) -> None:
        out = self.typed(self.x, self.edge_index, self.torch.tensor([0, 1, 2]), self.query)
        out.sum().backward()
        self.assertIsNotNone(self.typed.layers[0].relation_weight.grad)

    def test_subgraph_tensors_maps_indices(self) -> None:
        from legal_graph.graph_encoder import subgraph_tensors
        from legal_graph.query_graph import QuerySubgraph

        sub = QuerySubgraph("q", ("a", "b"), ((0, 1, 4), (1, 0, 15)), ("a",), "ok")
        feats = {"a": self.torch.ones(6), "b": self.torch.zeros(6)}
        x, index, types = subgraph_tensors(sub, feats, "cpu")
        self.assertEqual(tuple(x.shape), (2, 6))
        self.assertEqual(index.tolist(), [[0, 1], [1, 0]])
        self.assertEqual(types.tolist(), [4, 15])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$env:PYTHONPATH="src"; .venv-ml\Scripts\python -m unittest tests.test_graph_encoder -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legal_graph.graph_encoder'`

- [ ] **Step 3: Implement**

```python
"""Relation-aware graph encoder that turns a query subgraph into K soft tokens."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .query_graph import QuerySubgraph


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int) -> None:
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim)
        self.relation_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        nn.init.xavier_uniform_(self.relation_weight.view(num_relations * in_dim, out_dim))

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        out = self.self_loop(h)
        if edge_index.numel() == 0:
            return out
        source, target = edge_index[0], edge_index[1]
        messages = torch.bmm(h[source].unsqueeze(1), self.relation_weight[edge_type]).squeeze(1)
        num_rel = self.relation_weight.shape[0]
        key = target * num_rel + edge_type
        counts = torch.zeros(h.shape[0] * num_rel, device=h.device, dtype=h.dtype).index_add_(0, key, torch.ones_like(key, dtype=h.dtype))
        messages = messages / counts[key].unsqueeze(1)
        return out.index_add(0, target, messages)


class GraphPrefixEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        model_dim: int,
        num_tokens: int,
        num_relations: int,
        typed: bool,
        num_layers: int = 2,
        output_rms: float = 0.02,
    ) -> None:
        super().__init__()
        self.typed = typed
        relations = num_relations if typed else 1
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList(RGCNLayer(hidden_dim, hidden_dim, relations) for _ in range(num_layers))
        self.pool_queries = nn.Parameter(torch.randn(num_tokens, hidden_dim) / math.sqrt(hidden_dim))
        self.query_proj = nn.Linear(model_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.scale = nn.Parameter(torch.tensor(float(output_rms)))
        self.null_tokens = nn.Parameter(torch.randn(num_tokens, model_dim) * output_rms)

    def forward(self, node_features: Tensor, edge_index: Tensor, edge_type: Tensor, query: Tensor) -> Tensor:
        if node_features.shape[0] == 0:
            return self.null_tokens
        if not self.typed:
            edge_type = torch.zeros_like(edge_type)
        h = torch.relu(self.input_proj(node_features))
        for layer in self.layers:
            h = torch.relu(layer(h, edge_index, edge_type))
        queries = self.pool_queries + self.query_proj(query.to(h.dtype)).unsqueeze(0)
        scores = queries @ self.key_proj(h).T / math.sqrt(h.shape[-1])
        pooled = torch.softmax(scores, dim=-1) @ h
        return self.norm(self.projector(pooled)) * self.scale


def subgraph_tensors(sub: QuerySubgraph, feature_lookup: Mapping[str, Tensor], device) -> tuple[Tensor, Tensor, Tensor]:
    if not sub.node_ids:
        any_feature = next(iter(feature_lookup.values()))
        empty = torch.zeros(0, any_feature.shape[-1], device=device, dtype=any_feature.dtype)
        return empty, torch.zeros(2, 0, dtype=torch.long, device=device), torch.zeros(0, dtype=torch.long, device=device)
    x = torch.stack([feature_lookup[node_id] for node_id in sub.node_ids]).to(device)
    if sub.edges:
        index = torch.tensor([[s for s, _, _ in sub.edges], [t for _, t, _ in sub.edges]], dtype=torch.long, device=device)
        types = torch.tensor([r for _, _, r in sub.edges], dtype=torch.long, device=device)
    else:
        index = torch.zeros(2, 0, dtype=torch.long, device=device)
        types = torch.zeros(0, dtype=torch.long, device=device)
    return x, index, types


@torch.no_grad()
def embed_node_texts(texts: Mapping[str, str], tokenizer, embedding: nn.Embedding, max_tokens: int, device) -> dict[str, Tensor]:
    """Mean of the frozen LM input embeddings over each node's short descriptive text."""
    out: dict[str, Tensor] = {}
    for node_id, text in texts.items():
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"] or [tokenizer.eos_token_id]
        vectors = embedding(torch.tensor(ids, device=device))
        out[node_id] = vectors.float().mean(dim=0).cpu()
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH="src"; .venv-ml\Scripts\python -m unittest tests.test_graph_encoder -v`
Expected: 6 tests PASS (and SKIP under `py -3` without torch)

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/graph_encoder.py tests/test_graph_encoder.py
git commit -m "feat: R-GCN prefix encoder with typed/untyped modes and null tokens"
```

---

### Task 4: Leakage-controlled outcome examples

**Files:**
- Create: `src/legal_graph/outcome_examples.py`
- Test: `tests/test_outcome_examples.py`

**Interfaces:**
- Consumes: `build_input_view`, `infer_disposition`, `LABELS` (`outcomes.py`); `_sha256_file` (`training.py`).
- Produces:
  - `@dataclass(frozen=True) OutcomeExample(case_id: str, document_id: str, split: str, view_text: str, label: str | None, decision_date: date | None)`
  - `load_outcome_examples(corpus_path: Path, manifest_path: Path, split: str, character_limit: int, *, allow_test_labels: bool = False) -> tuple[OutcomeExample, ...]`
  - `node_feature_texts(graph: GraphSnapshot, provisions_path: Path, max_chars: int = 400) -> dict[str, str]`

Rules:
- `split` is one of `train`, `validation`, `test`, mapped through the manifest fields.
- For `train` and `validation`, unlabelled cases are dropped.
- For `test` with `allow_test_labels=False`, every case is returned with `label=None`. With `allow_test_labels=True`, unlabelled cases are dropped and labels are filled in.
- Case-node feature text is the case title only. Provision text is the label plus the first `max_chars` characters of the provision text. No outcome text ever enters node features.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.outcome_examples import load_outcome_examples
from legal_graph.training import _sha256_file

BODY = "Facts of the matter. " * 50


def _write(root: Path) -> tuple[Path, Path]:
    rows = [
        {"case_id": "c1", "document_id": "d1", "origin_split": "train", "decision_date": "2016-01-02",
         "text": BODY + "The appeal is allowed."},
        {"case_id": "c2", "document_id": "d2", "origin_split": "train", "decision_date": None, "text": BODY + "Ordered accordingly."},
        {"case_id": "c3", "document_id": "d3", "origin_split": "test", "decision_date": "2016-03-04",
         "text": "The appeal is dismissed early. " + BODY + "The appeal is dismissed."},
    ]
    corpus = root / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    manifest = root / "m.json"
    manifest.write_text(json.dumps({"artifact_type": "judgment_training_corpus", "status": "validated",
        "corpus_sha256": _sha256_file(corpus), "training_splits": ["train"], "validation_splits": ["validation"],
        "held_out_splits": ["test"]}), encoding="utf-8")
    return corpus, manifest


class ExampleTests(unittest.TestCase):
    def test_train_drops_unlabelled_and_masks_view(self) -> None:
        with TemporaryDirectory() as d:
            corpus, manifest = _write(Path(d))
            train = load_outcome_examples(corpus, manifest, "train", 6000)
        self.assertEqual([(e.case_id, e.label) for e in train], [("c1", "allowed")])
        self.assertNotIn("allowed", train[0].view_text.replace("outcome_phrase", ""))

    def test_test_labels_hidden_by_default(self) -> None:
        with TemporaryDirectory() as d:
            corpus, manifest = _write(Path(d))
            hidden = load_outcome_examples(corpus, manifest, "test", 6000)
            shown = load_outcome_examples(corpus, manifest, "test", 6000, allow_test_labels=True)
        self.assertEqual([e.label for e in hidden], [None])
        self.assertEqual([e.label for e in shown], ["dismissed"])
        self.assertIn("outcome_phrase", hidden[0].view_text)

    def test_rejects_corpus_hash_mismatch(self) -> None:
        with TemporaryDirectory() as d:
            corpus, manifest = _write(Path(d))
            corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_outcome_examples(corpus, manifest, "train", 6000)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_examples -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Split-aware, leakage-controlled outcome examples for the model runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .models import GraphSnapshot, NodeType
from .outcomes import LABELS, build_input_view, infer_disposition
from .training import _sha256_file

_FIELDS = {"train": "training_splits", "validation": "validation_splits", "test": "held_out_splits"}


@dataclass(frozen=True, slots=True)
class OutcomeExample:
    case_id: str
    document_id: str
    split: str
    view_text: str
    label: str | None
    decision_date: date | None


def load_outcome_examples(
    corpus_path: Path,
    manifest_path: Path,
    split: str,
    character_limit: int,
    *,
    allow_test_labels: bool = False,
) -> tuple[OutcomeExample, ...]:
    if split not in _FIELDS:
        raise ValueError(f"unknown split {split!r}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "validated" or "test" not in manifest.get("held_out_splits", []):
        raise ValueError("training manifest must be validated and hold out test")
    if manifest.get("corpus_sha256") != _sha256_file(corpus_path):
        raise ValueError("corpus checksum does not match training manifest")
    allowed = set(manifest[_FIELDS[split]])
    hide = split == "test" and not allow_test_labels
    examples: list[OutcomeExample] = []
    with corpus_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["origin_split"] not in allowed:
                continue
            text = str(row["text"])
            label = infer_disposition(text[int(len(text) * 0.8):])
            if label not in LABELS:
                label = None
            if label is None and not hide:
                continue
            raw_date = row.get("decision_date")
            examples.append(OutcomeExample(
                case_id=str(row["case_id"]),
                document_id=str(row["document_id"]),
                split=split,
                view_text=build_input_view(text, character_limit),
                label=None if hide else label,
                decision_date=date.fromisoformat(raw_date) if raw_date else None,
            ))
    return tuple(examples)


def node_feature_texts(graph: GraphSnapshot, provisions_path: Path, max_chars: int = 400) -> dict[str, str]:
    provision_text: dict[str, str] = {}
    if provisions_path.exists():
        with provisions_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                provision_text[str(row["node_id"])] = str(row.get("text", ""))[:max_chars]
    texts: dict[str, str] = {}
    for node in graph.nodes:
        if node.node_type is NodeType.PROVISION:
            texts[node.node_id] = f"{node.label}. {provision_text.get(node.node_id, '')}".strip()
        else:
            texts[node.node_id] = node.label
    return texts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_examples -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/outcome_examples.py tests/test_outcome_examples.py
git commit -m "feat: leakage-controlled outcome examples with hidden test labels"
```

---

### Task 5: Prompt, verbalisers, and candidate scoring

**Files:**
- Create: `src/legal_graph/outcome_prompt.py`
- Test: `tests/test_outcome_prompt.py`

**Interfaces:**
- Produces:
  - `INSTRUCTION: str`, `ANSWER_PREFIX: str`, `VERBALIZERS: dict[str, str]` (`" allowed"`, `" dismissed"`, `" disposed"`, `" partly allowed"`)
  - `prompt_token_ids(tokenizer, view_text: str, max_prompt_tokens: int) -> list[int]`: instruction plus the view truncated to fit, plus `ANSWER_PREFIX`
  - `label_token_ids(tokenizer) -> dict[str, tuple[int, ...]]`: non-empty, pairwise distinct
  - `continuation_log_prob(log_probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float`: sum of `log_probs[i][targets[i]]`
  - `pick_label(scores: Mapping[str, float]) -> str`: argmax, ties broken by `LABELS` order

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import unittest

from legal_graph.outcome_prompt import ANSWER_PREFIX, continuation_log_prob, label_token_ids, pick_label, prompt_token_ids


class WordTokenizer:
    """Whitespace tokenizer for deterministic tests."""

    eos_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = False, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]}


class PromptTests(unittest.TestCase):
    def test_prompt_truncates_view_but_keeps_answer_prefix(self) -> None:
        tok = WordTokenizer()
        ids = prompt_token_ids(tok, "word " * 500, max_prompt_tokens=60)
        self.assertLessEqual(len(ids), 60)
        tail = tok(ANSWER_PREFIX)["input_ids"]
        self.assertEqual(ids[-len(tail):], tail)

    def test_label_ids_are_distinct_and_multi_token_for_partly(self) -> None:
        ids = label_token_ids(WordTokenizer())
        self.assertEqual(len(set(ids.values())), 4)
        self.assertEqual(len(ids["partly_allowed"]), 2)

    def test_continuation_sums_all_tokens(self) -> None:
        log_probs = [[-1.0, -2.0], [-3.0, -0.5]]
        self.assertAlmostEqual(continuation_log_prob(log_probs, [1, 1]), -2.5)

    def test_pick_label_breaks_ties_in_schema_order(self) -> None:
        self.assertEqual(pick_label({"dismissed": -1.0, "allowed": -1.0, "disposed": -3.0, "partly_allowed": -9.0}), "allowed")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_prompt -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Prompt template, label verbalisers, and candidate scoring for the outcome task."""

from __future__ import annotations

from typing import Mapping, Sequence

from .outcomes import LABELS

INSTRUCTION = (
    "Read the opening of an Indian Supreme Court judgment. Phrases that state the final "
    "disposition have been replaced by outcome_phrase. Answer with the final outcome: "
    "allowed, dismissed, disposed, or partly allowed.\n\nJudgment:\n"
)
ANSWER_PREFIX = "\n\nOutcome:"
VERBALIZERS = {label: " " + label.replace("_", " ") for label in LABELS}


def prompt_token_ids(tokenizer, view_text: str, max_prompt_tokens: int) -> list[int]:
    head = tokenizer(INSTRUCTION, add_special_tokens=False)["input_ids"]
    tail = tokenizer(ANSWER_PREFIX, add_special_tokens=False)["input_ids"]
    body = tokenizer(view_text, add_special_tokens=False)["input_ids"]
    room = max_prompt_tokens - len(head) - len(tail)
    if room <= 0:
        raise ValueError("max_prompt_tokens leaves no room for the judgment view")
    return list(head) + list(body[:room]) + list(tail)


def label_token_ids(tokenizer) -> dict[str, tuple[int, ...]]:
    ids = {label: tuple(tokenizer(text, add_special_tokens=False)["input_ids"]) for label, text in VERBALIZERS.items()}
    if any(not value for value in ids.values()) or len(set(ids.values())) != len(ids):
        raise ValueError("label verbalisers must tokenise to non-empty, distinct sequences")
    return ids


def continuation_log_prob(log_probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    if len(log_probs) != len(targets):
        raise ValueError("one log-probability row is required per target token")
    return float(sum(row[target] for row, target in zip(log_probs, targets)))


def pick_label(scores: Mapping[str, float]) -> str:
    return max(LABELS, key=lambda label: (scores[label], -LABELS.index(label)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_prompt -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/outcome_prompt.py tests/test_outcome_prompt.py
git commit -m "feat: outcome prompt, verbalisers, and deterministic candidate scoring"
```

---

### Task 6: Training loop, validation, and run manifest

**Files:**
- Create: `src/legal_graph/outcome_training.py`
- Test: `tests/test_outcome_training.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 3, 4, 5; `evaluate` (`outcomes.py`); `_parameter_report`, `_pin_base_model`, `_sha256_file` (`training.py`).
- Produces:
  - `@dataclass(frozen=True) OutcomeTrainConfig` with fields: `experiment_id, variant, selector, model_source, model_revision, local_model_path, corpus_path, training_data_manifest, graph_path, provisions_path, output_dir, input_character_limit, max_prompt_tokens, epochs, learning_rate, encoder_learning_rate, gradient_accumulation, seed, quantize_4bit, min_cuda_memory_bytes, lora_rank, lora_alpha, lora_dropout, lora_target_modules, num_graph_tokens, graph_hidden_dim, graph_layers, node_budget, expansion_hops, node_feature_tokens, max_train_examples`
  - `VARIANTS = ("text", "graph_untyped", "graph_typed", "graph_random")`
  - `load_outcome_config(path: Path) -> OutcomeTrainConfig` (rejects unknown variant or selector; `variant == "graph_random"` forces `selector="random"`)
  - `class OutcomeModel` with `.prefix(example, sub) -> Tensor | None`, `.loss(example, sub) -> Tensor`, `.score(example, sub, ablate_graph=False) -> dict[str, float]`
  - `train_outcome(config, workspace_root, *, model=None, tokenizer=None) -> dict`. Injecting a tiny `model`/`tokenizer` is how the CPU test runs without downloads.
  - Run directory layout: `adapter/`, `encoder.pt` (graph variants only), `config.json`, `validation-predictions.jsonl`, `validation-metrics.json`, `subgraph-stats.json`, `run.manifest.json`

Behaviour:
- Training examples come from `load_outcome_examples(split="train")`. The loop raises if any `label is None`.
- Each step builds the subgraph through `build_query_subgraph(... exclude_node_ids={example.case_id}, as_of_date=example.decision_date ...)`.
- `inputs_embeds = cat([Z, embed(prompt_ids + label_ids + [eos])])`, with `labels = [-100] * (K + len(prompt_ids)) + label_ids + [eos]`.
- The text variant uses `Z = None` (no prefix positions).
- The optimiser is AdamW with two parameter groups: LoRA at `learning_rate`, encoder at `encoder_learning_rate`.
- After each epoch: score validation, compute `evaluate()`, and keep the best checkpoint by validation macro-F1. Ties go to the earlier epoch.
- `run.manifest.json` records: `status="trained_not_frozen"`, config SHA-256, corpus, manifest and graph hashes, a parameter report that includes encoder parameters, best epoch, validation metrics, subgraph coverage (share of examples with `status == "ok"`), CUDA device, and versions.
- Training aborts if the trainable fraction exceeds 0.02.

- [ ] **Step 1: Write the failing CPU test with a tiny randomly initialised Llama**

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _has(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class CharTokenizer:
    eos_token_id = 1
    pad_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None, **_: object):
        ids = [2 + (ord(ch) % 120) for ch in text]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}

    def save_pretrained(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


@unittest.skipUnless(_has("torch") and _has("transformers") and _has("peft"), "ML stack not installed")
class TinyTrainingTests(unittest.TestCase):
    def _workspace(self, root: Path, variant: str) -> Path:
        from tests.test_outcome_examples import _write
        from tests.test_query_graph import _graph
        from legal_graph.manifests import graph_to_dict

        corpus, manifest = _write(root)
        (root / "graph.json").write_text(json.dumps(graph_to_dict(_graph())), encoding="utf-8")
        config = {
            "schema_version": "1.0.0", "method": "graph_outcome_qlora_v1", "experiment_id": f"tiny-{variant}",
            "variant": variant, "selector": "expansion", "model_source": "tiny", "model_revision": "none",
            "local_model_path": "unused", "corpus_path": str(corpus), "training_data_manifest": str(manifest),
            "graph_path": str(root / "graph.json"), "provisions_path": str(root / "none.jsonl"),
            "output_dir": str(root / "run"), "input_character_limit": 300, "max_prompt_tokens": 200,
            "epochs": 1, "learning_rate": 1e-3, "encoder_learning_rate": 1e-3, "gradient_accumulation": 1,
            "seed": 7, "quantize_4bit": False, "min_cuda_memory_bytes": 0, "max_train_examples": 8,
            "lora": {"rank": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
            "graph": {"num_tokens": 2, "hidden_dim": 8, "layers": 2, "node_budget": 6, "expansion_hops": 2, "node_feature_tokens": 8},
        }
        path = root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _tiny_model(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        return LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512))

    def test_graph_typed_run_writes_manifest_and_validation(self) -> None:
        from legal_graph.outcome_training import load_outcome_config, train_outcome

        with TemporaryDirectory() as d:
            root = Path(d)
            config = load_outcome_config(self._workspace(root, "graph_typed"))
            with self.assertRaises(ValueError):  # the fixture has no validation cases
                train_outcome(config, root, model=self._tiny_model(), tokenizer=CharTokenizer())

    def test_text_variant_has_no_encoder(self) -> None:
        from legal_graph.outcome_training import OutcomeModel
        import torch

        model = OutcomeModel(self._tiny_model(), CharTokenizer(), encoder=None, max_prompt_tokens=200, device=torch.device("cpu"))
        self.assertIsNone(model.prefix_for(None))

    def test_rejects_unknown_variant(self) -> None:
        from legal_graph.outcome_training import load_outcome_config

        with TemporaryDirectory() as d:
            path = self._workspace(Path(d), "graph_typed")
            data = json.loads(path.read_text(encoding="utf-8"))
            data["variant"] = "graph_magic"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_outcome_config(path)
```

The fixture deliberately lacks validation cases, and the run must refuse rather than train without a model-selection split. Step 4 adds a validation row and asserts a full run.

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; .venv-ml\Scripts\python -m unittest tests.test_outcome_training -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'legal_graph.outcome_training'`

- [ ] **Step 3: Implement**

```python
"""Graph-augmented (or text-only) QLoRA outcome classifier: training and validation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .manifests import graph_from_dict
from .models import RetrievalConfig
from .outcome_examples import OutcomeExample, load_outcome_examples, node_feature_texts
from .outcome_prompt import label_token_ids, pick_label, prompt_token_ids
from .outcomes import LABELS, evaluate
from .query_graph import NUM_RELATION_TYPES, QuerySubgraph, build_query_subgraph
from .training import _parameter_report, _pin_base_model, _sha256_file

VARIANTS = ("text", "graph_untyped", "graph_typed", "graph_random")
SELECTORS = ("expansion", "pcst", "random")


@dataclass(frozen=True, slots=True)
class OutcomeTrainConfig:
    experiment_id: str
    variant: str
    selector: str
    model_source: str
    model_revision: str
    local_model_path: Path
    corpus_path: Path
    training_data_manifest: Path
    graph_path: Path
    provisions_path: Path
    output_dir: Path
    input_character_limit: int
    max_prompt_tokens: int
    epochs: int
    learning_rate: float
    encoder_learning_rate: float
    gradient_accumulation: int
    seed: int
    quantize_4bit: bool
    min_cuda_memory_bytes: int
    max_train_examples: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    num_graph_tokens: int
    graph_hidden_dim: int
    graph_layers: int
    node_budget: int
    expansion_hops: int
    node_feature_tokens: int


def load_outcome_config(path: Path) -> OutcomeTrainConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0.0" or data.get("method") != "graph_outcome_qlora_v1":
        raise ValueError("config must use schema 1.0.0 and graph_outcome_qlora_v1")
    variant = str(data["variant"])
    selector = "random" if variant == "graph_random" else str(data.get("selector", "expansion"))
    if variant not in VARIANTS or selector not in SELECTORS:
        raise ValueError(f"unknown variant {variant!r} or selector {selector!r}")
    lora, graph = data["lora"], data["graph"]
    return OutcomeTrainConfig(
        experiment_id=str(data["experiment_id"]), variant=variant, selector=selector,
        model_source=str(data["model_source"]), model_revision=str(data["model_revision"]),
        local_model_path=Path(data["local_model_path"]), corpus_path=Path(data["corpus_path"]),
        training_data_manifest=Path(data["training_data_manifest"]), graph_path=Path(data["graph_path"]),
        provisions_path=Path(data["provisions_path"]), output_dir=Path(data["output_dir"]),
        input_character_limit=int(data["input_character_limit"]), max_prompt_tokens=int(data["max_prompt_tokens"]),
        epochs=int(data["epochs"]), learning_rate=float(data["learning_rate"]),
        encoder_learning_rate=float(data["encoder_learning_rate"]), gradient_accumulation=int(data["gradient_accumulation"]),
        seed=int(data["seed"]), quantize_4bit=bool(data["quantize_4bit"]),
        min_cuda_memory_bytes=int(data["min_cuda_memory_bytes"]), max_train_examples=int(data["max_train_examples"]),
        lora_rank=int(lora["rank"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
        lora_target_modules=tuple(lora["target_modules"]),
        num_graph_tokens=int(graph["num_tokens"]), graph_hidden_dim=int(graph["hidden_dim"]),
        graph_layers=int(graph["layers"]), node_budget=int(graph["node_budget"]),
        expansion_hops=int(graph["expansion_hops"]), node_feature_tokens=int(graph["node_feature_tokens"]),
    )


class OutcomeModel:
    """Wraps the (PEFT) causal LM, the optional graph encoder, and node features."""

    def __init__(self, lm, tokenizer, encoder, max_prompt_tokens: int, device, features: dict | None = None) -> None:
        import torch

        self.torch = torch
        self.lm = lm
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_prompt_tokens = max_prompt_tokens
        self.device = device
        self.features = features or {}
        self.label_ids = label_token_ids(tokenizer)
        self.embedding = lm.get_input_embeddings()

    def _embed(self, ids):
        return self.embedding(self.torch.tensor(ids, device=self.device)).unsqueeze(0)

    def prefix_for(self, sub: QuerySubgraph | None, query=None, *, ablate: bool = False):
        if self.encoder is None or sub is None:
            return None
        from .graph_encoder import subgraph_tensors

        if ablate or not sub.node_ids:
            return self.encoder.null_tokens.unsqueeze(0)
        x, index, types = subgraph_tensors(sub, self.features, self.device)
        return self.encoder(x.to(self.encoder.input_proj.weight.dtype), index, types, query).unsqueeze(0)

    def _inputs(self, example: OutcomeExample, sub: QuerySubgraph | None, ablate: bool = False):
        prompt = prompt_token_ids(self.tokenizer, example.view_text, self.max_prompt_tokens)
        prompt_embeds = self._embed(prompt)
        query = prompt_embeds[0].float().mean(dim=0)
        prefix = self.prefix_for(sub, query, ablate=ablate)
        if prefix is not None:
            prompt_embeds = self.torch.cat([prefix.to(prompt_embeds.dtype), prompt_embeds], dim=1)
        return prompt, prompt_embeds

    def loss(self, example: OutcomeExample, sub: QuerySubgraph | None):
        if example.label is None:
            raise ValueError("training requires labelled examples")
        _, prompt_embeds = self._inputs(example, sub)
        answer = list(self.label_ids[example.label]) + [self.tokenizer.eos_token_id]
        embeds = self.torch.cat([prompt_embeds, self._embed(answer).to(prompt_embeds.dtype)], dim=1)
        labels = [-100] * prompt_embeds.shape[1] + answer
        out = self.lm(inputs_embeds=embeds, labels=self.torch.tensor([labels], device=self.device))
        return out.loss

    def score(self, example: OutcomeExample, sub: QuerySubgraph | None, *, ablate_graph: bool = False) -> dict[str, float]:
        torch = self.torch
        with torch.no_grad():
            _, prompt_embeds = self._inputs(example, sub, ablate=ablate_graph)
            scores: dict[str, float] = {}
            for label, ids in self.label_ids.items():
                embeds = torch.cat([prompt_embeds, self._embed(list(ids)).to(prompt_embeds.dtype)], dim=1)
                logits = self.lm(inputs_embeds=embeds).logits[0].float()
                start = prompt_embeds.shape[1] - 1
                log_probs = torch.log_softmax(logits[start:start + len(ids)], dim=-1)
                scores[label] = float(sum(log_probs[i, token] for i, token in enumerate(ids)))
        return scores


def train_outcome(config: OutcomeTrainConfig, workspace_root: Path, *, model=None, tokenizer=None) -> dict[str, Any]:
    import torch

    root = workspace_root
    output = root / config.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    corpus, manifest = root / config.corpus_path, root / config.training_data_manifest
    train = load_outcome_examples(corpus, manifest, "train", config.input_character_limit)
    validation = load_outcome_examples(corpus, manifest, "validation", config.input_character_limit)
    if not train or not validation:
        raise ValueError("train and validation splits must both contain labelled cases")
    rng = random.Random(config.seed)
    train = tuple(sorted(train, key=lambda e: hashlib.sha256(f"{config.seed}|{e.case_id}".encode()).hexdigest())[: config.max_train_examples])

    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lm, tokenizer = (model, tokenizer) if model is not None else _load_lm(config, root, device)
    lm = _attach_lora(lm, config)
    graph = graph_from_dict(json.loads((root / config.graph_path).read_text(encoding="utf-8")))
    encoder, features = None, {}
    if config.variant != "text":
        from .graph_encoder import GraphPrefixEncoder, embed_node_texts

        texts = node_feature_texts(graph, root / config.provisions_path)
        features = embed_node_texts(texts, tokenizer, lm.get_input_embeddings(), config.node_feature_tokens, device)
        model_dim = lm.get_input_embeddings().weight.shape[1]
        rms = float(lm.get_input_embeddings().weight.float().pow(2).mean().sqrt())
        encoder = GraphPrefixEncoder(
            node_dim=model_dim, hidden_dim=config.graph_hidden_dim, model_dim=model_dim,
            num_tokens=config.num_graph_tokens, num_relations=NUM_RELATION_TYPES,
            typed=config.variant == "graph_typed", num_layers=config.graph_layers, output_rms=rms,
        ).to(device)
    report = _parameter_report(lm)
    encoder_params = sum(p.numel() for p in encoder.parameters()) if encoder else 0
    trainable = report["trainable_parameters"] + encoder_params
    fraction = trainable / (report["total_parameters"] + encoder_params)
    if fraction > 0.02:
        raise RuntimeError(f"trainable fraction {fraction:.4f} exceeds the 2% cap")

    wrapper = OutcomeModel(lm, tokenizer, encoder, config.max_prompt_tokens, device, features)
    retrieval = RetrievalConfig(node_budget=config.node_budget, expansion_hops=config.expansion_hops)

    def subgraph(example: OutcomeExample) -> QuerySubgraph | None:
        if encoder is None:
            return None
        return build_query_subgraph(
            graph, query_id=example.case_id, document_id=example.document_id, view_text=example.view_text,
            as_of_date=example.decision_date, exclude_node_ids=frozenset({example.case_id}),
            config=retrieval, selector=config.selector, random_seed=config.seed,
        )

    train_subs = {e.case_id: subgraph(e) for e in train}
    val_subs = {e.case_id: subgraph(e) for e in validation}
    groups = [{"params": [p for p in lm.parameters() if p.requires_grad], "lr": config.learning_rate}]
    if encoder is not None:
        groups.append({"params": list(encoder.parameters()), "lr": config.encoder_learning_rate})
    optimiser = torch.optim.AdamW(groups)

    best: dict[str, Any] | None = None
    history = []
    for epoch in range(1, config.epochs + 1):
        lm.train()
        order = list(train)
        rng.shuffle(order)
        running = 0.0
        for step, example in enumerate(order, start=1):
            loss = wrapper.loss(example, train_subs[example.case_id]) / config.gradient_accumulation
            loss.backward()
            running += float(loss.detach()) * config.gradient_accumulation
            if step % config.gradient_accumulation == 0 or step == len(order):
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
        lm.eval()
        predictions = []
        for example in validation:
            scores = wrapper.score(example, val_subs[example.case_id])
            predictions.append({"case_id": example.case_id, "gold": example.label, "predicted": pick_label(scores), "scores": scores})
        metrics = evaluate([p["gold"] for p in predictions], [p["predicted"] for p in predictions])
        history.append({"epoch": epoch, "train_loss": running / len(order), "validation_macro_f1": metrics["macro_f1"], "validation_accuracy": metrics["accuracy"]})
        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = {"epoch": epoch, "metrics": metrics, "predictions": predictions}
            _save_checkpoint(output, lm, tokenizer, encoder, config)

    assert best is not None
    _write_jsonl(output / "validation-predictions.jsonl", best["predictions"])
    _write_json(output / "validation-metrics.json", best["metrics"])
    coverage = {
        "train_nonempty_share": _nonempty(train_subs.values()),
        "validation_nonempty_share": _nonempty(val_subs.values()),
    }
    _write_json(output / "subgraph-stats.json", coverage)
    manifest_payload = {
        "schema_version": "1.0.0",
        "status": "trained_not_frozen",
        "experiment_id": config.experiment_id,
        "variant": config.variant,
        "selector": config.selector,
        "config_sha256": hashlib.sha256(json.dumps(_config_dict(config), sort_keys=True).encode()).hexdigest(),
        "corpus_sha256": _sha256_file(corpus),
        "training_manifest_sha256": _sha256_file(manifest),
        "graph_sha256": _sha256_file(root / config.graph_path),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "best_epoch": best["epoch"],
        "history": history,
        "validation_metrics": {"accuracy": best["metrics"]["accuracy"], "macro_f1": best["metrics"]["macro_f1"]},
        "subgraph_coverage": coverage,
        "parameter_report": {**report, "encoder_parameters": encoder_params, "trainable_fraction_with_encoder": fraction},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
    }
    _write_json(output / "run.manifest.json", manifest_payload)
    return manifest_payload


def _nonempty(subs) -> float | None:
    subs = [s for s in subs if s is not None]
    return sum(s.status == "ok" for s in subs) / len(subs) if subs else None


def _load_lm(config: OutcomeTrainConfig, root: Path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < config.min_cuda_memory_bytes:
        raise RuntimeError("GPU memory is below the configured minimum")
    path = root / config.local_model_path
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch.float16}
    if config.quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        kwargs["device_map"] = {"": 0}
    lm = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not config.quantize_4bit:
        lm = lm.to(device)
    lm.config.use_cache = False
    return lm, tokenizer


def _attach_lora(lm, config: OutcomeTrainConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config.quantize_4bit:
        lm.gradient_checkpointing_enable()
        lm = prepare_model_for_kbit_training(lm)
    lm = get_peft_model(lm, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                                       bias="none", task_type="CAUSAL_LM", target_modules=list(config.lora_target_modules)))
    _pin_base_model(lm.peft_config, config.model_source, config.model_revision)
    return lm


def _save_checkpoint(output: Path, lm, tokenizer, encoder, config: OutcomeTrainConfig) -> None:
    import torch

    output.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    if encoder is not None:
        torch.save(encoder.state_dict(), output / "encoder.pt")
    _write_json(output / "config.json", _config_dict(config))


def _config_dict(config: OutcomeTrainConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value for key, value in asdict(config).items()}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
```

- [ ] **Step 4: Add a full tiny run and run the tests**

Append a validation row to the fixture inside this test module, so the shared `_write` helper stays unchanged for Task 4:

```python
    def test_full_tiny_run_on_cpu(self) -> None:
        from legal_graph.outcome_training import load_outcome_config, train_outcome
        from legal_graph.training import _sha256_file

        with TemporaryDirectory() as d:
            root = Path(d)
            path = self._workspace(root, "graph_typed")
            data = json.loads(path.read_text(encoding="utf-8"))
            corpus = Path(data["corpus_path"])
            with corpus.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"case_id": "c4", "document_id": "d4", "origin_split": "validation",
                    "decision_date": "2016-05-01", "text": "Section 482 of the CrPC. " + "x " * 40 + "The appeal is dismissed."}) + "\n")
            manifest = Path(data["training_data_manifest"])
            m = json.loads(manifest.read_text(encoding="utf-8")); m["corpus_sha256"] = _sha256_file(corpus)
            manifest.write_text(json.dumps(m), encoding="utf-8")
            result = train_outcome(load_outcome_config(path), root, model=self._tiny_model(), tokenizer=CharTokenizer())
            run = root / "run"
            self.assertEqual(result["status"], "trained_not_frozen")
            self.assertTrue((run / "encoder.pt").exists())
            self.assertTrue((run / "adapter").exists())
            self.assertEqual(result["validation_examples"], 1)
            self.assertLessEqual(result["parameter_report"]["trainable_fraction_with_encoder"], 1.0)
```

The tiny model's trainable fraction exceeds 2% because the model is so small. For this test only, the cap is bypassed by setting the environment variable `LEGAL_GRAPH_SKIP_PARAM_CAP=1`. Implement that in `train_outcome` as `if fraction > 0.02 and os.environ.get("LEGAL_GRAPH_SKIP_PARAM_CAP") != "1": raise ...`, and set the variable in this test with `unittest.mock.patch.dict(os.environ, {...})`. Every real config keeps the cap.

Run: `$env:PYTHONPATH="src"; .venv-ml\Scripts\python -m unittest tests.test_outcome_training -v`
Expected: 4 tests PASS on CPU in under a minute

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/outcome_training.py tests/test_outcome_training.py
git commit -m "feat: graph-augmented QLoRA outcome training with validation selection"
```

---

### Task 7: Freeze and evaluate once on the test split

**Files:**
- Create: `src/legal_graph/outcome_eval.py`
- Test: `tests/test_outcome_eval.py`

**Interfaces:**
- Consumes: run directory from Task 6; `OutcomeModel`, `load_outcome_config`, `_load_lm`; `load_outcome_examples(..., allow_test_labels=True)`.
- Produces:
  - `freeze_run(run_dir: Path) -> dict`: writes `freeze.json` with the SHA-256 of every file under `adapter/`, plus `encoder.pt` and `config.json`; status `frozen`. Refuses if `freeze.json` already exists.
  - `verify_freeze(run_dir: Path) -> None`: raises `ValueError("run changed after freeze")` on any hash mismatch.
  - `evaluate_frozen(run_dir: Path, workspace_root: Path, *, ablate_graph: bool = False, model=None, tokenizer=None) -> dict`: verifies the freeze, refuses if `test-evaluation[-ablated].json` exists, scores the test split, and writes `test-predictions[-ablated].jsonl` and `test-evaluation[-ablated].json`.

- [ ] **Step 1: Write the failing pure tests for freeze and verify** (no torch needed)

```python
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from legal_graph.outcome_eval import freeze_run, verify_freeze


def _run(root: Path) -> Path:
    run = root / "run"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "encoder.pt").write_bytes(b"encoder")
    (run / "config.json").write_text("{}", encoding="utf-8")
    return run


class FreezeTests(unittest.TestCase):
    def test_freeze_then_verify_passes(self) -> None:
        with TemporaryDirectory() as d:
            run = _run(Path(d))
            frozen = freeze_run(run)
            self.assertEqual(frozen["status"], "frozen")
            verify_freeze(run)

    def test_change_after_freeze_is_rejected(self) -> None:
        with TemporaryDirectory() as d:
            run = _run(Path(d))
            freeze_run(run)
            (run / "encoder.pt").write_bytes(b"retrained")
            with self.assertRaises(ValueError):
                verify_freeze(run)

    def test_double_freeze_is_rejected(self) -> None:
        with TemporaryDirectory() as d:
            run = _run(Path(d))
            freeze_run(run)
            with self.assertRaises(FileExistsError):
                freeze_run(run)
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_eval -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Freeze a trained run, then evaluate it on the held-out test split exactly once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _hashes(run_dir: Path) -> dict[str, str]:
    files = sorted(p for p in (run_dir / "adapter").rglob("*") if p.is_file())
    files += [p for p in (run_dir / "encoder.pt", run_dir / "config.json") if p.exists()]
    return {str(p.relative_to(run_dir)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def freeze_run(run_dir: Path) -> dict:
    target = run_dir / "freeze.json"
    if target.exists():
        raise FileExistsError(f"{run_dir} is already frozen")
    payload = {"schema_version": "1.0.0", "status": "frozen", "files": _hashes(run_dir)}
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_freeze(run_dir: Path) -> None:
    target = run_dir / "freeze.json"
    if not target.exists():
        raise ValueError("run is not frozen")
    if json.loads(target.read_text(encoding="utf-8"))["files"] != _hashes(run_dir):
        raise ValueError("run changed after freeze")


def evaluate_frozen(run_dir: Path, workspace_root: Path, *, ablate_graph: bool = False, model=None, tokenizer=None) -> dict:
    import torch

    from .graph_encoder import GraphPrefixEncoder, embed_node_texts
    from .manifests import graph_from_dict
    from .models import RetrievalConfig
    from .outcome_examples import load_outcome_examples, node_feature_texts
    from .outcome_prompt import pick_label
    from .outcome_training import OutcomeModel, _load_lm, load_outcome_config
    from .outcomes import evaluate
    from .query_graph import NUM_RELATION_TYPES, build_query_subgraph

    verify_freeze(run_dir)
    suffix = "-ablated" if ablate_graph else ""
    result_path = run_dir / f"test-evaluation{suffix}.json"
    if result_path.exists():
        raise FileExistsError(f"{result_path.name} already exists; the test split is evaluated once")
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config_path = run_dir / "config.reload.json"
    config_path.write_text(json.dumps({"schema_version": "1.0.0", "method": "graph_outcome_qlora_v1", **_nest(raw)}), encoding="utf-8")
    config = load_outcome_config(config_path)
    config_path.unlink()
    root = workspace_root
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        from peft import PeftModel

        base, tokenizer = _load_lm(config, root, device)
        model = PeftModel.from_pretrained(base, run_dir / "adapter")
    model.eval()
    graph = graph_from_dict(json.loads((root / config.graph_path).read_text(encoding="utf-8")))
    encoder, features = None, {}
    if config.variant != "text":
        texts = node_feature_texts(graph, root / config.provisions_path)
        features = embed_node_texts(texts, tokenizer, model.get_input_embeddings(), config.node_feature_tokens, device)
        dim = model.get_input_embeddings().weight.shape[1]
        encoder = GraphPrefixEncoder(dim, config.graph_hidden_dim, dim, config.num_graph_tokens, NUM_RELATION_TYPES,
                                     typed=config.variant == "graph_typed", num_layers=config.graph_layers).to(device)
        encoder.load_state_dict(torch.load(run_dir / "encoder.pt", map_location=device))
        encoder.eval()
    wrapper = OutcomeModel(model, tokenizer, encoder, config.max_prompt_tokens, device, features)
    examples = load_outcome_examples(root / config.corpus_path, root / config.training_data_manifest, "test",
                                     config.input_character_limit, allow_test_labels=True)
    retrieval = RetrievalConfig(node_budget=config.node_budget, expansion_hops=config.expansion_hops)
    rows = []
    for example in examples:
        sub = None if encoder is None else build_query_subgraph(
            graph, query_id=example.case_id, document_id=example.document_id, view_text=example.view_text,
            as_of_date=example.decision_date, exclude_node_ids=frozenset({example.case_id}),
            config=retrieval, selector=config.selector, random_seed=config.seed)
        scores = wrapper.score(example, sub, ablate_graph=ablate_graph)
        rows.append({"case_id": example.case_id, "gold": example.label, "predicted": pick_label(scores), "scores": scores})
    metrics = evaluate([r["gold"] for r in rows], [r["predicted"] for r in rows])
    (run_dir / f"test-predictions{suffix}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    payload = {"status": "measured_on_test", "ablate_graph": ablate_graph, "metrics": metrics, "freeze_verified": True}
    result_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def _nest(flat: dict) -> dict:
    """Rebuild the nested config schema from the flat dataclass dump saved by training."""
    moved = {"lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules", "num_graph_tokens", "graph_hidden_dim",
             "graph_layers", "node_budget", "expansion_hops", "node_feature_tokens"}
    nested = {k: v for k, v in flat.items() if k not in moved}
    nested["lora"] = {"rank": flat["lora_rank"], "alpha": flat["lora_alpha"], "dropout": flat["lora_dropout"], "target_modules": flat["lora_target_modules"]}
    nested["graph"] = {"num_tokens": flat["num_graph_tokens"], "hidden_dim": flat["graph_hidden_dim"], "layers": flat["graph_layers"],
                       "node_budget": flat["node_budget"], "expansion_hops": flat["expansion_hops"], "node_feature_tokens": flat["node_feature_tokens"]}
    return nested
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_eval -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/outcome_eval.py tests/test_outcome_eval.py
git commit -m "feat: freeze-then-evaluate-once test protocol with graph-token ablation"
```

---

### Task 8: Paired bootstrap, seed aggregation, and forecast comparison

**Files:**
- Create: `src/legal_graph/comparison.py`
- Test: `tests/test_comparison.py`

**Interfaces:**
- Consumes: `evaluate` (`outcomes.py`); the projection report JSON.
- Produces:
  - `paired_bootstrap(gold, pred_a, pred_b, *, metric="macro_f1", resamples=10000, seed=20261006) -> dict` with keys `delta`, `lower`, `upper`, `share_a_not_better`
  - `aggregate_seeds(values: Sequence[float]) -> dict` with keys `mean`, `std`, `n`
  - `compare_to_projection(measured: Mapping[str, float], projection: Mapping) -> list[dict]`, one row per variant with keys `variant_id`, `projected`, `lower`, `upper`, `measured`, `inside_interval`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import unittest

from legal_graph.comparison import aggregate_seeds, compare_to_projection, paired_bootstrap


class ComparisonTests(unittest.TestCase):
    def test_identical_predictions_have_zero_delta(self) -> None:
        gold = ["allowed", "dismissed", "disposed", "partly_allowed"] * 5
        result = paired_bootstrap(gold, gold, gold, resamples=200)
        self.assertEqual((result["delta"], result["lower"], result["upper"]), (0.0, 0.0, 0.0))

    def test_better_model_has_positive_interval(self) -> None:
        gold = ["allowed", "dismissed"] * 20
        perfect = list(gold)
        poor = ["allowed"] * 40
        result = paired_bootstrap(gold, perfect, poor, metric="accuracy", resamples=500)
        self.assertGreater(result["lower"], 0.0)
        self.assertLess(result["share_a_not_better"], 0.05)

    def test_bootstrap_is_seeded(self) -> None:
        gold = ["allowed", "dismissed", "allowed", "disposed"] * 5
        a = ["allowed"] * 20
        b = ["dismissed"] * 20
        self.assertEqual(paired_bootstrap(gold, a, b, resamples=100), paired_bootstrap(gold, a, b, resamples=100))

    def test_aggregate_seeds(self) -> None:
        self.assertEqual(aggregate_seeds([0.5, 0.7]), {"mean": 0.6, "std": 0.1414213562373095, "n": 2})

    def test_compare_to_projection_flags_interval(self) -> None:
        projection = {"variants": [{"variant_id": "v", "splits": {"test": {"accuracy": 0.76,
            "accuracy_interval": {"lower": 0.63, "upper": 0.86}}}}]}
        rows = compare_to_projection({"v": 0.60}, projection)
        self.assertEqual(rows[0]["inside_interval"], False)
```

- [ ] **Step 2: Run to verify failure**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_comparison -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Paired bootstrap, seed aggregation, and measured-versus-forecast comparison."""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from .outcomes import evaluate


def paired_bootstrap(gold, pred_a, pred_b, *, metric: str = "macro_f1", resamples: int = 10000, seed: int = 20261006) -> dict:
    if not (len(gold) == len(pred_a) == len(pred_b)) or not gold:
        raise ValueError("bootstrap needs equally sized, non-empty sequences")
    def score(indices, pred):
        return float(evaluate([gold[i] for i in indices], [pred[i] for i in indices])[metric])
    base = list(range(len(gold)))
    delta = score(base, pred_a) - score(base, pred_b)
    generator = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        sample = [generator.randrange(len(gold)) for _ in base]
        deltas.append(score(sample, pred_a) - score(sample, pred_b))
    deltas.sort()
    return {
        "metric": metric,
        "delta": delta,
        "lower": deltas[int(0.025 * resamples)],
        "upper": deltas[min(resamples - 1, int(0.975 * resamples))],
        "share_a_not_better": sum(d <= 0 for d in deltas) / resamples,
        "resamples": resamples,
    }


def aggregate_seeds(values: Sequence[float]) -> dict:
    if not values:
        raise ValueError("no seed values")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"mean": round(mean, 12), "std": std, "n": len(values)}


def compare_to_projection(measured: Mapping[str, float], projection: Mapping) -> list[dict]:
    rows = []
    for variant in projection["variants"]:
        variant_id = variant["variant_id"]
        if variant_id not in measured:
            continue
        test = variant["splits"]["test"]
        interval = test["accuracy_interval"]
        value = measured[variant_id]
        rows.append({
            "variant_id": variant_id, "projected": test["accuracy"], "lower": interval["lower"], "upper": interval["upper"],
            "measured": value, "inside_interval": interval["lower"] <= value <= interval["upper"],
        })
    return rows
```

Note: `aggregate_seeds([0.5, 0.7])` gives `mean` `0.6` only after rounding, because floating point yields `0.6` minus 1e-16. The `round(..., 12)` keeps the test exact.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_comparison -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/legal_graph/comparison.py tests/test_comparison.py
git commit -m "feat: paired bootstrap and measured-versus-forecast comparison"
```

---

### Task 9: CLI, variant configs, and a local GPU smoke run

**Files:**
- Modify: `src/legal_graph/cli.py` (add 4 subcommands after `project-outcomes`)
- Create: `configs/training/outcome/qwen25-0.5b-smoke-text.json`, `qwen25-0.5b-smoke-graph-typed.json`, `qwen3-8b-text.json`, `qwen3-8b-graph-untyped.json`, `qwen3-8b-graph-typed.json`, `qwen3-8b-graph-typed-pcst.json`, `qwen3-8b-graph-random.json`, `saul-7b-graph-typed.json`, `qwen3-4b-{text,graph-untyped,graph-typed,graph-random}.json`
- Test: `tests/test_outcome_configs.py`

**Interfaces:**
- Consumes: `load_outcome_config`, `train_outcome`, `freeze_run`, `evaluate_frozen`, `paired_bootstrap`, `compare_to_projection`.
- Produces: CLI commands `train-outcome --config`, `freeze-outcome --run`, `evaluate-outcome --run [--ablate-graph]`, `compare-outcomes --run-a --run-b [--metric]`.

- [ ] **Step 1: Write a config-validity test over every shipped config**

```python
from __future__ import annotations

from pathlib import Path
import unittest

from legal_graph.outcome_training import load_outcome_config

CONFIG_DIR = Path("configs/training/outcome")


class OutcomeConfigTests(unittest.TestCase):
    def test_every_config_loads(self) -> None:
        paths = sorted(CONFIG_DIR.glob("*.json"))
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            with self.subTest(path=path.name):
                config = load_outcome_config(path)
                self.assertTrue(str(config.output_dir).startswith("checkpoints/outcome/"))

    def test_headline_pair_matches_except_variant(self) -> None:
        text = load_outcome_config(CONFIG_DIR / "qwen3-8b-text.json")
        typed = load_outcome_config(CONFIG_DIR / "qwen3-8b-graph-typed.json")
        for field in ("model_revision", "max_prompt_tokens", "epochs", "learning_rate", "seed", "lora_rank", "max_train_examples"):
            self.assertEqual(getattr(text, field), getattr(typed, field), field)
```

- [ ] **Step 2: Write the headline config** (other configs differ only in `experiment_id`, `variant`, `selector`, `output_dir`, model fields, and `seed`)

```json
{
  "schema_version": "1.0.0",
  "method": "graph_outcome_qlora_v1",
  "experiment_id": "qwen3-8b-outcome-graph-typed-v1",
  "variant": "graph_typed",
  "selector": "expansion",
  "model_source": "Qwen/Qwen3-8B",
  "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
  "local_model_path": "models/qwen3-8b",
  "corpus_path": "data/clean/judgments/sc-2016/judgments.jsonl",
  "training_data_manifest": "data/clean/judgments/training.manifest.json",
  "graph_path": "data/graphs/judgments/sc-2016/graph.json",
  "provisions_path": "data/clean/india_statutes/provisions.jsonl",
  "output_dir": "checkpoints/outcome/qwen3-8b-graph-typed-v1",
  "input_character_limit": 6000,
  "max_prompt_tokens": 1536,
  "epochs": 3,
  "learning_rate": 0.0001,
  "encoder_learning_rate": 0.0005,
  "gradient_accumulation": 8,
  "seed": 20261006,
  "quantize_4bit": true,
  "min_cuda_memory_bytes": 14000000000,
  "max_train_examples": 359,
  "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
  "graph": {"num_tokens": 8, "hidden_dim": 256, "layers": 2, "node_budget": 32, "expansion_hops": 2, "node_feature_tokens": 48}
}
```

The smoke configs use `"model_source": "Qwen/Qwen2.5-0.5B-Instruct"`, the revision recorded in Task 0, `"local_model_path": "models/qwen2.5-0.5b-instruct"`, `"quantize_4bit": false`, `"min_cuda_memory_bytes": 0`, `"max_train_examples": 24`, `"epochs": 1`, `"max_prompt_tokens": 512`, and `"output_dir": "checkpoints/outcome/smoke-<variant>"`.

- [ ] **Step 3: Add the CLI subcommands** (follow the existing `outcome_parser` pattern in `cli.py`)

```python
    train_outcome_parser = subcommands.add_parser("train-outcome", help="train a text or graph-augmented outcome model")
    train_outcome_parser.add_argument("--config", type=Path, required=True)
    freeze_parser = subcommands.add_parser("freeze-outcome", help="freeze a trained outcome run before test evaluation")
    freeze_parser.add_argument("--run", type=Path, required=True)
    evaluate_parser = subcommands.add_parser("evaluate-outcome", help="evaluate a frozen run once on the test split")
    evaluate_parser.add_argument("--run", type=Path, required=True)
    evaluate_parser.add_argument("--ablate-graph", action="store_true")
    compare_parser = subcommands.add_parser("compare-outcomes", help="paired bootstrap between two evaluated runs")
    compare_parser.add_argument("--run-a", type=Path, required=True)
    compare_parser.add_argument("--run-b", type=Path, required=True)
    compare_parser.add_argument("--metric", default="macro_f1", choices=("macro_f1", "accuracy"))
```

And in the dispatch:

```python
    elif args.command == "train-outcome":
        from .outcome_training import load_outcome_config, train_outcome
        print(json.dumps(train_outcome(load_outcome_config(args.config), Path.cwd()), indent=2, sort_keys=True))
    elif args.command == "freeze-outcome":
        from .outcome_eval import freeze_run
        print(json.dumps(freeze_run(args.run), indent=2, sort_keys=True))
    elif args.command == "evaluate-outcome":
        from .outcome_eval import evaluate_frozen
        print(json.dumps(evaluate_frozen(args.run, Path.cwd(), ablate_graph=args.ablate_graph), indent=2, sort_keys=True))
    elif args.command == "compare-outcomes":
        from .comparison import paired_bootstrap
        def rows(run):
            return {r["case_id"]: r for r in map(json.loads, (run / "test-predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        a, b = rows(args.run_a), rows(args.run_b)
        ids = sorted(a)
        if ids != sorted(b):
            raise SystemExit("runs were evaluated on different case sets")
        print(json.dumps(paired_bootstrap([a[i]["gold"] for i in ids], [a[i]["predicted"] for i in ids],
                                          [b[i]["predicted"] for i in ids], metric=args.metric), indent=2))
```

- [ ] **Step 4: Run the config tests, then the full suite**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_outcome_configs -v; py -3 -m unittest discover -s tests -v`
Expected: 2 config tests PASS; whole suite passes (torch tests SKIP under `py -3`)

- [ ] **Step 5: Smoke-train on the RTX 3060**

```powershell
$env:PYTHONPATH="src"
.venv-ml\Scripts\python -m legal_graph.cli train-outcome --config configs/training/outcome/qwen25-0.5b-smoke-text.json
.venv-ml\Scripts\python -m legal_graph.cli train-outcome --config configs/training/outcome/qwen25-0.5b-smoke-graph-typed.json
```

Expected for each: `"status": "trained_not_frozen"`, validation metrics present, and `subgraph_coverage.validation_nonempty_share` between 0 and 1. For the graph run, encoder parameters are reported. Record wall time per step; Plan 3 uses it to budget Kaggle hours.

- [ ] **Step 6: Commit**

```bash
git add src/legal_graph/cli.py configs/training/outcome tests/test_outcome_configs.py
git commit -m "feat: outcome CLI, variant configs, and local smoke configuration"
```

---

### Task 10: Kaggle driver and one Kaggle smoke run

**Files:**
- Create: `scripts/kaggle_run.py`
- Modify: `scripts/build_kaggle_notebook.py` (outcome `RUN` options: `outcome-smoke`, `outcome-<config stem>`)
- Test: `tests/test_kaggle_run.py`

**Interfaces:**
- Consumes: `kaggle` CLI 2.x (`py -3.14 -m kaggle`), with the token at `~/.kaggle/access_token`.
- Produces:
  - `kernel_metadata(slug: str, title: str, notebook: str, *, gpu: bool = True, internet: bool = True) -> dict`, always with `is_private: True`
  - `push(folder: Path) -> str`
  - `status(ref: str) -> str`
  - `pull(ref: str, dest: Path) -> list[Path]`
  - CLI: `py -3 scripts/kaggle_run.py push|status|pull ...`

- [ ] **Step 1: Write the failing pure tests**

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("kaggle_run", Path("scripts/kaggle_run.py"))


class KaggleMetadataTests(unittest.TestCase):
    def test_metadata_is_private_gpu_notebook(self) -> None:
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        meta = module.kernel_metadata("akxran/outcome-smoke", "outcome smoke", "outcome_smoke.ipynb")
        self.assertTrue(meta["is_private"])
        self.assertTrue(meta["enable_gpu"])
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertEqual(meta["id"], "akxran/outcome-smoke")
```

- [ ] **Step 2: Implement**

```python
"""Push private Kaggle GPU kernels, poll them, and pull their outputs (Kaggle CLI 2.x)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def kernel_metadata(slug: str, title: str, notebook: str, *, gpu: bool = True, internet: bool = True) -> dict:
    return {
        "id": slug, "title": title, "code_file": notebook, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": gpu, "enable_internet": internet,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }


def _kaggle(*args: str) -> str:
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def push(folder: Path) -> str:
    return _kaggle("kernels", "push", "-p", str(folder))


def status(ref: str) -> str:
    return _kaggle("kernels", "status", ref)


def pull(ref: str, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", ref, "-p", str(dest))
    return sorted(p for p in dest.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_push = sub.add_parser("push"); p_push.add_argument("folder", type=Path)
    p_status = sub.add_parser("status"); p_status.add_argument("ref")
    p_pull = sub.add_parser("pull"); p_pull.add_argument("ref"); p_pull.add_argument("dest", type=Path)
    args = parser.parse_args()
    if args.command == "push":
        print(push(args.folder))
    elif args.command == "status":
        print(status(args.ref))
    else:
        print(json.dumps([str(p) for p in pull(args.ref, args.dest)], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the test**

Run: `$env:PYTHONPATH="src"; py -3 -m unittest tests.test_kaggle_run -v`
Expected: 1 test PASS

- [ ] **Step 4: Extend the notebook builder** with outcome runs. The new `RUN` values map to `configs/training/outcome/<stem>.json`; the run cell calls `train-outcome`, and, for graph variants, nothing else. Freezing and test evaluation happen in a separate, later kernel per Task 7. Rebuild and inspect:

Run: `py -3 scripts/build_kaggle_notebook.py --run outcome-qwen3-4b-graph-typed --out kaggle/outcome-qwen3-4b-graph-typed/`
Expected: the folder contains the `.ipynb` and `kernel-metadata.json` with `is_private: true`.

- [ ] **Step 5: Push one Kaggle smoke run** (with the user's go-ahead, since it spends quota)

```powershell
py -3 scripts/kaggle_run.py push kaggle/outcome-qwen3-4b-graph-typed
py -3 scripts/kaggle_run.py status akxran/outcome-qwen3-4b-graph-typed
py -3 scripts/kaggle_run.py pull akxran/outcome-qwen3-4b-graph-typed checkpoints/kaggle/outcome-qwen3-4b-graph-typed
```

Expected: status reaches `complete`; the pulled folder holds `run.manifest.json` with `"status": "trained_not_frozen"` and validation metrics. Check the right-hand output tree, not the folder card, which can read "empty" even when files exist.

- [ ] **Step 6: Commit**

```bash
git add scripts/kaggle_run.py scripts/build_kaggle_notebook.py tests/test_kaggle_run.py
git commit -m "feat: private Kaggle kernel driver and outcome notebook runs"
```

---

## Self-Review Notes

- **Spec coverage:**
  - Retrieval eligibility (Section 5.2): Task 1.
  - PCST (Section 5.2): Task 2, named as a greedy approximation.
  - R-GCN, pooling, projector and null token (Section 6): Task 3.
  - Masked view and hidden test labels (Section 3.4): Task 4.
  - Label-only loss positions (Section 6): Task 6.
  - 2% cap: Task 6.
  - Freeze before test, single evaluation, graph-token removal (Section 10.2–10.3): Task 7.
  - Paired bootstrap and forecast comparison (Section 10): Task 8.
  - Random-retrieval control: the `graph_random` variant (Tasks 1 and 6).
  - Seeds: per-config `seed`, aggregated in Task 8.
  - Not covered by this plan: the auxiliary alignment and link losses ($\alpha$, $\beta$ in Section 6). They are deferred to Plan 3 as an ablation, because the headline comparison uses the task loss only. The paper's limitations list already states this.
- **Type consistency:**
  - `QuerySubgraph.edges` holds `(source_index, target_index, relation_index)` throughout.
  - `NUM_RELATION_TYPES` feeds `GraphPrefixEncoder(num_relations=...)`.
  - `OutcomeModel.score(..., ablate_graph=...)` is used identically in Tasks 6 and 7.
- **Review Focus pins:**
  1. Empty seeds: Task 1 `test_no_citation_gives_empty_subgraph` and Task 3 `test_empty_graph_returns_learned_null_tokens`.
  2. BNS in 2016: Task 1 `test_only_train_split_and_past_nodes_are_reachable`.
  3. Unlabelled test examples: Task 4 `test_test_labels_hidden_by_default`, plus `OutcomeModel.loss` raising on `label is None`.
  4. Post-freeze change: Task 7 `test_change_after_freeze_is_rejected`.
  5. Multi-token verbaliser and ties: Task 5 tests.
