"""Load and validate stored runs before computing research conclusions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from crlab.evaluation.metrics import evaluate_query
from crlab.schemas import Query, SearchHit


def _verify_id_metrics(row: dict, run_path: Path) -> None:
    """Replay all ID-based metrics using persisted rankings rather than trusted scores.

    Legacy Cookbook content matching is not reconstructible without raw texts,
    so its values are checked for bounds and aggregate consistency separately.
    """
    measured = row["metrics"]
    keys = ("hit", "recall", "all_evidence", "mrr", "ndcg")
    cutoffs = sorted({
        int(name.rsplit("@", 1)[1])
        for name in measured
        if "@" in name and name.split("@", 1)[0] in keys
    })
    query = Query(
        query_id=row["query_id"],
        text=row["query"],
        golden_chunk_ids=tuple(row["gold_chunk_ids"]),
        golden_contents=tuple("gold-" + str(i) for i in range(len(row["gold_chunk_ids"]))),
    )
    for k in cutoffs:
        by_cutoff = row.get("rankings_by_cutoff")
        entries = by_cutoff.get(str(k)) if by_cutoff is not None else row.get("rankings")
        if not isinstance(entries, list):
            raise ValueError(f"Missing saved ranking at cutoff {k}: {run_path}")
        hits = []
        for index, item in enumerate(entries, 1):
            identity = item.get("chunk_id") if isinstance(item, dict) else None
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"Invalid ranked chunk identity: {run_path}")
            hits.append(SearchHit(identity, 0.0, index, "replayed"))
        computed = evaluate_query(query, hits, {}, (k,))
        for name in keys:
            field = f"{name}@{k}"
            if field in measured and not math.isclose(
                measured[field], computed[field], abs_tol=1e-9, rel_tol=0
            ):
                raise ValueError(
                    f"Ranking and ID metric disagree for {row['query_id']} "
                    f"at {field}: {run_path}"
                )


class IncompatibleRuns(ValueError):
    """Comparisons would be invalid due to differing corpus, qrels or queries."""


@dataclass(frozen=True)
class Run:
    path: Path
    manifest: dict[str, Any]
    metrics: dict[str, float]
    queries: dict[str, dict[str, Any]]

    @property
    def label(self) -> str:
        return self.manifest.get("experiment_id") or self.path.name

    @property
    def dataset_key(self) -> tuple[str, str]:
        return (
            self.manifest["corpus_sha256"],
            self.manifest["evaluation_sha256"],
        )


def _load_json(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return obj


def load_run(path: str | Path) -> Run:
    """Reject partial, contradictory or non-finite artifacts (v1 and v2)."""
    root = Path(path)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError(f"Unsupported run manifest schema: {root}")
    for name in ("corpus_sha256", "evaluation_sha256"):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Missing source identity {name} in {root}")
    if manifest.get("status") != "succeeded":
        raise ValueError(f"Only succeeded runs may be analyzed: {root}")
    if not metrics:
        raise ValueError(f"Empty metrics file: {root}")
    for key, value in metrics.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid aggregate metric {key!r}: {root}")
        if not (0 <= value <= 1):
            raise ValueError(f"Metric outside [0,1]: {key} in {root}")

    rows: dict[str, dict] = {}
    with (root / "per_query.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record {index}: {root}")
            row = json.loads(line)
            qid = row["query_id"]
            if not isinstance(qid, str) or not qid or qid in rows:
                raise ValueError(f"Duplicate/missing query ID {qid!r}: {root}")
            if not isinstance(row.get("query"), str):
                raise ValueError(f"Missing query text {qid!r}: {root}")
            gold = row.get("gold_chunk_ids")
            if not isinstance(gold, list) or not gold or len(set(gold)) != len(gold):
                raise ValueError(f"Invalid gold chunk IDs {qid!r}: {root}")
            measurements = row.get("metrics")
            if not isinstance(measurements, dict) or set(measurements) != set(metrics):
                raise ValueError(f"Metric schema disagreement for {qid!r}: {root}")
            for key, value in measurements.items():
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError(f"Invalid query-level metric {key!r}: {root}")
                if not 0 <= value <= 1:
                    raise ValueError(f"Out-of-range query-level metric {key!r}: {root}")
            _verify_id_metrics(row, root)
            rows[qid] = row
    if not rows or len(rows) != manifest.get("query_count"):
        raise ValueError(f"Missing or extra query rows in {root}")

    # Prevent outdated metrics.json from being used against different rankings.
    for metric, saved in metrics.items():
        recalculated = mean(row["metrics"][metric] for row in rows.values())
        if not math.isclose(saved, recalculated, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Aggregate and query metrics disagree at {metric}: {root}")
    return Run(root, manifest, metrics, rows)


def assert_pairable(reference: Run, candidate: Run, metric: str) -> list[str]:
    """Align by query ID; ensure not just file names but entire evidence identity matches.

    Different models, retrieval methods and contexts are intentional ablations.
    Different queries / Qrels are *not* comparable within a paired test.
    """
    if reference.dataset_key != candidate.dataset_key:
        raise IncompatibleRuns("Corpus or evaluation SHA-256 differs")
    if metric not in reference.metrics or metric not in candidate.metrics:
        raise IncompatibleRuns(f"Requested metric {metric!r} not present in both runs")
    if set(reference.queries) != set(candidate.queries):
        raise IncompatibleRuns("Query ID sets differ")
    for qid in sorted(reference.queries):
        a, b = reference.queries[qid], candidate.queries[qid]
        if a["query"] != b["query"]:
            raise IncompatibleRuns(f"Query text changed at {qid}")
        if a["gold_chunk_ids"] != b["gold_chunk_ids"]:
            raise IncompatibleRuns(f"Golden chunk IDs changed at {qid}")
    return sorted(reference.queries)
