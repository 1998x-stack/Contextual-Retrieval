"""Paired retrieval comparisons, query-level errors and label-aware breakdowns."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from crlab.analysis.io import Run, assert_pairable
from crlab.analysis.statistics import paired_bootstrap, paired_signflip_pvalue


def load_query_labels(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Optional explicit JSONL: {query_id, category, group_id}.

    A category is user-supplied gold metadata, never inferred from score.
    group_id identifies correlated questions for cluster resampling.
    """
    if path is None:
        return {}
    labels: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = record["query_id"]
            if qid in labels:
                raise ValueError(f"Duplicate query annotation {qid}")
            label = record.get("category", "unlabeled")
            group = record.get("group_id", qid)
            if not all(isinstance(v, str) and v.strip() for v in (qid, label, group)):
                raise ValueError(f"Invalid query annotation {record}")
            labels[qid] = {"category": label, "group_id": group}
    return labels


def _rank_of_first_gold(row: dict, cutoff: int) -> int | None:
    # Cookbook mode emits a distinct ranking for each K.
    cutoff_rankings = row.get("rankings_by_cutoff")
    if cutoff_rankings:
        ranked = cutoff_rankings.get(str(cutoff))
        if ranked is None:
            return None
    else:
        ranked = row.get("rankings", [])
    gold = set(row["gold_chunk_ids"])
    for rank, item in enumerate(ranked[:cutoff], start=1):
        if item["chunk_id"] in gold:
            return rank
    return None


def compare_runs(
    baseline: Run,
    candidate: Run,
    *,
    metric: str,
    resamples: int = 2000,
    permutations: int = 5000,
    seed: int = 42,
    labels: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ids = assert_pairable(baseline, candidate, metric)
    annotations = labels or {}
    if set(annotations) - set(ids):
        raise ValueError("Annotation file contains unknown query IDs")
    cut = int(metric.rsplit("@", 1)[1]) if "@" in metric else None
    rows: list[dict[str, Any]] = []
    for qid in ids:
        left, right = baseline.queries[qid], candidate.queries[qid]
        a, b = left["metrics"][metric], right["metrics"][metric]
        detail: dict[str, Any] = {
            "query_id": qid,
            "query": left["query"],
            "gold_chunk_ids": left["gold_chunk_ids"],
            "baseline_score": a,
            "candidate_score": b,
            "delta": b - a,
            "category": annotations.get(qid, {}).get("category", "unlabeled"),
            "group_id": annotations.get(qid, {}).get("group_id", qid),
            "change": "improved" if b > a else "regressed" if b < a else "unchanged",
        }
        if cut is not None:
            detail["baseline_first_gold_rank"] = _rank_of_first_gold(left, cut)
            detail["candidate_first_gold_rank"] = _rank_of_first_gold(right, cut)
        rows.append(detail)

    deltas = [row["delta"] for row in rows]
    groups = [row["group_id"] for row in rows] if labels is not None else None
    low, high = paired_bootstrap(
        deltas, resamples=resamples, seed=seed, groups=groups,
    )
    p = paired_signflip_pvalue(
        deltas, permutations=permutations, seed=seed, groups=groups,
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row["delta"])
    breakdown = {
        label: {"query_count": len(d), "mean_delta": mean(d)}
        for label, d in sorted(grouped.items())
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "baseline": str(baseline.path),
        "candidate": str(candidate.path),
        "baseline_experiment": baseline.label,
        "candidate_experiment": candidate.label,
        "metric": metric,
        "query_count": len(rows),
        "corpus_sha256": baseline.dataset_key[0],
        "evaluation_sha256": baseline.dataset_key[1],
        "baseline_mean": baseline.metrics[metric],
        "candidate_mean": candidate.metrics[metric],
        "mean_delta": mean(deltas),
        "ci_method": "percentile_paired_bootstrap",
        "confidence": 0.95,
        "ci_low": low,
        "ci_high": high,
        "test_method": "paired_signflip_two_sided",
        "pvalue": p,
        "resamples": resamples,
        "permutations": permutations,
        "seed": seed,
        "resampling_unit": "query" if labels is None else "group_id",
        "num_groups": len(set(groups)) if groups is not None else len(rows),
        "improved": sum(row["change"] == "improved" for row in rows),
        "regressed": sum(row["change"] == "regressed" for row in rows),
        "unchanged": sum(row["change"] == "unchanged" for row in rows),
        "category_breakdown": breakdown,
        "top_regressions": sorted(
            (row for row in rows if row["delta"] < 0),
            key=lambda row: (row["delta"], row["query_id"])
        )[:10],
        "top_improvements": sorted(
            (row for row in rows if row["delta"] > 0),
            key=lambda row: (-row["delta"], row["query_id"])
        )[:10],
        "caveat": (
            "A paired association on one fixed evaluation set, not a causal estimate "
            "or proof of generalization. Repeated prompt/model tuning on the test "
            "set invalidates held-out claims."
        ),
    }
    return result, rows
