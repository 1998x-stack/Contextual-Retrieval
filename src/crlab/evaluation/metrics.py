"""Chunk-level IR metrics and Cookbook-compatible legacy content matching."""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, Mapping, Sequence

from crlab.schemas import Query, SearchHit


def _ranked_ids(hits: Sequence[SearchHit], cutoff: int) -> list[str]:
    """Discard repeated IDs (keeping the first occurrence) before evaluating."""
    seen: set[str] = set()
    result: list[str] = []
    for hit in hits:
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        result.append(hit.chunk_id)
        if len(result) >= cutoff:
            break
    return result


def evaluate_query(
    query: Query,
    hits: Sequence[SearchHit],
    content_by_id: Mapping[str, str],
    cutoffs: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    """Scores are fractions in [0, 1]; no implicit percentage conversion."""
    gold = set(query.golden_chunk_ids)
    if not gold:
        raise ValueError("All evaluated queries need at least one gold chunk")
    values: dict[str, float] = {}
    for k in cutoffs:
        if k <= 0:
            raise ValueError("Cutoffs must be positive")
        ids = _ranked_ids(hits, k)
        hits_gold = [identity in gold for identity in ids]
        found = sum(hits_gold)
        values[f"hit@{k}"] = float(found > 0)
        values[f"recall@{k}"] = found / len(gold)
        values[f"all_evidence@{k}"] = float(found == len(gold))
        first_rank = next((i + 1 for i, relevant in enumerate(hits_gold) if relevant), 0)
        values[f"mrr@{k}"] = 1 / first_rank if first_rank else 0.0
        dcg = sum(
            int(relevant) / math.log2(index + 2)
            for index, relevant in enumerate(hits_gold)
        )
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        values[f"ndcg@{k}"] = dcg / ideal

        # Cookbook compares strings, *not* document UUIDs. Deliberately preserve
        # duplicate string behavior to expose historical evaluation discrepancies.
        retrieved_contents = {
            content_by_id[identity].strip()
            for identity in ids
            if identity in content_by_id
        }
        matches = sum(
            content.strip() in retrieved_contents
            for content in query.golden_contents
        )
        values[f"cookbook_pass@{k}"] = matches / len(query.golden_contents)
    return values


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate zero query evaluations")
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Per-query metric fields must be identical")
    return {key: mean(row[key] for row in rows) for key in sorted(keys)}
