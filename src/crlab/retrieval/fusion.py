"""Retrieval fusion: faithful Cookbook reciprocal rank + standard RRF control."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from crlab.schemas import Chunk, SearchHit


def fuse(
    ranked_sources: Sequence[tuple[float, Sequence[SearchHit]]],
    k: int,
    *,
    smooth: float = 0.0,
    chunk_by_id: Mapping[str, Chunk] | None = None,
) -> list[SearchHit]:
    """Score each candidate by sum(weight / (smooth + one_based_rank)).

    For Cookbook-compatible behavior use smooth=0 and weights (0.8,0.2).
    When chunk_by_id is present, reproduce Cookbook tie-break using source
    doc_id and original_index, descending. Original score is kept for audit.
    """
    if k <= 0:
        return []
    if smooth < 0:
        raise ValueError("RRF smoothing cannot be negative")
    scores: dict[str, float] = {}
    for weight, hits in ranked_sources:
        if weight < 0:
            raise ValueError("Fusion weights must be nonnegative")
        observed: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id in observed:
                continue
            observed.add(hit.chunk_id)
            scores[hit.chunk_id] = (
                scores.get(hit.chunk_id, 0.0) + weight / (smooth + rank)
            )

    def tie_key(identity: str) -> tuple[str, int]:
        if chunk_by_id and identity in chunk_by_id:
            chunk = chunk_by_id[identity]
            return chunk.source_doc_id, chunk.index
        return identity, 0

    ranked = sorted(
        scores,
        key=lambda identity: (scores[identity], *tie_key(identity)),
        reverse=True,
    )[:k]
    return [
        SearchHit(identity, scores[identity], rank, "fusion")
        for rank, identity in enumerate(ranked, start=1)
    ]
