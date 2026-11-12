"""Exact dot-product retrieval (matches Cookbook's similarity definition)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from crlab.schemas import Chunk, SearchHit


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must have the same nonzero dimensionality")
    if any(not math.isfinite(value) for value in left) or any(
        not math.isfinite(value) for value in right
    ):
        raise ValueError("Embedding values must be finite")
    result = float(sum(a * b for a, b in zip(left, right, strict=True)))
    if not math.isfinite(result):
        raise ValueError("Dot product overflowed to a nonfinite score")
    return result


class ExactVector:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embed_query: Callable[[str], Sequence[float]],
    ) -> None:
        if not chunks or len(chunks) != len(embeddings):
            raise ValueError("One embedding required per chunk")
        dim = len(embeddings[0])
        if not dim or any(len(vector) != dim for vector in embeddings):
            raise ValueError("Inconsistent embedding dimensions")
        if any(not math.isfinite(value) for vector in embeddings for value in vector):
            raise ValueError("Embedding values must be finite")
        self.chunks = tuple(chunks)
        self.embeddings = tuple(tuple(x) for x in embeddings)
        self.embed_query = embed_query

    def search(self, query: str, k: int) -> list[SearchHit]:
        if k <= 0:
            return []
        query_vector = self.embed_query(query)
        scores = [
            dot_product(vector, query_vector)
            for vector in self.embeddings
        ]
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: (-scores[i], self.chunks[i].chunk_id),
        )[:k]
        return [
            SearchHit(self.chunks[i].chunk_id, scores[i], rank, "vector")
            for rank, i in enumerate(ranked, start=1)
        ]
