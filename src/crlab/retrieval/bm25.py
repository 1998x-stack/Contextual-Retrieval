"""Transparent local BM25 baseline; NOT Elasticsearch English analyzer parity."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from crlab.schemas import Chunk, SearchHit

_WORD = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


class LocalBM25:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        context_by_id: dict[str, str] | None = None,
    ) -> None:
        if k1 <= 0 or not (0 <= b <= 1):
            raise ValueError("Expected k1 > 0 and 0 <= b <= 1")
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.k1, self.b = k1, b
        self.chunks = tuple(chunks)
        context_by_id = context_by_id or {}
        self.tf = [
            Counter(tokenize(
                f"{context_by_id.get(chunk.chunk_id, '')} {chunk.content}"
                if chunk.chunk_id in context_by_id else chunk.content
            ))
            for chunk in chunks
        ]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg_len = sum(self.lengths) / len(self.lengths) or 1
        postings: dict[str, list[int]] = defaultdict(list)
        for position, terms in enumerate(self.tf):
            for term in terms:
                postings[term].append(position)
        self.postings = dict(postings)

    def search(self, query: str, k: int) -> list[SearchHit]:
        if k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        n = len(self.chunks)
        for term in set(tokenize(query)):
            positions = self.postings.get(term, [])
            if not positions:
                continue
            df = len(positions)
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for position in positions:
                tf = self.tf[position][term]
                normalization = self.k1 * (
                    1 - self.b + self.b * self.lengths[position] / self.avg_len
                )
                scores[position] += idf * tf * (self.k1 + 1) / (tf + normalization)
        # Only matches participate, just like a term-based search engine query.
        ranked = sorted(scores, key=lambda p: (-scores[p], self.chunks[p].chunk_id))[:k]
        return [
            SearchHit(self.chunks[position].chunk_id, scores[position], rank, "local_bm25")
            for rank, position in enumerate(ranked, start=1)
        ]
