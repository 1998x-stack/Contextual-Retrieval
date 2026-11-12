"""Cohere reranking with the Cookbook candidate text layout and verified IDs."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Protocol

from crlab.schemas import SearchHit


class Reranker(Protocol):
    model: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        *,
        original_by_id: Mapping[str, str],
        context_by_id: Mapping[str, str] | None,
        top_k: int,
    ) -> list[SearchHit]: ...


def cookbook_rerank_document(original: str, context: str | None) -> str:
    """Actual Cookbook rerank text: original, then 'Context: <generated text>'."""
    if not context:
        return original
    return f"{original}\n\nContext: {context}"


class CohereReranker:
    """External API adapter; requires an explicit model ID and opt-in at CLI."""

    def __init__(self, model: str, *, client: object | None = None) -> None:
        if not model or not model.strip():
            raise ValueError("Cohere rerank model ID must be explicit")
        self.model = model
        if client is None:
            if not os.getenv("COHERE_API_KEY"):
                raise RuntimeError("Set COHERE_API_KEY before a paid reranking run")
            try:
                import cohere
            except ImportError as exc:
                raise RuntimeError(
                    "Install the reranking provider: pip install -e '.[cohere]'"
                ) from exc
            client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        self.client = client

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        *,
        original_by_id: Mapping[str, str],
        context_by_id: Mapping[str, str] | None,
        top_k: int,
    ) -> list[SearchHit]:
        if top_k <= 0 or not candidates:
            return []
        if len({hit.chunk_id for hit in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate IDs cannot be reranked")
        documents = [
            cookbook_rerank_document(
                original_by_id[hit.chunk_id],
                context_by_id.get(hit.chunk_id) if context_by_id is not None else None,
            )
            for hit in candidates
        ]
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )
        returned = response.results
        if not isinstance(returned, Sequence) or len(returned) != min(top_k, len(documents)):
            raise ValueError("Reranker returned an incomplete or invalid result list")
        used: set[int] = set()
        scored: list[SearchHit] = []
        for rank, item in enumerate(returned, 1):
            if type(item.index) is not int:
                raise ValueError("Reranker candidate index must be an integer")
            index = item.index
            relevance_score = float(item.relevance_score)
            if not math.isfinite(relevance_score):
                raise ValueError("Reranker relevance score must be finite")
            if index < 0 or index >= len(candidates) or index in used:
                raise ValueError(f"Reranker returned invalid candidate index {index}")
            used.add(index)
            scored.append(SearchHit(
                chunk_id=candidates[index].chunk_id,
                score=relevance_score,
                rank=rank,
                source="cohere_rerank",
            ))
        return scored
