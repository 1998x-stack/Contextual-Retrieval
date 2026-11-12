"""Reproducible retrieval, contextual indexing, and reranking benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from crlab.evaluation.metrics import aggregate, evaluate_query
from crlab.retrieval.bm25 import LocalBM25
from crlab.retrieval.fusion import fuse
from crlab.retrieval.vector import ExactVector
from crlab.schemas import Dataset, SearchHit


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class RerankingProvider(Protocol):
    model: str

    def rerank(
        self, query: str, candidates: Sequence[SearchHit], *,
        original_by_id: Mapping[str, str],
        context_by_id: Mapping[str, str] | None,
        top_k: int,
    ) -> list[SearchHit]: ...


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def run(
    dataset: Dataset,
    *,
    method: str,
    embedder: EmbeddingProvider | None = None,
    cutoffs: tuple[int, ...] = (5, 10, 20),
    candidate_pool: int = 150,
    semantic_weight: float = 0.8,
    bm25_weight: float = 0.2,
    context_by_id: Mapping[str, str] | None = None,
    lexical_backend: str = "local",
    es_index_name: str | None = None,
    elasticsearch_url: str = "http://localhost:9200",
    es_recreate: bool = False,
    reranker: RerankingProvider | None = None,
    rerank_mode: str = "fixed",
) -> tuple[dict[str, float], list[dict]]:
    """Evaluate actual retrieved candidates; never substitute reference scores.

    Fixed rerank mode: one shared pool of 'candidate_pool' per query,
    reranked once to max(cutoffs), then evaluated across all K.

    Cookbook rerank mode: vector-only search k*10 then rerank to k for
    each cutoff separately, matching the example's k-dependent candidates.
    """
    if method not in {"bm25", "vector", "hybrid"}:
        raise ValueError(f"Unknown method: {method}")
    if lexical_backend not in {"local", "elasticsearch"}:
        raise ValueError(f"Unknown lexical backend: {lexical_backend}")
    if rerank_mode not in {"fixed", "cookbook"}:
        raise ValueError(f"Unknown rerank mode: {rerank_mode}")
    if not cutoffs or min(cutoffs) <= 0 or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Cutoffs must be distinct positive integers")
    if candidate_pool < max(cutoffs):
        raise ValueError("Candidate pool must be >= maximum evaluated cutoff")
    if method != "bm25" and embedder is None:
        raise ValueError(f"Method {method} requires an explicit embedder")
    if rerank_mode == "cookbook" and (
        reranker is None or method != "vector"
    ):
        raise ValueError("Cookbook rerank mode requires a vector retriever and reranker")
    if min(semantic_weight, bm25_weight) < 0:
        raise ValueError("Fusion weights cannot be negative")
    if context_by_id is not None and set(context_by_id) != set(dataset.chunk_by_id):
        raise ValueError("Context must cover all original chunks, without extras")

    by_id = dataset.chunk_by_id
    original_by_id = {key: chunk.content for key, chunk in by_id.items()}
    indexed_texts = [
        (
            f"{context_by_id[chunk.chunk_id]}\n\n{chunk.content}"
            if context_by_id is not None else chunk.content
        )
        for chunk in dataset.chunks
    ]
    lexical = None
    if method in {"bm25", "hybrid"}:
        if lexical_backend == "local":
            lexical = LocalBM25(
                dataset.chunks,
                context_by_id=dict(context_by_id) if context_by_id is not None else None,
            )
        else:
            if not es_index_name:
                raise ValueError("An explicit --es-index is required for Elasticsearch")
            from crlab.retrieval.elasticsearch import ElasticsearchBM25
            lexical = ElasticsearchBM25(
                dataset.chunks,
                index_name=es_index_name,
                endpoint=elasticsearch_url,
                context_by_id=context_by_id,
                recreate=es_recreate,
            )

    vector = None
    if method in {"vector", "hybrid"}:
        assert embedder is not None
        embeddings = embedder.embed_documents(indexed_texts)
        vector = ExactVector(
            dataset.chunks, embeddings,
            lambda query: embedder.embed_queries([query])[0],
        )

    per_query: list[dict] = []
    score_rows: list[dict[str, float]] = []
    for query in dataset.queries:
        if rerank_mode == "cookbook":
            assert vector is not None and reranker is not None
            scores: dict[str, float] = {}
            rankings_by_cutoff: dict[str, list[dict]] = {}
            candidates_by_cutoff: dict[str, list[dict]] = {}
            for k in sorted(cutoffs):
                raw_candidates = vector.search(query.text, k * 10)
                reranked = reranker.rerank(
                    query.text, raw_candidates,
                    original_by_id=original_by_id,
                    context_by_id=context_by_id,
                    top_k=k,
                )
                if len(reranked) != min(k, len(raw_candidates)):
                    raise ValueError("Reranker returned an unexpected number of candidates")
                scores.update(evaluate_query(query, reranked, original_by_id, (k,)))
                rankings_by_cutoff[str(k)] = [asdict(hit) for hit in reranked]
                candidates_by_cutoff[str(k)] = [asdict(hit) for hit in raw_candidates]
            final_rankings = rankings_by_cutoff[str(max(cutoffs))]
            record = {
                "query_id": query.query_id,
                "query": query.text,
                "gold_chunk_ids": list(query.golden_chunk_ids),
                "rankings": final_rankings,
                "rankings_by_cutoff": rankings_by_cutoff,
                "candidate_rankings_by_cutoff": candidates_by_cutoff,
                "metrics": scores,
            }
        else:
            if method == "bm25":
                assert lexical is not None
                candidates = lexical.search(query.text, candidate_pool)
            elif method == "vector":
                assert vector is not None
                candidates = vector.search(query.text, candidate_pool)
            else:
                assert vector is not None and lexical is not None
                semantic = vector.search(query.text, candidate_pool)
                lexical_hits = lexical.search(query.text, candidate_pool)
                candidates = fuse(
                    [(semantic_weight, semantic), (bm25_weight, lexical_hits)],
                    candidate_pool, chunk_by_id=by_id,
                )
            if reranker is not None:
                ranked = reranker.rerank(
                    query.text, candidates,
                    original_by_id=original_by_id,
                    context_by_id=context_by_id,
                    top_k=max(cutoffs),
                )
                if len(ranked) != min(max(cutoffs), len(candidates)):
                    raise ValueError("Reranker returned an unexpected number of candidates")
            else:
                ranked = candidates
            scores = evaluate_query(query, ranked, original_by_id, cutoffs)
            record = {
                "query_id": query.query_id,
                "query": query.text,
                "gold_chunk_ids": list(query.golden_chunk_ids),
                "rankings": [asdict(hit) for hit in ranked],
                "candidate_rankings": [asdict(hit) for hit in candidates],
                "metrics": scores,
            }
        score_rows.append(scores)
        per_query.append(record)
    return aggregate(score_rows), per_query


def save_run(
    out: str | Path,
    *,
    method: str,
    scores: dict[str, float],
    per_query: list[dict],
    corpus_path: str | Path,
    evaluation_path: str | Path,
    model_name: str | None = None,
    candidate_pool: int = 150,
    weights: tuple[float, float] = (0.8, 0.2),
    context_manifest: dict | None = None,
    lexical_backend: str = "local",
    es_index_name: str | None = None,
    reranker_model: str | None = None,
    rerank_mode: str = "fixed",
    experiment_id: str | None = None,
) -> None:
    path = Path(out)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run artifacts in {path}")
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "track": "method_reproduction",
        "experiment_id": experiment_id,
        "method": method,
        "vector_model": model_name,
        "lexical_engine": lexical_backend if method != "vector" else None,
        "elasticsearch_index": es_index_name if lexical_backend == "elasticsearch" else None,
        "candidate_pool": candidate_pool,
        "reranker_model": reranker_model,
        "rerank_mode": rerank_mode if reranker_model is not None else None,
        "fusion": "cookbook_weighted_reciprocal" if method == "hybrid" else None,
        "fusion_weights": list(weights) if method == "hybrid" else None,
        "corpus_sha256": file_hash(corpus_path),
        "evaluation_sha256": file_hash(evaluation_path),
        "query_count": len(per_query),
        "context": (
            {
                "strategy": context_manifest["strategy"],
                "model": context_manifest["model"],
                "prompt_version": context_manifest["prompt_version"],
                "records_sha256": context_manifest["records_sha256"],
            } if context_manifest is not None else None
        ),
        "notes": (
            "Independent executed result; no official benchmark scores substituted. "
            "For Elasticsearch: English analyzer over content and contextualized_content; "
            "ties sorted by chunk_id. Local BM25 is not Elasticsearch parity."
        ),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    with (path / "per_query.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_query:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
