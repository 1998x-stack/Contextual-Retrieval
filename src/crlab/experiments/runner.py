"""Reproducible baseline execution against normalized Anthropic data."""

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
from crlab.schemas import Dataset


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


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
) -> tuple[dict[str, float], list[dict]]:
    """Execute a complete query set; results are real rankings, never reference values."""
    if method not in {"bm25", "vector", "hybrid"}:
        raise ValueError(f"Unknown method: {method}")
    if not cutoffs or min(cutoffs) <= 0:
        raise ValueError("Cutoffs must be nonempty positive integers")
    if candidate_pool < max(cutoffs):
        raise ValueError("Candidate pool must be >= maximum evaluated cutoff")
    if method != "bm25" and embedder is None:
        raise ValueError(f"Method {method} requires an explicit embedder")

    by_id = dataset.chunk_by_id
    if context_by_id is not None and set(context_by_id) != set(by_id):
        raise ValueError("Context must cover all original chunks, without extras")
    # Context is indexed, but original chunk bodies remain the evaluation evidence.
    text_by_id = {
        key: f"{context_by_id[key]}\n\n{chunk.content}" if context_by_id is not None
        else chunk.content
        for key, chunk in by_id.items()
    }
    bm25 = LocalBM25(
        dataset.chunks,
        context_by_id=dict(context_by_id) if context_by_id is not None else None,
    ) if method in {"bm25", "hybrid"} else None
    vector = None
    if method in {"vector", "hybrid"}:
        assert embedder is not None
        embeddings = embedder.embed_documents([text_by_id[chunk.chunk_id] for chunk in dataset.chunks])
        vector = ExactVector(
            dataset.chunks,
            embeddings,
            lambda query: embedder.embed_queries([query])[0],
        )

    rows: list[dict] = []
    values: list[dict[str, float]] = []
    for query in dataset.queries:
        if method == "bm25":
            assert bm25 is not None
            hits = bm25.search(query.text, candidate_pool)
        elif method == "vector":
            assert vector is not None
            hits = vector.search(query.text, candidate_pool)
        else:
            assert vector is not None and bm25 is not None
            semantic = vector.search(query.text, candidate_pool)
            lexical = bm25.search(query.text, candidate_pool)
            hits = fuse(
                [(semantic_weight, semantic), (bm25_weight, lexical)],
                candidate_pool,
                chunk_by_id=by_id,
            )
        scores = evaluate_query(
            query, hits,
            {key: chunk.content for key, chunk in by_id.items()},
            cutoffs,
        )
        values.append(scores)
        rows.append({
            "query_id": query.query_id,
            "query": query.text,
            "gold_chunk_ids": list(query.golden_chunk_ids),
            "rankings": [asdict(hit) for hit in hits],
            "metrics": scores,
        })
    return aggregate(values), rows


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
) -> None:
    path = Path(out)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run artifacts in {path}")
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "succeeded",
        "track": "method_reproduction",
        "method": method,
        "vector_model": model_name,
        "lexical_engine": "local_python_bm25" if method != "vector" else None,
        "candidate_pool": candidate_pool,
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
            "This is an independently executed result, not an official score. "
            "Local BM25 does not replicate the Elasticsearch English analyzer."
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
