"""Regression tests for security, correctness and reproducibility review fixes.

All providers are stubs; these tests make no billable API calls.
"""

import json
import math
from types import SimpleNamespace

import pytest

from crlab.contextualization.cache import context_cache_key
from crlab.contextualization.engine import (
    AnthropicContextualizer,
    MetadataContextualizer,
)
from crlab.experiments.matrix import plan_matrix, plan_model_sweep
from crlab.experiments.runner import run, save_run
from crlab.providers.voyage import VoyageEmbedder
from crlab.reranking.cohere import CohereReranker
from crlab.retrieval.vector import dot_product, ExactVector
from crlab.schemas import SearchHit


class StubReranker:
    model = "test-only"

    def __init__(self, mode):
        self.mode = mode

    def rerank(self, query, candidates, *, original_by_id, context_by_id, top_k):
        if self.mode == "alien":
            return [SearchHit("alien-document:0", 1, 1, "fake")]
        if self.mode == "duplicate":
            return [
                SearchHit(candidates[0].chunk_id, 1, 1, "fake"),
                SearchHit(candidates[0].chunk_id, 0.9, 2, "fake"),
            ]
        if self.mode == "wrong-rank":
            return [SearchHit(candidates[0].chunk_id, 1, 0, "fake")]
        if self.mode == "nonfinite":
            return [SearchHit(candidates[0].chunk_id, float("nan"), 1, "fake")]
        raise AssertionError("Unexpected stub")


def test_rejects_unknown_reranker_evidence_in_fixed_pool(dataset):
    with pytest.raises(ValueError, match="unknown or duplicate candidate ID"):
        run(dataset, method="bm25", reranker=StubReranker("alien"),
            cutoffs=(1,), candidate_pool=3)


def test_rejects_duplicate_reranker_evidence(dataset):
    with pytest.raises(ValueError, match="unknown or duplicate candidate ID"):
        run(dataset, method="bm25", reranker=StubReranker("duplicate"),
            cutoffs=(1, 2), candidate_pool=3)


@pytest.mark.parametrize("mode", ["wrong-rank", "nonfinite"])
def test_rejects_invalid_reranker_rank_and_score(dataset, mode):
    with pytest.raises(ValueError, match="invalid ranks or nonfinite"):
        run(dataset, method="bm25", reranker=StubReranker(mode),
            cutoffs=(1,), candidate_pool=3)


def test_atomic_run_publishing_never_exposes_a_succeeded_partial_manifest(
    sample_files, tmp_path,
):
    output = tmp_path / "failed-run"
    with pytest.raises(ValueError, match="Out of range float values"):
        save_run(
            output, method="bm25", scores={"recall@20": float("nan")},
            per_query=[], corpus_path=sample_files[0],
            evaluation_path=sample_files[1],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed-run.partial-*"))


def test_atomic_run_rejects_existing_empty_output(sample_files, tmp_path):
    output = tmp_path / "prior-empty"
    output.mkdir()
    with pytest.raises(FileExistsError):
        save_run(
            output, method="bm25", scores={"recall@20": 0.0},
            per_query=[], corpus_path=sample_files[0],
            evaluation_path=sample_files[1],
        )
    assert not (output / "manifest.json").exists()


def test_prompt_template_fingerprint_invalidation(dataset):
    provider = MetadataContextualizer()
    chunk = dataset.chunks[0]
    key = context_cache_key(provider, "parent document", chunk)
    provider.prompt_fingerprint = "changed-template"
    assert context_cache_key(provider, "parent document", chunk) != key
    assert len(AnthropicContextualizer.prompt_fingerprint) == 64


def test_embedder_deduplicates_and_separates_document_and_query_cache(tmp_path):
    calls = []

    def embed(texts, **kwargs):
        calls.append((list(texts), kwargs))
        mode = kwargs.get("input_type")
        return SimpleNamespace(
            embeddings=[[1, 0.0] if mode == "document" else [0.0, 1]
                        for _ in texts]
        )

    fake_client = SimpleNamespace(embed=embed)
    cache = tmp_path / "embeddings.json"
    provider = VoyageEmbedder(
        "some-current-model", cache,
        document_input_type="document",
        query_input_type="query",
        batch_size=4,
        client=fake_client,
    )
    assert provider.embed_documents(["same", "same"]) == [[1, 0.0], [1, 0.0]]
    assert provider.embed_queries(["same"]) == [[0.0, 1]]
    assert len(calls) == 2
    assert calls[0] == (["same"], {"model": "some-current-model",
                                  "input_type": "document"})
    assert calls[1][1]["input_type"] == "query"
    assert len(json.loads(cache.read_text())) == 2
    provider2 = VoyageEmbedder(
        "some-current-model", cache,
        document_input_type="document", query_input_type="query",
        client=fake_client,
    )
    assert provider2.embed_documents(["same"]) == [[1, 0.0]]
    assert provider2.embed_queries(["same"]) == [[0.0, 1]]
    assert len(calls) == 2  # Both directional cache entries are reused


def test_rejects_nonfinite_provider_embeddings(tmp_path):
    def embed(texts, **kwargs):
        return SimpleNamespace(embeddings=[[float("nan"), 1]])
    provider = VoyageEmbedder(
        "m", tmp_path / "bad.json",
        client=SimpleNamespace(embed=embed),
    )
    with pytest.raises(ValueError, match="invalid vector"):
        provider.embed_documents(["bad"])
    assert not (tmp_path / "bad.json").exists()


def test_rejects_nonfinite_dot_products_and_vectors(dataset):
    with pytest.raises(ValueError, match="finite"):
        dot_product([float("inf"), 1], [0, 2])
    with pytest.raises(ValueError, match="finite"):
        ExactVector(dataset.chunks, [[float("nan")] for _ in dataset.chunks],
                    lambda _: [1])


def test_cohere_rejects_fractional_index_and_nan_relevance():
    def bad_index(**kwargs):
        return SimpleNamespace(results=[
            SimpleNamespace(index=0.5, relevance_score=0.9),
        ])
    provider = CohereReranker(
        "mock-model", client=SimpleNamespace(rerank=bad_index)
    )
    with pytest.raises(ValueError, match="integer"):
        provider.rerank(
            "Q", [SearchHit("a", 1, 1, "vec")],
            original_by_id={"a": "A"}, context_by_id=None, top_k=1,
        )
    provider.client = SimpleNamespace(rerank=lambda **kwargs: SimpleNamespace(
        results=[SimpleNamespace(index=0, relevance_score=float("nan"))]
    ))
    with pytest.raises(ValueError, match="finite"):
        provider.rerank(
            "Q", [SearchHit("a", 1, 1, "vec")],
            original_by_id={"a": "A"}, context_by_id=None, top_k=1,
        )


def test_embedding_input_mode_reaches_every_matrix_command(tmp_path):
    plans = plan_matrix(
        ["E01"], data_dir="data/raw/anthropic", out_root=str(tmp_path),
        embedding_input_mode="asymmetric", allow_paid_api=True,
    )
    command = plans[0]["command"]
    assert command[command.index("--embedding-input-mode") + 1] == "asymmetric"

    sweep = plan_model_sweep(
        ["model-a"], ["E01"], data_dir="fixture", out_root=str(tmp_path),
        embedding_input_mode="asymmetric", allow_paid_api=True,
    )
    cmd = sweep[0]["experiments"][0]["command"]
    assert cmd[cmd.index("--embedding-input-mode") + 1] == "asymmetric"
