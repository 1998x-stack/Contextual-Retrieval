import pytest

from crlab.retrieval.bm25 import LocalBM25
from crlab.retrieval.fusion import fuse
from crlab.retrieval.vector import ExactVector, dot_product
from crlab.schemas import Chunk, SearchHit


def test_local_bm25_matches_rare_term(dataset):
    index = LocalBM25(dataset.chunks)
    results = index.search("Gamma search", 3)
    assert results[0].chunk_id == "doc-B:0"
    assert index.search("absentterm12345", 3) == []


def test_context_can_improve_keyword_recall():
    chunks = [Chunk("a:0", "a", 0, "return []"),
              Chunk("b:0", "b", 0, "unrelated result")]
    baseline = LocalBM25(chunks)
    contextual = LocalBM25(
        chunks, context_by_id={"a:0": "MemoryManager search filters low relevance"}
    )
    assert baseline.search("MemoryManager", 2) == []
    assert contextual.search("MemoryManager", 2)[0].chunk_id == "a:0"


def test_exact_dot_product_and_dimensionality():
    assert dot_product([1, 2], [3, 4]) == 11.0
    with pytest.raises(ValueError, match="dimensionality"):
        dot_product([1], [2, 3])


def test_vector_ranking(dataset):
    vectors = [[1, 0], [0, 1], [0.25, 0.75]]
    index = ExactVector(dataset.chunks, vectors, lambda _: [1, 0])
    assert index.search("memory", 1)[0].chunk_id == "doc-A:0"


def test_cookbook_fusion_weights():
    semantic = [SearchHit("a", 0.95, 1, "vector"),
                SearchHit("b", 0.8, 2, "vector")]
    lexical = [SearchHit("b", 2.7, 1, "bm25"),
               SearchHit("a", 1.1, 2, "bm25")]
    results = fuse([(0.8, semantic), (0.2, lexical)], 2)
    assert [x.chunk_id for x in results] == ["a", "b"]
    assert results[0].score == pytest.approx(0.8 + 0.2 / 2)
    assert results[1].score == pytest.approx(0.8 / 2 + 0.2)


def test_fusion_deduplicates_within_source():
    source = [SearchHit("a", 2, 1, "x"), SearchHit("a", 1, 2, "x")]
    assert fuse([(1.0, source)], 5)[0].score == 1.0
