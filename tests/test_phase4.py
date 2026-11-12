"""Phase 4 tests: deterministic adapters, rerank protocols and experiment recipes."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from crlab.cli import main
from crlab.experiments.matrix import PHASE4_EXPERIMENTS, experiment_spec
from crlab.experiments.runner import run
from crlab.reranking.cohere import CohereReranker, cookbook_rerank_document
from crlab.retrieval.elasticsearch import ElasticsearchBM25
from crlab.schemas import SearchHit


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float("memory" in x.lower()), float("search" in x.lower()), 1] for x in texts]

    def embed_queries(self, texts):
        return [[float("memory" in x.lower()), float("search" in x.lower()), 1] for x in texts]


class FakeReranker:
    model = "fake"
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, *, original_by_id, context_by_id, top_k):
        self.calls.append((len(candidates), top_k, context_by_id is not None))
        return [
            SearchHit(hit.chunk_id, 1 / (rank + 1), rank + 1, "fake")
            for rank, hit in enumerate(reversed(candidates[:top_k]))
        ]


def test_registry_has_exactly_ten_valid_combinations():
    assert len(PHASE4_EXPERIMENTS) == 10
    assert experiment_spec("e01").method == "vector"
    assert experiment_spec("E05").use_context
    assert experiment_spec("E10").rerank and experiment_spec("E10").method == "hybrid"
    assert {id for id, s in PHASE4_EXPERIMENTS.items() if s.rerank} == {
        "E07", "E08", "E09", "E10"
    }
    with pytest.raises(ValueError, match="Unknown experiment"):
        experiment_spec("E11")


def test_fixed_pool_rerank_once_per_query_preserves_candidates(dataset):
    reranker = FakeReranker()
    scores, rows = run(dataset, method="vector", embedder=FakeEmbedder(),
                       reranker=reranker, candidate_pool=3, cutoffs=(1, 2))
    assert len(reranker.calls) == len(dataset.queries)
    assert all(call == (3, 2, False) for call in reranker.calls)
    assert all(len(row["candidate_rankings"]) == 3 for row in rows)
    assert all(len(row["rankings"]) == 2 for row in rows)
    assert "recall@1" in scores and "recall@2" in scores


def test_cookbook_mode_reranks_each_cutoff_independently(dataset):
    reranker = FakeReranker()
    scores, rows = run(dataset, method="vector", embedder=FakeEmbedder(),
                       reranker=reranker, rerank_mode="cookbook",
                       candidate_pool=3, cutoffs=(1, 2))
    assert len(reranker.calls) == 2 * len(dataset.queries)
    assert [call[1] for call in reranker.calls[:2]] == [1, 2]
    assert set(rows[0]["rankings_by_cutoff"]) == {"1", "2"}
    assert len(rows[0]["candidate_rankings_by_cutoff"]["1"]) == 3
    assert "recall@2" in scores


def test_cookbook_mode_rejects_hybrid(dataset):
    with pytest.raises(ValueError, match="requires a vector"):
        run(dataset, method="bm25", reranker=FakeReranker(),
            rerank_mode="cookbook", candidate_pool=3, cutoffs=(1,))


def test_hybrid_fixed_mode_accepts_context(dataset):
    contexts = {c.chunk_id: f"From {c.source_doc_id}" for c in dataset.chunks}
    reranker = FakeReranker()
    scores, rows = run(dataset, method="hybrid", embedder=FakeEmbedder(),
                       reranker=reranker, context_by_id=contexts,
                       candidate_pool=3, cutoffs=(1, 2))
    assert len(rows) == 2 and "recall@2" in scores
    assert all(call[2] for call in reranker.calls)


def test_cohere_adapter_preserves_original_candidate_ids():
    captured = []
    def respond(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(results=[
            SimpleNamespace(index=1, relevance_score=0.99),
            SimpleNamespace(index=0, relevance_score=0.5),
        ])
    r = CohereReranker("explicit-model", client=SimpleNamespace(rerank=respond))
    hits = r.rerank("query", [
        SearchHit("a", 3, 1, "vector"),
        SearchHit("b", 2, 2, "vector"),
    ], original_by_id={"a": "Original A", "b": "Original B"},
       context_by_id={"a": "Context A", "b": "Context B"}, top_k=2)
    assert [h.chunk_id for h in hits] == ["b", "a"]
    assert hits[0].score == 0.99
    assert captured[0]["documents"] == [
        "Original A\n\nContext: Context A",
        "Original B\n\nContext: Context B",
    ]
    assert cookbook_rerank_document("raw", None) == "raw"


def test_cohere_rejects_duplicate_or_out_of_range_indices():
    r = CohereReranker(
        "model", client=SimpleNamespace(rerank=lambda **kw: SimpleNamespace(
            results=[SimpleNamespace(index=7, relevance_score=0.3)]
        ))
    )
    with pytest.raises(ValueError, match="invalid candidate index"):
        r.rerank("q", [SearchHit("a", 1, 1, "vector")],
                 original_by_id={"a": "raw"}, context_by_id=None, top_k=1)
    with pytest.raises(ValueError, match="Duplicate"):
        r.rerank("q", [SearchHit("a", 1, 1, "vector"),
                        SearchHit("a", 1, 2, "vector")],
                 original_by_id={"a": "raw"}, context_by_id=None, top_k=1)


def test_elasticsearch_index_uses_two_english_analyzed_fields(monkeypatch, dataset):
    recorded = []
    helpers = ModuleType("elasticsearch.helpers")
    def bulk(client, actions, raise_on_error=True):
        entries = list(actions)
        recorded.extend(entries)
        return len(entries), []
    helpers.bulk = bulk
    monkeypatch.setitem(sys.modules, "elasticsearch.helpers", helpers)

    class IndexAPI:
        def __init__(self):
            self.created = []
            self.refreshes = []
        def exists(self, *, index): return False
        def create(self, **kwargs): self.created.append(kwargs)
        def refresh(self, *, index): self.refreshes.append(index)

    class Client:
        def __init__(self): self.indices = IndexAPI(); self.queries = []
        def search(self, **kwargs):
            self.queries.append(kwargs)
            return {"hits": {"hits": [
                {"_source": {"chunk_id": "doc-A:0"}, "_score": 2.3}
            ]}}

    client = Client()
    context = {c.chunk_id: f"Context for {c.chunk_id}" for c in dataset.chunks}
    index = ElasticsearchBM25(dataset.chunks, index_name="crlab_test",
                              context_by_id=context, client=client)
    assert len(recorded) == 3
    assert recorded[0]["_id"] == "doc-A:0"
    assert recorded[0]["_source"]["content"] == "Alpha handles memory."
    assert recorded[0]["_source"]["contextualized_content"] == "Context for doc-A:0"
    props = client.indices.created[0]["mappings"]["properties"]
    assert props["content"]["analyzer"] == "english"
    assert props["contextualized_content"]["analyzer"] == "english"
    assert client.indices.created[0]["settings"]["similarity"]["default"]["type"] == "BM25"
    assert client.indices.refreshes == ["crlab_test"]
    assert index.search("memory", 10)[0].chunk_id == "doc-A:0"
    assert client.queries[0]["query"]["multi_match"]["fields"] == [
        "content", "contextualized_content"
    ]


def test_elasticsearch_fails_closed_on_existing_or_unsafe_index(dataset):
    class Client:
        indices = SimpleNamespace(exists=lambda **kw: True)
    with pytest.raises(ValueError, match="already exists"):
        ElasticsearchBM25(dataset.chunks, index_name="crlab_test", client=Client())
    with pytest.raises(ValueError, match="start with"):
        ElasticsearchBM25(dataset.chunks, index_name="production", client=Client())


def test_cli_lists_matrix_and_runs_offline_e02(sample_files, tmp_path, capsys):
    main(["matrix", "list"])
    output = capsys.readouterr().out
    assert "E01" in output and "E10" in output
    path = tmp_path / "e02"
    main(["eval", "run", "--experiment", "E02",
          "--data-dir", str(sample_files[0].parent), "--out", str(path)])
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["experiment_id"] == "E02"
    assert manifest["lexical_engine"] == "local_python_bm25"


def test_cli_rejects_incomplete_experiments_without_api(sample_files, tmp_path):
    root = str(sample_files[0].parent)
    for experiment in ("E04", "E08"):
        with pytest.raises(SystemExit) as err:
            main(["eval", "run", "--experiment", experiment,
                  "--data-dir", root, "--out", str(tmp_path / experiment)])
        assert err.value.code == 2
    with pytest.raises(SystemExit) as err:
        main(["eval", "run", "--experiment", "E02", "--rerank-model", "model",
              "--data-dir", root, "--out", str(tmp_path / "invalid")])
    assert err.value.code == 2
