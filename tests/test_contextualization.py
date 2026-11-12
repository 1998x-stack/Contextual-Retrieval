"""Offline test suite for Phase 3 generation, caching, integrity and retrieval."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crlab.contextualization.artifacts import (
    generate_context_bundle, load_context_bundle,
)
from crlab.contextualization.cache import ContextCache, context_cache_key
from crlab.contextualization.engine import (
    AnthropicContextualizer, ContextResult, ContextUsage, MetadataContextualizer,
    contextualized_text,
)
from crlab.experiments.runner import run
from crlab.schemas import Chunk


class FakeContextualizer:
    strategy = "fake"
    model = "offline-mock"
    prompt_version = "test-v1"
    max_output_tokens = 100
    cache_enabled = True

    def __init__(self):
        self.calls = []

    def generate(self, document, chunk):
        self.calls.append((document, chunk.chunk_id))
        return ContextResult(f"Background for {chunk.chunk_id}", ContextUsage(
            input_tokens=10, output_tokens=4,
            cache_creation_input_tokens=3, cache_read_input_tokens=2,
        ))


class CapturingEmbedder:
    def __init__(self):
        self.documents = []

    def embed_documents(self, texts):
        self.documents = list(texts)
        return [[float("memory" in text.lower()), 1.0] for text in texts]

    def embed_queries(self, texts):
        return [[float("memory" in text.lower()), 1.0] for text in texts]


def test_anthropic_prompt_exact_shape_without_real_api():
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=" useful context ")],
            usage=SimpleNamespace(
                input_tokens=15, output_tokens=6,
                cache_creation_input_tokens=100, cache_read_input_tokens=200,
            ),
        )
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    provider = AnthropicContextualizer("explicit-model", client=client)
    chunk = Chunk("doc:0", "doc", 0, "return []")
    result = provider.generate("function search(): return []", chunk)
    assert result.text == "useful context"
    assert result.usage.cache_read_input_tokens == 200
    assert requests[0]["model"] == "explicit-model"
    assert requests[0]["max_tokens"] == 2048
    content = requests[0]["messages"][0]["content"]
    assert "<document>" in content[0]["text"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "<chunk>\nreturn []" in content[1]["text"]
    assert "improving search retrieval" in content[1]["text"]


def test_anthropic_prompt_cache_is_optional():
    def create(**kwargs):
        assert "cache_control" not in kwargs["messages"][0]["content"][0]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=1),
        )
    provider = AnthropicContextualizer(
        "model", client=SimpleNamespace(messages=SimpleNamespace(create=create)),
        cache_enabled=False,
    )
    assert provider.generate("whole", Chunk("d:0", "d", 0, "part")).text == "ok"


def test_local_cache_keys_change_with_source_and_prompt(tmp_path):
    provider = FakeContextualizer()
    chunk = Chunk("d:0", "d", 0, "original")
    key = context_cache_key(provider, "document", chunk)
    assert key != context_cache_key(provider, "document changed", chunk)
    assert key != context_cache_key(provider, "document", replace(chunk, content="new"))
    provider.prompt_version = "test-v2"
    assert key != context_cache_key(provider, "document", chunk)
    with ContextCache(tmp_path / "cache.sqlite3") as cache:
        assert cache.get(key) is None
        cache.put(key, ContextResult("context", ContextUsage(input_tokens=2)))
        assert cache.get(key).usage.input_tokens == 2


def test_generation_uses_cached_records_and_does_not_double_charge(sample_files, dataset, tmp_path):
    provider = FakeContextualizer()
    corpus, evaluation = sample_files
    cache = tmp_path / "cache.sqlite3"
    first = tmp_path / "first"
    second = tmp_path / "second"
    m1 = generate_context_bundle(corpus, evaluation, first, provider, cache_path=cache)
    assert m1["cache_misses"] == len(dataset.chunks)
    assert m1["cache_hits"] == 0
    assert m1["billed_api_usage"]["input_tokens"] == 30
    m2 = generate_context_bundle(corpus, evaluation, second, provider, cache_path=cache)
    assert m2["cache_misses"] == 0
    assert m2["cache_hits"] == len(dataset.chunks)
    assert m2["billed_api_usage"]["input_tokens"] == 0
    assert len(provider.calls) == len(dataset.chunks)
    assert m1["records_sha256"] == m2["records_sha256"]
    bundle = load_context_bundle(
        second, dataset, corpus_path=corpus, evaluation_path=evaluation
    )
    assert bundle.indexed_text("doc-A:0", "Alpha handles memory.").endswith(
        "\n\nAlpha handles memory."
    )


def test_offline_metadata_context_and_e05(sample_files, dataset, tmp_path):
    corpus, evaluation = sample_files
    out = tmp_path / "metadata"
    manifest = generate_context_bundle(
        corpus, evaluation, out, MetadataContextualizer(),
        cache_path=tmp_path / "local.sqlite3",
    )
    assert manifest["strategy"] == "metadata"
    assert manifest["billed_api_usage"]["input_tokens"] == 0
    bundle = load_context_bundle(
        out, dataset, corpus_path=corpus, evaluation_path=evaluation
    )
    assert "source-A" in bundle.contexts["doc-A:0"]
    assert "Alpha handles memory." not in bundle.contexts["doc-A:0"]
    scores, rows = run(
        dataset, method="bm25", context_by_id=bundle.contexts, candidate_pool=20
    )
    assert "recall@20" in scores and len(rows) == 2


def test_bundle_tampering_is_rejected(sample_files, dataset, tmp_path):
    corpus, evaluation = sample_files
    out = tmp_path / "metadata"
    generate_context_bundle(
        corpus, evaluation, out, MetadataContextualizer(),
        cache_path=tmp_path / "local.sqlite3",
    )
    records = out / "contexts.jsonl"
    records.write_text(records.read_text() + "{}\n")
    with pytest.raises(ValueError, match="records checksum"):
        load_context_bundle(out, dataset, corpus_path=corpus, evaluation_path=evaluation)


def test_bundle_source_digest_is_checked(sample_files, dataset, tmp_path):
    corpus, evaluation = sample_files
    out = tmp_path / "metadata"
    generate_context_bundle(
        corpus, evaluation, out, MetadataContextualizer(),
        cache_path=tmp_path / "local.sqlite3",
    )
    changed = tmp_path / "modified.json"
    changed.write_text(corpus.read_text() + "\n")
    with pytest.raises(ValueError, match="corpus checksum"):
        load_context_bundle(out, dataset, corpus_path=changed, evaluation_path=evaluation)


def test_completed_bundle_refuses_overwrite(sample_files, tmp_path):
    corpus, evaluation = sample_files
    out = tmp_path / "bundle"
    provider = FakeContextualizer()
    cache = tmp_path / "local.sqlite3"
    generate_context_bundle(corpus, evaluation, out, provider, cache_path=cache)
    with pytest.raises(FileExistsError):
        generate_context_bundle(corpus, evaluation, out, provider, cache_path=cache)
    assert len(provider.calls) == 3


def test_context_changes_indexed_input_not_original_evidence(dataset):
    embedder = CapturingEmbedder()
    contexts = {chunk.chunk_id: f"Context for {chunk.chunk_id}" for chunk in dataset.chunks}
    scores, rows = run(
        dataset, method="vector", embedder=embedder,
        context_by_id=contexts, candidate_pool=20,
    )
    assert "Context for doc-A:0\n\nAlpha handles memory." in embedder.documents
    assert len(rows) == 2
    assert "cookbook_pass@20" in scores
    with pytest.raises(ValueError, match="all original chunks"):
        run(dataset, method="bm25", context_by_id={"doc-A:0": "partial"},
            candidate_pool=20)


def test_cli_offline_context_end_to_end(sample_files, tmp_path, capsys):
    from crlab.cli import main
    corpus, _ = sample_files
    data_dir = corpus.parent
    out = tmp_path / "bundle"
    main(["context", "generate", "--provider", "metadata",
          "--data-dir", str(data_dir), "--out", str(out),
          "--cache-path", str(tmp_path / "local.sqlite3")])
    assert json.loads(capsys.readouterr().out)["manifest"]["chunk_count"] == 3
    main(["context", "validate", "--data-dir", str(data_dir),
          "--context-dir", str(out)])
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    results = tmp_path / "e05"
    main(["eval", "run", "--method", "bm25", "--data-dir", str(data_dir),
          "--context-dir", str(out), "--out", str(results)])
    run_manifest = json.loads((results / "manifest.json").read_text())
    assert run_manifest["context"]["strategy"] == "metadata"


def test_claude_api_permission_required(sample_files, tmp_path):
    from crlab.cli import main
    with pytest.raises(SystemExit) as err:
        main(["context", "generate", "--provider", "anthropic",
              "--model", "model-id", "--data-dir", str(sample_files[0].parent),
              "--out", str(tmp_path / "out")])
    assert err.value.code == 2


def test_empty_context_is_rejected():
    with pytest.raises(ValueError, match="empty context"):
        ContextResult("", ContextUsage())
    with pytest.raises(ValueError, match="Missing generated context"):
        contextualized_text("", "chunk")
