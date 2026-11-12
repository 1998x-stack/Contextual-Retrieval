"""BEIR's document-level qrels are never silently treated as chunk annotations."""

import json

import pytest

from crlab.analysis.io import load_run
from crlab.cli import main
from crlab.datasets.anthropic import DatasetError
from crlab.datasets.beir import load_dataset, paths


@pytest.fixture
def beir_fixture(tmp_path):
    directory = tmp_path / "beir"
    (directory / "qrels").mkdir(parents=True)
    (directory / "corpus.jsonl").write_text("\n".join([
        json.dumps({"_id": "d1", "title": "Memory", "text": "Memory index is here."}),
        json.dumps({"_id": "d2", "title": "Search", "text": "Search index is elsewhere."}),
        json.dumps({"_id": "d3", "title": "Other", "text": "No result here."}),
    ]) + "\n")
    (directory / "queries.jsonl").write_text("\n".join([
        json.dumps({"_id": "q1", "text": "Where is memory index?"}),
        json.dumps({"_id": "q2", "text": "Where is search index?"}),
        json.dumps({"_id": "unjudged", "text": "Never included in qrels"}),
    ]) + "\n")
    (directory / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q1\td3\t0\n"
        "q2\td2\t2\n"
    )
    return directory


def test_beir_document_identity_and_granularity(beir_fixture):
    corpus, queries, qrels = paths(beir_fixture)
    dataset = load_dataset(corpus, queries, qrels)
    assert dataset.document_count == 3
    assert len(dataset.chunks) == 3
    assert len(dataset.queries) == 2
    assert dataset.queries[0].query_id == "beir:q1"
    assert dataset.queries[0].golden_chunk_ids == ("beir:d1:0",)
    assert dataset.queries[1].golden_chunk_ids == ("beir:d2:0",)
    assert dataset.chunk_by_id["beir:d1:0"].content == "Memory\n\nMemory index is here."


def test_beir_rejects_missing_gold_document(beir_fixture):
    corpus, queries, qrels = paths(beir_fixture)
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\tnot-exists\t1\n")
    with pytest.raises(DatasetError, match="unknown document"):
        load_dataset(corpus, queries, qrels)


def test_beir_rejects_duplicate_document_id(beir_fixture):
    corpus, queries, qrels = paths(beir_fixture)
    with corpus.open("a") as stream:
        stream.write(json.dumps({
            "_id": "d1", "title": "Duplicate", "text": "overlapping"
        }) + "\n")
    with pytest.raises(DatasetError, match="Duplicate"):
        load_dataset(corpus, queries, qrels)


def test_beir_rejects_query_without_positive_judgments(beir_fixture):
    corpus, queries, qrels = paths(beir_fixture)
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t0\n")
    with pytest.raises(DatasetError, match="No BEIR queries"):
        load_dataset(corpus, queries, qrels)


def test_beir_cli_data_validate_and_bm25_eval(beir_fixture, tmp_path, capsys):
    main(["data", "validate", "--dataset", "beir",
          "--data-dir", str(beir_fixture)])
    data = json.loads(capsys.readouterr().out)
    assert data["label_granularity"] == "document"
    assert data["documents"] == 3
    assert data["queries"] == 2
    root = tmp_path / "out"
    main(["eval", "run", "--dataset", "beir",
          "--data-dir", str(beir_fixture),
          "--experiment", "E02", "--out", str(root)])
    run = load_run(root)
    assert run.manifest["dataset_kind"] == "beir_document"
    assert run.manifest["label_granularity"] == "document"
    assert len(run.manifest["query_source_sha256"]) == 64
    assert run.manifest["evaluation_sha256"]
    assert run.metrics["recall@20"] == 1.0


def test_beir_refuses_chunk_context_without_evidence_labels(beir_fixture, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["eval", "run", "--dataset", "beir",
              "--data-dir", str(beir_fixture),
              "--context-dir", str(tmp_path / "dummy"),
              "--method", "bm25", "--out", str(tmp_path / "run")])
    assert error.value.code == 2


def test_beir_official_count_check_is_invalid(beir_fixture):
    with pytest.raises(SystemExit) as error:
        main(["data", "validate", "--dataset", "beir",
              "--expect-official", "--data-dir", str(beir_fixture)])
    assert error.value.code == 2


def test_beir_rejects_path_traversal_in_split(beir_fixture):
    with pytest.raises(ValueError, match="single filename"):
        paths(beir_fixture, split="../other")
