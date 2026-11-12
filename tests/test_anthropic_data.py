import json

import pytest

from crlab.datasets.anthropic import DatasetError, chunk_key, load_dataset


def test_joins_golden_chunk_ids_to_corpus(dataset):
    assert dataset.document_count == 2
    assert len(dataset.chunks) == 3
    assert dataset.queries[0].golden_chunk_ids == ("doc-A:0",)
    assert dataset.queries[1].golden_contents == ("Gamma handles search.",)
    assert dataset.chunk_by_id["doc-A:1"].source_doc_id == "source-A"
    assert chunk_key("abc", "02") == "abc:2"


def test_rejects_invalid_golden_reference(sample_files):
    corpus, evaluation = sample_files
    row = json.loads(evaluation.read_text().splitlines()[0])
    row["golden_chunk_uuids"] = [["unknown-document", 99]]
    evaluation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing from corpus"):
        load_dataset(corpus, evaluation)


def test_rejects_conflicting_snapshot(sample_files):
    corpus, evaluation = sample_files
    row = json.loads(evaluation.read_text().splitlines()[0])
    row["golden_documents"][0]["chunks"][0]["content"] = "Not a real chunk"
    evaluation.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="differs"):
        load_dataset(corpus, evaluation)


def test_duplicate_chunks_fail(sample_files):
    corpus, evaluation = sample_files
    docs = json.loads(corpus.read_text())
    docs[0]["chunks"].append(dict(docs[0]["chunks"][0]))
    corpus.write_text(json.dumps(docs), encoding="utf-8")
    with pytest.raises(DatasetError, match="Duplicate chunk"):
        load_dataset(corpus, evaluation)
