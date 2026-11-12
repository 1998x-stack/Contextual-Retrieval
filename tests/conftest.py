"""Small, fully offline fixture resembling the Anthropic schema."""

import json

import pytest

from crlab.datasets.anthropic import load_dataset


@pytest.fixture
def sample_files(tmp_path):
    docs = [
        {
            "doc_id": "source-A",
            "original_uuid": "doc-A",
            "content": "Alpha handles memory. Beta handles storage.",
            "chunks": [
                {"chunk_id": "c-a0", "original_index": 0, "content": "Alpha handles memory."},
                {"chunk_id": "c-a1", "original_index": 1, "content": "Beta handles storage."},
            ],
        },
        {
            "doc_id": "source-B",
            "original_uuid": "doc-B",
            "content": "Gamma handles search.",
            "chunks": [
                {"chunk_id": "c-b0", "original_index": 0, "content": "Gamma handles search."},
            ],
        },
    ]
    queries = [
        {
            "query": "Which component handles memory?",
            "golden_chunk_uuids": [["doc-A", 0]],
            "golden_documents": [
                {
                    "uuid": "doc-A",
                    "chunks": [{"index": 0, "content": "Alpha handles memory."}],
                }
            ],
        },
        {
            "query": "Which component handles search?",
            "golden_chunk_uuids": [["doc-B", 0]],
            "golden_documents": [
                {
                    "uuid": "doc-B",
                    "chunks": [{"index": 0, "content": "Gamma handles search."}],
                }
            ],
        },
    ]
    corpus = tmp_path / "codebase_chunks.json"
    eval_file = tmp_path / "evaluation_set.jsonl"
    corpus.write_text(json.dumps(docs), encoding="utf-8")
    eval_file.write_text(
        "\n".join(json.dumps(query) for query in queries) + "\n",
        encoding="utf-8",
    )
    return corpus, eval_file


@pytest.fixture
def dataset(sample_files):
    return load_dataset(*sample_files)
