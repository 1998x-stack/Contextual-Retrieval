"""Load the public Anthropic codebase evaluation, preserving original boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from crlab.schemas import Chunk, Dataset, Query


class DatasetError(ValueError):
    """Input cannot be normalized to a valid corpus with chunk-level qrels."""


def chunk_key(original_uuid: str, original_index: int) -> str:
    """Canonical ID for a chunk; original text is deliberately not part of its identity."""
    return f"{original_uuid}:{int(original_index)}"


def load_dataset(
    corpus_path: str | Path,
    evaluation_path: str | Path,
    *,
    strict: bool = True,
) -> Dataset:
    """Read official JSON/JSONL without recomputing chunk boundaries or golden labels."""
    with Path(corpus_path).open("r", encoding="utf-8") as stream:
        raw_documents = json.load(stream)
    if not isinstance(raw_documents, list) or not raw_documents:
        raise DatasetError("Expected nonempty JSON list of documents")

    chunks: list[Chunk] = []
    chunk_lookup: dict[str, Chunk] = {}
    seen_docs: set[str] = set()

    for document in raw_documents:
        doc_uuid = str(document["original_uuid"])
        if not doc_uuid or doc_uuid in seen_docs:
            raise DatasetError(f"Missing/duplicate document UUID: {doc_uuid}")
        seen_docs.add(doc_uuid)
        if not isinstance(document["content"], str):
            raise DatasetError(f"Document {doc_uuid} must have complete text")
        for raw in document["chunks"]:
            index = int(raw["original_index"])
            identity = chunk_key(doc_uuid, index)
            if identity in chunk_lookup:
                raise DatasetError(f"Duplicate chunk: {identity}")
            chunk = Chunk(
                chunk_id=identity,
                document_id=doc_uuid,
                index=index,
                content=str(raw["content"]),
                source_doc_id=str(document.get("doc_id", "")),
            )
            chunks.append(chunk)
            chunk_lookup[identity] = chunk
    if not chunks:
        raise DatasetError("Corpus has no chunks")

    queries: list[Query] = []
    with Path(evaluation_path).open("r", encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            text = raw["query"]
            if not isinstance(text, str) or not text.strip():
                raise DatasetError(f"Empty query at row {row_number}")

            evidence = {
                str(doc["uuid"]): {
                    int(chunk["index"]): str(chunk["content"]).strip()
                    for chunk in doc["chunks"]
                }
                for doc in raw["golden_documents"]
            }

            gold_ids: list[str] = []
            gold_contents: list[str] = []
            for doc_uuid, index_value in raw["golden_chunk_uuids"]:
                doc_uuid, index = str(doc_uuid), int(index_value)
                identity = chunk_key(doc_uuid, index)
                chunk = chunk_lookup.get(identity)
                snapshot = evidence.get(doc_uuid, {}).get(index)
                if chunk is None or snapshot is None:
                    if strict:
                        raise DatasetError(
                            f"Golden reference {identity} in row {row_number} "
                            "missing from corpus or golden_documents"
                        )
                    continue
                if chunk.content.strip() != snapshot:
                    if strict:
                        raise DatasetError(
                            f"Golden snapshot content differs for {identity}, row {row_number}"
                        )
                    # Keep original golden snapshot for faithful legacy evaluation.
                gold_ids.append(identity)
                gold_contents.append(snapshot)

            if not gold_ids:
                raise DatasetError(f"Query row {row_number} has no valid gold evidence")
            if len(set(gold_ids)) != len(gold_ids):
                raise DatasetError(f"Duplicate gold ID in query row {row_number}")
            queries.append(
                Query(
                    query_id=f"q_{len(queries):06d}",
                    text=text,
                    golden_chunk_ids=tuple(gold_ids),
                    golden_contents=tuple(gold_contents),
                )
            )
    if not queries:
        raise DatasetError("Evaluation has no nonempty queries")
    return Dataset(chunks=tuple(chunks), queries=tuple(queries), document_count=len(seen_docs))
