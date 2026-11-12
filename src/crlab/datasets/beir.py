"""BEIR document-level adapter: one search unit per corpus document.

BEIR qrels are DOCUMENT-level labels. No implicit conversion to chunk-relevance
is allowed; this adapter indexes the entire document as a single retrieval unit.
The output is a method-extension benchmark, not the official Anthropic corpus.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from crlab.datasets.anthropic import DatasetError
from crlab.schemas import Chunk, Dataset, Query


def paths(root: str | Path, *, split: str = "test") -> tuple[Path, Path, Path]:
    root = Path(root)
    if not split or "/" in split or "\\" in split or split in (".", ".."):
        raise ValueError("BEIR split must be a single filename component")
    return root / "corpus.jsonl", root / "queries.jsonl", root / "qrels" / f"{split}.tsv"


def _read_jsonl(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record.get(key)
            if not isinstance(item_id, str) or not item_id.strip():
                raise DatasetError(f"Missing BEIR ID at {path}:{lineno}")
            if item_id in rows:
                raise DatasetError(f"Duplicate BEIR ID {item_id} in {path}")
            rows[item_id] = record
    if not rows:
        raise DatasetError(f"No BEIR records found at {path}")
    return rows


def load_dataset(
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
) -> Dataset:
    corpus = _read_jsonl(Path(corpus_path), "_id")
    queries = _read_jsonl(Path(queries_path), "_id")
    chunks: list[Chunk] = []
    for document_id, record in corpus.items():
        title, content = record.get("title", ""), record.get("text", "")
        if not isinstance(title, str) or not isinstance(content, str):
            raise DatasetError(f"BEIR text fields must be strings: {document_id}")
        text = "\n\n".join(item for item in (title.strip(), content.strip()) if item)
        if not text:
            raise DatasetError(f"Empty BEIR document: {document_id}")
        chunk_id = f"beir:{document_id}:0"
        chunks.append(Chunk(
            chunk_id=chunk_id, document_id=f"beir:{document_id}",
            index=0, content=text, source_doc_id=document_id,
        ))
    by_source = {chunk.source_doc_id: chunk for chunk in chunks}
    relevance: dict[str, set[str]] = {}
    with Path(qrels_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise DatasetError(f"Invalid BEIR qrels header; expected {expected}")
        for lineno, row in enumerate(reader, 2):
            qid, docid = row["query-id"], row["corpus-id"]
            if qid not in queries:
                raise DatasetError(f"Qrels reference unknown query {qid} at line {lineno}")
            if docid not in by_source:
                raise DatasetError(f"Qrels reference unknown document {docid} at line {lineno}")
            try:
                grade = int(row["score"])
            except (ValueError, TypeError) as exc:
                raise DatasetError(f"Invalid qrel score at line {lineno}") from exc
            if grade > 0:
                relevance.setdefault(qid, set()).add(docid)
    normalized: list[Query] = []
    for query_id, record in queries.items():
        if query_id not in relevance:
            continue  # Unjudged/nonrelevant queries are excluded, never called a miss.
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(f"Empty BEIR query: {query_id}")
        relevant_ids = sorted(relevance[query_id])
        normalized.append(Query(
            query_id=f"beir:{query_id}",
            text=text,
            golden_chunk_ids=tuple(by_source[docid].chunk_id for docid in relevant_ids),
            golden_contents=tuple(by_source[docid].content for docid in relevant_ids),
        ))
    if not normalized:
        raise DatasetError("No BEIR queries have positive relevance labels")
    return Dataset(tuple(chunks), tuple(normalized), document_count=len(chunks))
