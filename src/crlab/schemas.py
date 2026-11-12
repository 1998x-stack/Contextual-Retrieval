"""Stable, minimal data contracts shared by all retrieval methods."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str  # original_uuid, not doc_id
    index: int
    content: str
    source_doc_id: str = ""

    @property
    def retrieval_text(self) -> str:
        return self.content


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str
    golden_chunk_ids: tuple[str, ...]
    golden_contents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Dataset:
    chunks: tuple[Chunk, ...]
    queries: tuple[Query, ...]
    document_count: int

    @property
    def chunk_by_id(self) -> dict[str, Chunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    rank: int
    source: str
