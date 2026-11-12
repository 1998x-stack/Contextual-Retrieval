"""Elasticsearch BM25 adapter reproducing Cookbook's English analyzer and fields.

Deliberate differences: stable chunk IDs in Elasticsearch _id, explicit
fresh-index management, deterministic sort on chunk_id when scores tie,
refresh after bulk load (not on every query). Read docs/06-phase4.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from crlab.schemas import Chunk, SearchHit


class ElasticsearchBM25:
    """Standalone lexical index with explicit lifecycle and content provenance.

    Existing indices are rejected unless recreate=True; this prevents a prior
    dataset from silently polluting reproducibility. Recreate is restricted
    to the one explicitly supplied index name, not wildcard patterns.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        index_name: str,
        endpoint: str = "http://localhost:9200",
        context_by_id: Mapping[str, str] | None = None,
        recreate: bool = False,
        client: object | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot create a lexical index with zero chunks")
        if not index_name or not index_name.startswith("crlab_"):
            raise ValueError("Elasticsearch index must start with 'crlab_'")
        if any(char in index_name for char in "*?\\, /#"):
            raise ValueError("Elasticsearch index may not contain wildcards or spaces")
        if context_by_id is not None and set(context_by_id) != {
            chunk.chunk_id for chunk in chunks
        }:
            raise ValueError("Generated contexts must match the complete indexed corpus")
        if client is None:
            try:
                from elasticsearch import Elasticsearch
            except ImportError as exc:
                raise RuntimeError(
                    "Install Elasticsearch adapter: pip install -e '.[elasticsearch]'"
                ) from exc
            client = Elasticsearch(endpoint)
        self.client = client
        self.index_name = index_name
        self.chunks = tuple(chunks)
        self.context_by_id = context_by_id or {}
        self._build(recreate=recreate)

    def _build(self, *, recreate: bool) -> None:
        if self.client.indices.exists(index=self.index_name):
            if not recreate:
                raise ValueError(
                    f"Index {self.index_name!r} already exists. "
                    "Pass --es-recreate-index to explicitly rebuild it."
                )
            self.client.indices.delete(index=self.index_name)

        self.client.indices.create(
            index=self.index_name,
            settings={
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,
            },
            mappings={"properties": {
                "content": {"type": "text", "analyzer": "english"},
                "contextualized_content": {"type": "text", "analyzer": "english"},
                "doc_id": {"type": "keyword", "index": False},
                "original_index": {"type": "integer", "index": False},
                "chunk_id": {"type": "keyword"},
            }},
        )
        try:
            from elasticsearch.helpers import bulk
        except ImportError as exc:
            raise RuntimeError(
                "Install Elasticsearch adapter: pip install -e '.[elasticsearch]'"
            ) from exc
        actions = (
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": chunk.chunk_id,
                "_source": {
                    "content": chunk.content,
                    "contextualized_content": self.context_by_id.get(chunk.chunk_id, ""),
                    "doc_id": chunk.source_doc_id,
                    "original_index": chunk.index,
                    "chunk_id": chunk.chunk_id,
                },
            }
            for chunk in self.chunks
        )
        succeeded, _ = bulk(self.client, actions, raise_on_error=True)
        if succeeded != len(self.chunks):
            raise RuntimeError(
                f"Indexed {succeeded} of {len(self.chunks)} expected chunks"
            )
        self.client.indices.refresh(index=self.index_name)

    def search(self, query: str, k: int) -> list[SearchHit]:
        if k <= 0:
            return []
        response = self.client.search(
            index=self.index_name,
            size=k,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                },
            },
            sort=[{"_score": {"order": "desc"}}, {"chunk_id": {"order": "asc"}}],
        )
        found = response["hits"]["hits"]
        hits = [
            SearchHit(
                chunk_id=item["_source"]["chunk_id"],
                score=float(item["_score"]),
                rank=rank,
                source="elasticsearch_bm25",
            )
            for rank, item in enumerate(found, 1)
        ]
        if len({x.chunk_id for x in hits}) != len(hits):
            raise ValueError("Elasticsearch returned duplicate chunk identifiers")
        return hits
