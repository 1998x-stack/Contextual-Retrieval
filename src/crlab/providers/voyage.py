"""Voyage embedding provider with direction-aware content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence


class VoyageEmbedder:
    """Explicitly control embedding input types; never mix query/document caches.

    Legacy Cookbook input_type=None behavior is the default. Modern models can
    use document_input_type='document', query_input_type='query' so identical
    raw text receives two different embeddings and cache entries.
    """

    def __init__(
        self,
        model: str,
        cache_path: str | Path,
        *,
        input_type: str | None = None,
        document_input_type: str | None = None,
        query_input_type: str | None = None,
        batch_size: int = 128,
        client: object | None = None,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("Embedding model ID must be nonempty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        allowed = {None, "document", "query"}
        if (input_type not in allowed or document_input_type not in allowed
                or query_input_type not in allowed):
            raise ValueError("input_type must be None, 'document' or 'query'")
        if input_type is not None and (document_input_type or query_input_type):
            raise ValueError("Use either legacy input_type or directional input types")
        self.model = model
        self.input_type = input_type
        self.document_input_type = document_input_type or input_type
        self.query_input_type = query_input_type or input_type
        self.batch_size = batch_size
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if not isinstance(self.cache, dict):
                raise ValueError("Voyage cache must be a JSON object")
        else:
            self.cache = {}
        if client is None:
            if not os.getenv("VOYAGE_API_KEY"):
                raise RuntimeError("Set VOYAGE_API_KEY before a billable API run")
            try:
                import voyageai
            except ImportError as exc:
                raise RuntimeError("Install provider: pip install -e '.[voyage]'") from exc
            client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.client = client

    def _key(self, text: str, mode: str | None) -> str:
        payload = json.dumps(
            [self.model, mode, text], ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _persist(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(self.cache, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            temporary.replace(self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _embed(self, texts: Sequence[str], *, mode: str | None) -> list[list[float]]:
        """Deduplicate repeated texts within one batch to avoid excess API charges."""
        results: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}
        for index, text in enumerate(texts):
            key = self._key(text, mode)
            if key in self.cache:
                results[index] = self.cache[key]
            elif key in missing:
                missing[key][1].append(index)
            else:
                missing[key] = (text, [index])
        unique = list(missing.items())
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start:start + self.batch_size]
            kwargs: dict[str, object] = {"model": self.model}
            if mode is not None:
                kwargs["input_type"] = mode
            response = self.client.embed([item[1][0] for item in batch], **kwargs)
            vectors = response.embeddings
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding provider returned incorrect batch size")
            parsed: list[list[float]] = []
            for vector in vectors:
                as_list = [float(number) for number in vector]
                if not as_list or any(not math.isfinite(v) for v in as_list):
                    raise ValueError("Embedding provider returned invalid vector values")
                parsed.append(as_list)
            if len({len(vector) for vector in parsed}) != 1:
                raise ValueError("Embedding provider returned inconsistent dimensions")
            for (key, (_, positions)), vector in zip(batch, parsed, strict=True):
                self.cache[key] = vector
                for position in positions:
                    results[position] = vector
            self._persist()
        if any(result is None for result in results):
            raise RuntimeError("Missing embedding after batch completion")
        return [result for result in results if result is not None]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, mode=self.document_input_type)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, mode=self.query_input_type)
