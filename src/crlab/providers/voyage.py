"""Opt-in Voyage embedding adapter with persistent content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


class VoyageEmbedder:
    """The same API handles documents and queries, as in Anthropic's Cookbook.

    input_type=None intentionally avoids adding query/document prefixes, preserving
    older Cookbook embedding semantics. Newer models might require a specific type.
    """

    def __init__(
        self,
        model: str,
        cache_path: str | Path,
        *,
        input_type: str | None = None,
        batch_size: int = 128,
    ) -> None:
        if not os.getenv("VOYAGE_API_KEY"):
            raise RuntimeError("Set VOYAGE_API_KEY explicitly before a paid API run")
        try:
            import voyageai  # optional, do not require for BM25 or tests
        except ImportError as exc:
            raise RuntimeError("Install provider: pip install -e '.[voyage]'") from exc
        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model = model
        self.input_type = input_type
        self.batch_size = batch_size
        self.cache_path = Path(cache_path)
        self.cache = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists()
            else {}
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def _key(self, text: str) -> str:
        payload = json.dumps(
            [self.model, self.input_type, text],
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _persist(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.cache), encoding="utf-8")
        temporary.replace(self.cache_path)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = self._key(text)
            if key in self.cache:
                results[index] = self.cache[key]
            else:
                missing.append((index, key, text))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            kwargs: dict[str, object] = {"model": self.model}
            if self.input_type is not None:
                kwargs["input_type"] = self.input_type
            response = self.client.embed(
                [text for _, _, text in batch], **kwargs
            )
            vectors = response.embeddings
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding provider returned incorrect batch size")
            for (index, key, _), vector in zip(batch, vectors, strict=True):
                as_list = [float(number) for number in vector]
                self.cache[key] = as_list
                results[index] = as_list
            self._persist()  # preserve progress for retries
        if any(result is None for result in results):
            raise RuntimeError("Missing embedding after batch completion")
        return [result for result in results if result is not None]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)
