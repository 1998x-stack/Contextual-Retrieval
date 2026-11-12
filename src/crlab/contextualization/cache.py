"""Content-addressed SQLite cache for expensive chunk context generation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from crlab.contextualization.engine import ContextResult, Contextualizer, ContextUsage
from crlab.schemas import Chunk

CACHE_SCHEMA = 1


def context_cache_key(contextualizer: Contextualizer, document: str, chunk: Chunk) -> str:
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "model": contextualizer.model,
        "strategy": contextualizer.strategy,
        "prompt_version": contextualizer.prompt_version,
        "prompt_fingerprint": getattr(
            contextualizer, "prompt_fingerprint", contextualizer.prompt_version
        ),
        "max_output_tokens": contextualizer.max_output_tokens,
        "prompt_cache": contextualizer.cache_enabled,
        "document_uuid": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "document_text": document,
        "chunk_text": chunk.content,
    }
    # No raw source text is written to the cache *key*.
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class ContextCache:
    """SQLite transactional writes and lookups; call close() when finished."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL
            )
        """)
        self.connection.commit()

    def get(self, key: str) -> ContextResult | None:
        result = self.connection.execute(
            "SELECT result_json FROM contexts WHERE key = ?", (key,)
        ).fetchone()
        if result is None:
            return None
        record = json.loads(result[0])
        return ContextResult(
            text=record["text"],
            usage=ContextUsage(**record["usage"]),
        )

    def put(self, key: str, result: ContextResult) -> None:
        value = json.dumps({
            "text": result.text,
            "usage": asdict(result.usage),
        }, ensure_ascii=False)
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO contexts (key, result_json) VALUES (?, ?)",
                (key, value),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ContextCache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
