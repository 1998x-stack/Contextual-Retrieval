"""Generate and validate auditable, immutable context bundles.

A context is derived, potentially hallucinated retrieval metadata: never replace
or overwrite the original chunk source with generated text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from crlab.contextualization.cache import ContextCache, context_cache_key
from crlab.contextualization.engine import (
    ContextUsage, Contextualizer, contextualized_text, sha256_text,
)
from crlab.datasets.anthropic import load_dataset
from crlab.experiments.runner import file_hash
from crlab.schemas import Dataset


@dataclass(frozen=True)
class ContextBundle:
    contexts: dict[str, str]
    manifest: dict

    def indexed_text(self, chunk_id: str, original_content: str) -> str:
        return contextualized_text(self.contexts[chunk_id], original_content)


def _corpus_documents(corpus_path: Path, dataset: Dataset) -> dict[str, str]:
    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = {str(doc["original_uuid"]): doc["content"] for doc in raw}
    if len(documents) != dataset.document_count:
        raise ValueError("Document UUID mismatch in source corpus")
    if set(documents) != {chunk.document_id for chunk in dataset.chunks}:
        raise ValueError("Chunks and documents have different source UUIDs")
    return documents


def generate_context_bundle(
    corpus_path: str | Path,
    evaluation_path: str | Path,
    output_dir: str | Path,
    contextualizer: Contextualizer,
    *,
    cache_path: str | Path,
) -> dict:
    """Produce complete bundle, preserving cache across interrupted attempts.

    Writes a manifest and contexts.jsonl only after all chunks have generated
    successfully. A completed output directory is never silently overwritten.
    """
    corpus_path = Path(corpus_path)
    evaluation_path = Path(evaluation_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    dataset = load_dataset(corpus_path, evaluation_path)
    documents = _corpus_documents(corpus_path, dataset)
    records: list[dict] = []
    usage = ContextUsage()
    misses = 0
    hits = 0
    with ContextCache(cache_path) as cache:
        for chunk in dataset.chunks:
            document = documents[chunk.document_id]
            key = context_cache_key(contextualizer, document, chunk)
            result = cache.get(key)
            if result is None:
                result = contextualizer.generate(document, chunk)
                cache.put(key, result)
                misses += 1
                usage = usage + result.usage
            else:
                hits += 1
            records.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "original_sha256": sha256_text(chunk.content),
                "document_sha256": sha256_text(document),
                "cache_key": key,
                "generated_context": result.text,
                "usage_at_generation": asdict(result.usage),
            })

    # The output uses one stable row per original chunk (same order).
    jsonl_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in records
    )
    record_digest = hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": contextualizer.strategy,
        "model": contextualizer.model,
        "prompt_version": contextualizer.prompt_version,
        "prompt_fingerprint": getattr(
            contextualizer, "prompt_fingerprint", contextualizer.prompt_version
        ),
        "max_output_tokens": contextualizer.max_output_tokens,
        "prompt_cache_enabled": contextualizer.cache_enabled,
        "chunk_count": len(records),
        "cache_hits": hits,
        "cache_misses": misses,
        "billed_api_usage": asdict(usage),
        "corpus_sha256": file_hash(corpus_path),
        "evaluation_sha256": file_hash(evaluation_path),
        "records_sha256": record_digest,
        "notes": (
            "Generated context is derived retrieval metadata; original evidence is separate. "
            "Usage counts only new API generations, not reused local-cache records."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "contexts.jsonl").write_text(jsonl_text, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_context_bundle(
    path: str | Path,
    dataset: Dataset,
    *,
    corpus_path: str | Path,
    evaluation_path: str | Path,
) -> ContextBundle:
    """Fail closed on stale, partial, tampered or wrong-dataset context indexes."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    contexts_file = path / "contexts.jsonl"
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported context artifact version")
    if manifest["corpus_sha256"] != file_hash(corpus_path):
        raise ValueError("Context bundle corpus checksum mismatch")
    if manifest["evaluation_sha256"] != file_hash(evaluation_path):
        raise ValueError("Context bundle evaluation checksum mismatch")
    if manifest["records_sha256"] != file_hash(contexts_file):
        raise ValueError("Context bundle records checksum mismatch")
    by_id = dataset.chunk_by_id
    if manifest["chunk_count"] != len(by_id):
        raise ValueError("Context bundle chunk count mismatch")
    documents = _corpus_documents(Path(corpus_path), dataset)
    contexts: dict[str, str] = {}
    with contexts_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank context artifact row")
            record = json.loads(line)
            identity = record["chunk_id"]
            if identity in contexts or identity not in by_id:
                raise ValueError(f"Duplicate or unknown context ID: {identity}")
            chunk = by_id[identity]
            if record["document_id"] != chunk.document_id:
                raise ValueError(f"Context source mismatch for {identity}")
            if record["original_sha256"] != sha256_text(chunk.content):
                raise ValueError(f"Context original text mismatch for {identity}")
            if record["document_sha256"] != sha256_text(documents[chunk.document_id]):
                raise ValueError(f"Context parent document mismatch for {identity}")
            context = record["generated_context"]
            if not isinstance(context, str) or not context.strip():
                raise ValueError(f"Empty context for {identity}")
            contexts[identity] = context
    if set(contexts) != set(by_id):
        raise ValueError(f"Context artifact missing {len(set(by_id) - set(contexts))} chunks")
    return ContextBundle(contexts=contexts, manifest=manifest)
