"""Reproduce Anthropic's search-oriented, per-chunk context generation.

No API is invoked at import time. The model is caller-specified and billable
requests must be explicitly authorized by the CLI.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Protocol

from crlab.schemas import Chunk

PROMPT_VERSION = "anthropic-cookbook-v1"

# Deliberately preserve the wording and two-text-block structure of the
# Anthropic cookbook rather than silently introducing prompt improvements.
DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""
CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 0:
            raise ValueError("Token usage must not be negative")

    def __add__(self, other: ContextUsage) -> ContextUsage:
        return ContextUsage(**{
            key: getattr(self, key) + getattr(other, key)
            for key in asdict(self)
        })


@dataclass(frozen=True)
class ContextResult:
    text: str
    usage: ContextUsage

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Contextualizer produced empty context")


class Contextualizer(Protocol):
    strategy: str
    model: str
    prompt_version: str
    max_output_tokens: int
    cache_enabled: bool

    def generate(self, document: str, chunk: Chunk) -> ContextResult: ...


class MetadataContextualizer:
    """Deterministic offline control; *not* a substitute for Claude's results."""

    strategy = "metadata"
    model = "deterministic-metadata"
    prompt_version = "metadata-v1"
    max_output_tokens = 0
    cache_enabled = False

    def generate(self, document: str, chunk: Chunk) -> ContextResult:
        del document
        return ContextResult(
            text=(
                f"Source document: {chunk.source_doc_id or chunk.document_id}. "
                f"Document UUID: {chunk.document_id}. "
                f"Chunk index: {chunk.index}."
            ),
            usage=ContextUsage(),
        )


class AnthropicContextualizer:
    """Official Cookbook prompting with a cache-marked *full document* prefix.

    The provided model must exist in the user's API account. We never silently
    replace a retired historical model, or truncate the input document.
    """

    strategy = "anthropic"
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        model: str,
        *,
        max_output_tokens: int = 2048,
        cache_enabled: bool = True,
        client: object | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("An explicit Anthropic model ID is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.cache_enabled = cache_enabled
        if client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set ANTHROPIC_API_KEY for a billable context generation run")
            try:
                import anthropic  # optional dependency
            except ImportError as exc:
                raise RuntimeError("Install: pip install -e '.[anthropic]'") from exc
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.client = client

    def generate(self, document: str, chunk: Chunk) -> ContextResult:
        prefix: dict[str, object] = {
            "type": "text",
            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=document),
        }
        if self.cache_enabled:
            prefix["cache_control"] = {"type": "ephemeral"}

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            messages=[{
                "role": "user",
                "content": [
                    prefix,
                    {"type": "text", "text": CHUNK_CONTEXT_PROMPT.format(
                        chunk_content=chunk.content
                    )},
                ],
            }],
        )
        output = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        usage = getattr(response, "usage", None)
        return ContextResult(
            text=output,
            usage=ContextUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_creation_input_tokens=(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
        )


def contextualized_text(context: str, original_content: str) -> str:
    """Index text mirrors official contextual vector example: context + blank line + chunk."""
    if not context.strip():
        raise ValueError("Missing generated context")
    return f"{context}\n\n{original_content}"
