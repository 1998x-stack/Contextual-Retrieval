"""Document-aware chunk context generation with separate, auditable source evidence."""

from crlab.contextualization.engine import (
    AnthropicContextualizer,
    ContextResult,
    ContextUsage,
    MetadataContextualizer,
)

__all__ = ["AnthropicContextualizer", "ContextResult", "ContextUsage", "MetadataContextualizer"]
