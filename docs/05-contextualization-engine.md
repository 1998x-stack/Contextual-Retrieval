# Phase 3: Contextualization Engine

## Objectives

Produce a succinct, per-chunk search context given a chunk and its unmodified full document. Use the official Anthropic prompt layout and optional server-side prefix caching. Preserve original evidence: generated context is *derived metadata*, not verified truth.

## Modules and responsibilities

| File | Scope |
|---|---|
| src/crlab/contextualization/engine.py | Anthropic + deterministic metadata contextualizers, usage tracking, official prompt |
| src/crlab/contextualization/cache.py | SQLite local, content-addressed generation cache |
| src/crlab/contextualization/artifacts.py | Immutable context bundle, source hash checks, generation and validation |
| src/crlab/experiments/runner.py | Context-aware vector, lexical and hybrid indexing; original Chunk as evaluation evidence |
| src/crlab/cli.py | Context generate/validate, eval --context-dir |

## Reproduction workflow

### Offline metadata control (not a Claude result)

~~~bash
crlab data download --out data/raw/anthropic
crlab context generate \
  --provider metadata \
  --data-dir data/raw/anthropic \
  --out runs/metadata-context
crlab context validate --context-dir runs/metadata-context
crlab eval run \
  --method bm25 \
  --context-dir runs/metadata-context \
  --out runs/e05-metadata
~~~

This control is deterministic, contains only document ID/index metadata, and requires zero LLM tokens. CI runs a metadata bundle + indexed BM25 evaluation against the real official 90-document corpus to test the artifact pipeline.

### Claude API contextualization (explicit opt-in)

~~~bash
python -m pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=YOUR_KEY
crlab context generate \
  --provider anthropic \
  --model YOUR_AVAILABLE_CLAUDE_MODEL_ID \
  --allow-paid-api \
  --cache-path data/cache/contexts.sqlite3 \
  --out runs/claude-context
crlab context validate --context-dir runs/claude-context
~~~

The model is never silently substituted. The default max output tokens is 2048 to reproduce the official Notebook API parameter; the prompt asks for short explanations. Server-side cache uses the document block with cache_control ephemeral, followed by a chunk-specific block. The CLI --no-prompt-cache disables the server-side prompt cache while the local SQLite cache remains active.

### E04–E06: contextualized retrieval

~~~bash
# E04: contextualized embedding (paid Voyage API)
crlab eval run --method vector --context-dir runs/claude-context --allow-paid-api --out runs/e04

# E05: contextualized lexical (offline local BM25)
crlab eval run --method bm25 --context-dir runs/claude-context --out runs/e05

# E06: contextualized hybrid (paid Voyage API)
crlab eval run --method hybrid --context-dir runs/claude-context --allow-paid-api --out runs/e06
~~~

These options build index inputs from generated_context, a blank line, and original_content. Golden evaluation continues to compare unmodified original content/IDs. The same context bundle can feed multiple embedding and fusion runs.

## Artifact contract

~~~text
runs/claude-context/
├── manifest.json
└── contexts.jsonl
~~~

Record: chunk_id, document_id, original_sha256, document_sha256, cache_key, generated_context, usage_at_generation.

Manifest: schema_version, created_at, strategy, model, prompt_version, max_output_tokens, prompt_cache_enabled, chunk_count, cache_hits/misses, billed_api_usage, corpus_sha256, evaluation_sha256, records_sha256.

A loader rejects corrupted checksums, duplicate/unknown IDs, missing chunks, stale source documents, source content mismatches, and empty generated contexts. A completed bundle is never overwritten automatically.

## Cache policy and billing

- Local cache keys include document/Chunk text, identity, provider model, strategy, prompt version, max output tokens and cache-control setting.
- A failed run retains the already generated cache entries in SQLite; re-run with a new output directory.
- Only cache misses contribute to billed_api_usage in the generated manifest.
- Token usage is provided in raw units, not historical dollar pricing. Provider output variability may prevent byte-identical regeneration across runs.
- Provider credentials and generated artifacts are gitignored; do not commit sensitive source-derived context.

## Fidelity limits

1. E05/E06 local BM25 concatenates Context and Chunk with regex tokenization; Cookbook used Elasticsearch's English analyzer over two fields. An Elasticsearch parity adapter remains future work.
2. Generation is sequential to preserve easy caching/restart behavior; high-throughput concurrency, rate limiting and backoff remain future work.
3. Nonempty-output, completeness and checksum verification do not establish factual correctness. Context hallucinations need separate semantic evaluation.
4. No paid live Claude or Voyage API experiments were executed by CI. Do not label metadata control as official accuracy reproduction.
5. Original historical models may be retired. Supply an actually available model and label its results method reproduction, not exact reproduction.
