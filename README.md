# Contextual Retrieval Research Lab

A reproducible, testable research framework for [Anthropic's Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval).

This repository distinguishes **official reference numbers**, **independent reproduction runs**, and **extensions**. It never labels an unexecuted experiment as reproduced.

## Status

- Phase 0: official source registry, experiment audit, reference metrics.
- Phase 1: Anthropic codebase corpus/evaluation adapter with stable IDs; ID-based and Cookbook-compatible metrics.
- Phase 2: exact dot-product vector retrieval, local BM25, weighted rank fusion, CLI, JSONL ranking artifacts, deterministic tests.
- Phase 3: offline metadata control, official Claude contextualizer, prompt caching, resumable SQLite context cache, checksum-validated context bundles, E04–E06 retrieval integration.
- Phase 4: Elasticsearch English-analyzer BM25, Cohere rerank adapter, E01–E10 registry, and fixed-pool versus Cookbook rerank evaluation.
- Planned: cross-dataset evaluation, statistical significance, Agent Memory.

## Quickstart

Requires Python >= 3.11.

~~~bash
python -m pip install -e ".[dev]"
crlab data download --out data/raw/anthropic
crlab data validate --data-dir data/raw/anthropic
crlab eval run --data-dir data/raw/anthropic --method bm25 --out runs/bm25-local
pytest -q
~~~

The BM25 implementation in Phase 2 is **local Python BM25 with regex tokenization**; it is *not* Elasticsearch's English analyzer and will not reproduce official scores exactly. Vector / hybrid modes require Voyage credentials, the optional Voyage SDK, and explicit API cost approval by the operator:

~~~bash
python -m pip install -e ".[dev,voyage]"
export VOYAGE_API_KEY="your-key"
crlab eval run --data-dir data/raw/anthropic --method vector --embedding-model voyage-2 --allow-paid-api --out runs/vector-voyage
crlab eval run --data-dir data/raw/anthropic --method hybrid --embedding-model voyage-2 --allow-paid-api --out runs/hybrid-local
~~~

By default no API key is loaded or sent by the repository. Do not commit credentials or generated proprietary data. Raw datasets are downloaded from a pinned Anthropic Cookbook commit (ce8b2629) and SHA-256 validated and are intentionally not vendored.

## Phase 3: contextual retrieval

**No API key required** for deterministic metadata control:

~~~bash
crlab context generate --provider metadata --data-dir data/raw/anthropic --out runs/metadata-context
crlab context validate --data-dir data/raw/anthropic --context-dir runs/metadata-context
crlab eval run --data-dir data/raw/anthropic --method bm25 --context-dir runs/metadata-context --out runs/e05-metadata-local
~~~

For real Claude context generation (billable), install `.[anthropic]`, supply your own model ID, API key and explicit permission:

~~~bash
python -m pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY="your-key"
crlab context generate --provider anthropic --model YOUR_AVAILABLE_CLAUDE_MODEL_ID --allow-paid-api --out runs/claude-context
crlab context validate --context-dir runs/claude-context
~~~

To run contextual vector/hybrid (additional Voyage API charges):

~~~bash
crlab eval run --method vector --context-dir runs/claude-context --embedding-model voyage-2 --allow-paid-api --out runs/e04-context-vector
crlab eval run --method bm25 --context-dir runs/claude-context --out runs/e05-context-bm25-local
crlab eval run --method hybrid --context-dir runs/claude-context --embedding-model voyage-2 --allow-paid-api --out runs/e06-context-hybrid-local
~~~

`runs/` and `data/cache/` are gitignored. Context source manifests include SHA-256 hashes of original corpus, evaluation, complete context records, plus charged API token usage. A repeated generation with the same cache key makes no new model calls, but creates a new auditable bundle. E05/E06 local-BM25 results are **not** identical to Elasticsearch BM25 from the Cookbook.

See [Phase 3 design](docs/05-contextualization-engine.md) for data provenance, cache invalidation and limitations.

## Phase 4: Elasticsearch and Reranking

Install optional adapters, and start a local Elasticsearch 8 server (see docs/06-phase4.md):

~~~bash
python -m pip install -e ".[dev,elasticsearch,cohere]"
docker compose up -d elasticsearch
crlab matrix list
# E02, original BM25 on the official corpus, with Elasticsearch analyzer:
crlab eval run --experiment E02 --lexical-backend elasticsearch --es-index crlab_e02 --out runs/e02-es
# E05, contextual BM25, using an existing context bundle:
crlab eval run --experiment E05 --context-dir runs/claude-context --lexical-backend elasticsearch --es-index crlab_e05 --out runs/e05-es
~~~

Elasticsearch refuses to reuse an existing named index by default. Add `--es-recreate-index` only when deliberately replacing this experiment's index. The `crlab_` prefix is required.

For a *billable* reranking run, configure `COHERE_API_KEY`, `VOYAGE_API_KEY`, an explicit available rerank model ID and `--allow-paid-api`. Two modes exist: `--rerank-mode fixed` uses a common 150-candidate pool, while `--rerank-mode cookbook` uses a vector-only K×10 candidate pool for each K (Cookbook example semantics). E09/E10 require `fixed` mode.

~~~bash
crlab eval run --experiment E08 --context-dir runs/claude-context \
  --rerank-model YOUR_COHERE_MODEL_ID --rerank-mode cookbook \
  --allow-paid-api --out runs/e08-rerank
~~~

Batch planning is safe by default and makes no API calls. It shows blockers before execution:

~~~bash
crlab matrix run --ids E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 \
  --context-dir runs/claude-context \
  --rerank-model YOUR_COHERE_MODEL_ID \
  --allow-paid-api --out-root runs/full-ablation
# Add --execute only after reviewing the full plan, and with API keys set.
~~~

An executable offline two-experiment comparison is:

~~~bash
crlab matrix run --ids E02 E05 \
  --context-dir runs/metadata-context \
  --lexical-backend elasticsearch \
  --out-root runs/phase4-offline --execute
~~~

See [Phase 4 design and reproduction table](docs/06-phase4.md).

## Core research contracts

- The original 90-document/737-chunk/248-query figures are **reference metadata**, validated against downloaded files at execution time.
- Document identity uses original_uuid. Chunk identity uses (original_uuid, original_index), not chunk body text.
- Metrics report both ID-based retrieval and the Cookbook's original content-matching legacy score. The latter can treat duplicated chunk bodies as equivalent.
- Queries with no valid golden evidence fail validation by default.
- Vector scoring defaults to **dot product**, as in the Cookbook. Query and document embeddings use the same model and input configuration.
- Hybrid score uses Cookbook's weighted 1/rank algorithm (default 0.8 semantic, 0.2 BM25); candidate pool defaults to 150 from each retriever.
- Local BM25 is an independent baseline, **not** Elasticsearch parity.
- Full per-query rankings and a run manifest are persisted; no fabricated API or experimental results.

## Research docs

- [Roadmap](docs/00-roadmap.md)
- [Official reproduction audit](docs/01-reproduction-audit.md)
- [Data contracts](docs/02-data-contracts.md)
- [Evaluation protocol](docs/03-evaluation-protocol.md)
- [Experiment matrix](docs/04-experiment-matrix.md)
- [Contextualization engine](docs/05-contextualization-engine.md)
- [Phase 4: BM25 and reranking](docs/06-phase4.md)

## Provenance

Research source: Anthropic, *Introducing Contextual Retrieval*, September 19, 2024.
Code source: [anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/contextual-embeddings).
This is an independent implementation, not an Anthropic-endorsed project.
