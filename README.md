# Contextual Retrieval Research Lab

A reproducible, testable research framework for [Anthropic's Contextual Retrieval](https://www.anthropic.com/research/contextual-retrieval).

This repository distinguishes **official reference numbers**, **independent reproduction runs**, and **extensions**. It never labels an unexecuted experiment as reproduced.

## Status

- Phase 0: official source registry, experiment audit, reference metrics.
- Phase 1: Anthropic codebase corpus/evaluation adapter with stable IDs; ID-based and Cookbook-compatible metrics.
- Phase 2: exact dot-product vector retrieval, local BM25, weighted rank fusion, CLI, JSONL ranking artifacts, deterministic tests.
- Planned: API-driven contextual generation and prompt caching, Elasticsearch parity, Cohere reranking, cross-dataset evaluation, Agent Memory.

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
crlab eval run --data-dir data/raw/anthropic --method vector --embedding-model voyage-2 --out runs/vector-voyage
crlab eval run --data-dir data/raw/anthropic --method hybrid --embedding-model voyage-2 --out runs/hybrid-local
~~~

By default no API key is loaded or sent by the repository. Do not commit credentials or generated proprietary data. Raw datasets are downloaded from a pinned Anthropic Cookbook commit (ce8b2629) and SHA-256 validated and are intentionally not vendored.

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

## Provenance

Research source: Anthropic, *Introducing Contextual Retrieval*, September 19, 2024.
Code source: [anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/contextual-embeddings).
This is an independent implementation, not an Anthropic-endorsed project.
