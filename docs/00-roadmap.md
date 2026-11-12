# Implementation roadmap

## Research objective

Reconstruct each part of Anthropic's Contextual Retrieval evaluation without conflating the original blog, the evolving Cookbook, and our independent reproduction.

| Phase | Scope | Exit criteria |
|---|---|---|
| 0 | Audit and provenance | Reference results sourced, version/algorithm deviations documented |
| 1 | Data + metrics | Validate corpus/queries/qrels; stable ID and legacy metrics tested |
| 2 | Baselines | Vector, BM25, hybrid; per-query artifacts and CLI |
| 3 | Contextualization | Chunk-specific prompt, original-context preservation, cache and usage |
| 4 | Reranking | Vector/contextual/vector+BM25 candidate parity, fixed-pool tests |
| 5 | Broad benchmark | Multiple models, corpora, statistical confidence intervals |
| 6 | Agent Memory | Temporal-aware and session-aware queries and evidence validity |

## Tracks

1. **Historical exact reproduction**: only when original dataset, model revision, indexing details, retrieval logic and metrics can be recovered.
2. **Method reproduction**: faithful algorithm with explicit deviations, e.g. current Voyage API and local BM25.
3. **Extension**: new datasets/models/methods, separately reported.

## Priority order

P0: make source files immutable; implement source adapters and dataset checks. P1: independently test metrics, BM25 and rank fusion. P2: support paid API runs with opt-in and caches. P3: compare reranking and cross-domain performance.

## Evidence policy

Never copy the official reported scores into a reproduced metrics file. A run can be marked successful only after the CLI produces a manifest, query-level rankings, and metrics from actual data. Include git commit, data hashes, provider model IDs, ranking configuration and run timestamp.
