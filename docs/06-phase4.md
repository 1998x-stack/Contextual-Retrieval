# Phase 4 — Elasticsearch BM25 and Cohere Reranking

## Reproduction boundary

The Anthropic Cookbook indexes two different text fields with Elasticsearch: the original Chunk in the content field and **generated context alone** in contextualized_content. Semantic retrieval instead embeds the concatenation of context, two line breaks, and original Chunk. Cohere reranks documents in the order original Chunk, two line breaks, "Context: " and generated context.

Those three text representations are intentionally *not interchangeable*. This phase preserves all three.

## Exact indexing and search contract

- Elasticsearch 8.15.5, English analyzer for both text fields, BM25 similarity, multi_match query over content and contextualized_content.
- Source ID uses original document UUID plus original Chunk index; original source text remains unchanged.
- Default hybrid candidate pool: 150 vector and 150 lexical candidates; fusion weights 0.8 / 0.2 and Cookbook's reciprocal-rank scoring.
- Explicitly named index beginning with crlab_ is mandatory. Already existing indices fail closed. The operator must deliberately use --es-recreate-index to replace the named index.
- The adapter bulk-indexes once and refreshes once, instead of refreshing for every query.
- For reproducibility, ES equal-score ties are resolved by chunk_id ascending, which differs slightly from unspecified Cookbook tie ordering.
- The baseline Python BM25 remains available for dependency-free comparison but is not Elasticsearch parity.

## Rerank strategy A: fixed candidate pool

Retriever emits 150 candidates (vector or hybrid). Cohere reranks the same candidates once per query to top 20; metrics at K = 5/10/20 use prefixes of this *same reranked list*. This is the scientifically controlled mode for E07–E10.

## Rerank strategy B: Cookbook mode

For each K separately, vector retrieval recalls K x 10 candidates; Cohere reranks them to K. For K = 5, 10, 20, these are 50/100/200 candidate pools. This mimics the Cookbook's example and differs from a fixed pool. Because the candidate sets differ, scores at different K need not derive from prefixes of one list.

Cookbook mode is only supported for vector retrieval (E07 and E08). E09 and E10 use fixed pool only.

Both strategies save the original candidate ranking and final reranking so unexpected gains/losses can be audited. Stable candidate IDs are taken from returned Cohere indices, not from matching document text.

## E01–E10 ablation matrix

| ID | Input | Retrieval | Rerank |
|---|---|---|---|
| E01 | Original | Vector | Off |
| E02 | Original | BM25 | Off |
| E03 | Original | Hybrid | Off |
| E04 | Contextual | Vector | Off |
| E05 | Contextual | BM25 | Off |
| E06 | Contextual | Hybrid | Off |
| E07 | Original | Vector | On |
| E08 | Contextual | Vector | On |
| E09 | Original | Hybrid | On |
| E10 | Contextual | Hybrid | On |

Matrix recipes are encoded in experiments/matrix.py and mirrored in configs/experiments/phase4-matrix.json. The CLI rejects missing context when an experiment requires it, as well as mismatched --method / --rerank-model flags.

## Reproduce offline Elasticsearch experiments

~~~bash
python -m pip install -e ".[dev,elasticsearch]"
docker compose up -d elasticsearch
crlab data download --out data/raw/anthropic
crlab data validate --data-dir data/raw/anthropic --expect-official
crlab matrix list

crlab eval run --experiment E02 \
  --lexical-backend elasticsearch \
  --es-index crlab_official_e02 \
  --out runs/e02-es

crlab context generate --provider metadata --out runs/metadata-context
crlab eval run --experiment E05 \
  --context-dir runs/metadata-context \
  --lexical-backend elasticsearch \
  --es-index crlab_metadata_e05 \
  --out runs/e05-es
~~~

These are real, non-API Elasticsearch retrieval runs, not Claude-based contextualization. CI reproduces them on the pinned 90-document / 737-chunk / 248-query corpus and uploads manifest, metrics and per-query artifacts.

## Paid provider experiment opt-in

You need a valid available Anthropic model ID, a valid Voyage embedding model ID, a valid Cohere Reranking model ID, the three corresponding API keys, and an explicit --allow-paid-api for each paid run. Historical API versions may no longer be available: never silently replace an unavailable 2024 model.

~~~bash
python -m pip install -e ".[anthropic,voyage,elasticsearch,cohere]"
export ANTHROPIC_API_KEY=YOUR_KEY
export VOYAGE_API_KEY=YOUR_KEY
export COHERE_API_KEY=YOUR_KEY

crlab context generate --provider anthropic \
  --model YOUR_AVAILABLE_CLAUDE_MODEL_ID \
  --allow-paid-api --out runs/claude-context

crlab eval run --experiment E08 \
  --context-dir runs/claude-context \
  --rerank-model YOUR_AVAILABLE_COHERE_MODEL_ID \
  --rerank-mode cookbook --allow-paid-api \
  --out runs/e08

crlab eval run --experiment E10 \
  --context-dir runs/claude-context \
  --lexical-backend elasticsearch --es-index crlab_e10 \
  --rerank-model YOUR_AVAILABLE_COHERE_MODEL_ID \
  --rerank-mode fixed --allow-paid-api \
  --out runs/e10
~~~

## How to interpret the results

- Legacy Cookbook Pass@K compares strings, whereas this lab also measures stable-ID Recall@K, Hit@K, MRR and NDCG.
- In the fixed mode, E09 versus E10 isolates contextualization under hybrid plus reranker; E03 versus E06 isolates contextualization under hybrid alone.
- Reranker alone has no way to rescue a gold Chunk missing from its initial candidate pool.
- Running Elasticsearch with source-only metadata is a weak deterministic control, not a surrogate for generated semantic context.
- Real Cohere and Voyage API results remain pending until an explicit paid-provider run; mocked API responses in tests are not benchmark data.

## Known differences and future work

1. ES tie-break by chunk_id is deterministic but differs from implicit shard-order ties in the Cookbook.
2. Elasticsearch 8.15.5 and Python SDK versions should be locked for strict reproducibility.
3. Context validation checks source hashes/completeness, not LLM factual correctness.
4. Phase 5 adds Paired Bootstrap confidence intervals, cost accounting, per-query category analysis, cross-model/cross-domain tests, and statistical interpretation.
