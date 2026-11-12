# Official reproduction audit (Phase 0)

## Primary sources

- Anthropic article: https://www.anthropic.com/research/contextual-retrieval
- Cookbook code: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb
- Corpus: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/data/codebase_chunks.json
- Eval set: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/data/evaluation_set.jsonl
- Original technical appendix linked by the article.

## Official reported figures, not reproduction outputs

The article reported average failed retrieval at K=20: 5.7% (embeddings), 3.7% (contextual embeddings), 2.9% (+ contextual BM25), 1.9% (+ reranking). Relative error-rate reductions are ~35%, ~49%, ~67% against 5.7%. These are **not** end-to-end QA accuracy increases.

The Cookbook summary contains these historical percentages:
- Baseline vector: Pass@5 80.92, Pass@10 87.15, Pass@20 90.06.
- Contextual vector: Pass@5 88.12, Pass@10 92.34, Pass@20 94.29.
- Contextual vector + reranking: Pass@5 92.15, Pass@10 95.26, Pass@20 97.45.

The current Cookbook's hybrid intermediate outputs and summary table are inconsistent; preserve both as distinct references. Do not silently select one.

## Observed implementation semantics in the Cookbook

- Source corpus entries: doc_id, original_uuid, content, chunks.
- Chunks: chunk_id, original_index, content.
- Evaluation rows: query, golden_chunk_uuids, golden_documents.
- Each golden_chunk_uuids entry contains (document UUID, index); golden_documents records have uuid and indexed chunks.
- Vector: Voyage voyage-2 embedding of chunks and queries, NumPy dot-product scores, descending ranking.
- Original evaluator: compare stripped chunk **content**, not stable IDs; multiple gold chunks => average matched fraction, denominator includes all queries even when no gold content resolved.
- Context generation: whole document with cache-control ephemeral, followed by target chunk and succinct search-oriented request.
- Hybrid: semantic and Elasticsearch BM25 each recall 150; weight 0.8/0.2, reciprocal rank 1/(rank+1) in zero-based indices. Notebook tie-break uses (score, doc_id, original_index) descending. This differs from smooth-RRF k=60.
- Cookbook reranking: contextual **vector** candidates, often top 10*K, not full hybrid; Cohere rerank-english-v3.0.
- Cookbook BM25 uses Elasticsearch English analyzer over original and contextualized content fields; the local baseline in this project deliberately does not claim analyzer parity.

## Limitations

- Source repository is not a frozen historical 2024 release. Pin a Cookbook commit hash in future exact-reproduction manifests.
- The article's full cross-domain query annotations are not established as openly downloadable by this project.
- Historical model snapshots may be retired. No substitute can be called exact reproduction.
- API outputs can drift. Cache immutable contextual strings, embeddings and rankings before comparisons.
- A 248-query benchmark may overfit from repeated parameter tuning; add separate development datasets.

## Source checks

Download script records SHA-256 hashes at runtime. Do not substitute manually copied sample data for the public official corpus.
