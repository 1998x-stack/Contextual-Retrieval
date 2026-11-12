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

## Verified data provenance (2026-11-12)

- Source repository commit: anthropics/claude-cookbooks @ ce8b2629efe8b24048b127b4c79af1e9cbab93d4.
- Corpus: 90 documents, 737 chunks; SHA-256 70dbed1cb8a90a107b23742b1537051346860f841ec8b03734c1298fa4a34d46 (1,126,046 bytes).
- Qrels: 248 queries; SHA-256 7186e2bed15547a2ed178134876b29e4abb206bf0394facbe61d58bf835ee073 (3,571,229 bytes).
- A previous planning description cited *9* documents: this was corrected after GitHub Actions downloaded and validated the actual JSON array.
- The CLI now verifies the pinned source checksum and rejects unexpected bytes.
