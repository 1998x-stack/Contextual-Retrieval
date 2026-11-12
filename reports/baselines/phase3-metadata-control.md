# Phase 3: offline metadata context versus original BM25

This is an **executed experiment**, not a reproduction of the Claude-powered approach.

## Source and controls

- Data: Anthropic Cookbook pinned corpus (90 docs, 737 original chunks), 248 golden queries.
- Corpus SHA-256: 70dbed1cb8a90a107b23742b1537051346860f841ec8b03734c1298fa4a34d46.
- Evaluation SHA-256: 7186e2bed15547a2ed178134876b29e4abb206bf0394facbe61d58bf835ee073.
- Retrieval: local regex-tokenized BM25 (not Elasticsearch).
- Gold: stable original UUID + chunk index; all 248 queries included.
- Context: deterministic source document / UUID / chunk index metadata, **no LLM**, **zero billed API tokens**.
- Top-K = 5, 10, 20; no reranker.

GitHub Actions:
- [Original local BM25 baseline](https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43143772430).
- [Metadata-context E05 control](https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43144521521).

## Observed metrics (fractions reported here as percentages)

| Metric | Original local BM25 | Metadata + local BM25 | Difference |
|---|---:|---:|---:|
| ID Recall@5 | 58.1317% | 58.1317% | +0.0000 pp |
| ID Recall@10 | 67.3387% | 66.5323% | -0.8064 pp |
| ID Recall@20 | 75.5280% | 75.5280% | +0.0000 pp |
| Hit@20 | 79.4355% | 79.4355% | +0.0000 pp |
| NDCG@20 | 52.9302% | 52.1073% | -0.8229 pp |
| Cookbook Pass@20 | 75.6624% | 75.6624% | +0.0000 pp |

Conclusion is deliberately narrow: adding bare source IDs and positions did not improve Recall@20 in this specific lexical control, and it slightly reduced Recall@10 and NDCG@20. It does not establish the expected performance of a Claude-generated, chunk-specific semantic context.

The generated context metadata is not a verified fact about the intended semantic meaning of any chunk. Future Phase 3 paid-provider runs should be compared to E01/E03 using the same corpus and metrics. The Elasticsearch parity check remains part of Phase 4.
