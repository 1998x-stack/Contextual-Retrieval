# Phase 4: executed Elasticsearch BM25 results

> Independent, observed results, **not** official model/reproduction figures.

## Data and environment

- Corpus: Anthropic Cookbook source pinned to ce8b2629, 90 documents, 737 chunks.
- Query set: 248 Golden Chunk queries.
- Corpus SHA-256: 70dbed1cb8a90a107b23742b1537051346860f841ec8b03734c1298fa4a34d46.
- Evaluation SHA-256: 7186e2bed15547a2ed178134876b29e4abb206bf0394facbe61d58bf835ee073.
- Engine: Elasticsearch 8.15.5; English analyzer and BM25; both original/contextual text fields.
- E02: original Chunk only.
- E05-control: original Chunk plus deterministic document IDs and Chunk index, *no model generation*.
- CI source: https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43146662930 (Elasticsearch job completed successfully; the separate unit test job on this earlier commit failed only on manifest backward compatibility, repaired in subsequent commit).
- Earlier local BM25 reference: https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43143772430

## Observed results

| Metric | Original local BM25 | Original ES BM25 E02 | Metadata-context ES BM25 |
|---|---:|---:|---:|
| ID Recall@5 | 58.1317% | 66.7579% | 66.7579% |
| ID Recall@10 | 67.3387% | 74.9568% | 74.9568% |
| ID Recall@20 | 75.5280% | 80.8948% | 80.8948% |
| NDCG@20 | 52.9302% | 60.4566% | 60.4566% |
| Hit@20 | 79.4355% | 85.0806% | 85.0806% |
| Cookbook Pass@20 | 75.6624% | 80.8948% | 80.8948% |

Metadata context made *no measured change* to the ES E02 rankings' reported metrics on this benchmark. This is not evidence that Claude chunk-specific context has no effect.

Differences between local and Elasticsearch include regex tokenizer versus English analyzer, scoring/mapping, and other implementation details; the table alone does not isolate which factor caused the improvement.

## Follow-on

Repeat with actual Claude-generated Context (E05) and with original/contextual Vector and Hybrid (E01/E03/E04/E06), then turn on Cohere in E07–E10 to measure its independent effect. Keep all experimental model IDs fixed within a comparison.
