# Experiment matrix

| ID | Chunk text | Retriever | Reranker | Track |
|---|---|---|---|---|
| E01 | original | vector | no | official method |
| E02 | original | BM25 | no | independent lexical baseline |
| E03 | original | vector+BM25 | no | method reproduction |
| E04 | contextualized | vector | no | Phase 3 |
| E05 | contextualized | BM25 | no | Phase 3 |
| E06 | contextualized | vector+BM25 | no | Phase 3 |
| E07 | original | vector | yes | Phase 4 |
| E08 | contextualized | vector | yes | Phase 4 |
| E09 | original | hybrid | yes | Phase 4 |
| E10 | contextualized | hybrid | yes | Phase 4 |

E02/E03 currently use local BM25 and do NOT match official Elasticsearch semantics. An Elasticsearch English analyzer adapter is required for higher fidelity. E01 needs a Voyage model, credentials and billable requests; no default proxy or synthetic official metrics.

Every row must report Pass@5/10/20, ID Recall@5/10/20, latency, model versions, index creation cost and query cost when available.

Planned cross-dataset matrix: SciFact, NFCorpus, FiQA, selected chunk-labeled codebase QA, and long-term trajectory Qrels. Document-level qrels must remain document-level.
