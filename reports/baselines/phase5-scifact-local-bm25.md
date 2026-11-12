# Executed Phase 5 baseline: BEIR SciFact local BM25

## Source

CI execution: https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43148718191

Artifact: https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43148718191/artifacts/13912443720

Official SciFact archive MD5: 5f7d1de60b170fc8027bb7898e2efca1, ZIP SHA-256: 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165.

Data: 5,183 full documents; 300 test Queries; each document is one indexed retrieval unit.

Retrieval algorithm: independent local Python BM25 with regex tokenizer.

| Metric | K=5 | K=10 | K=20 |
|---|---:|---:|---:|
| Document Recall@K | 72.5944% | 79.0944% | 83.2556% |
| Binary NDCG@K | 63.9408% | 66.1676% | 67.2736% |
| Hit@K | 75.00% | 81.00% | 85.00% |
| MRR@K | 61.9667% | 62.7618% | 63.0309% |

These numbers were computed in the actual CI job. They are NOT historical BEIR leaderboard scores, NOT Elasticsearch BM25 results, and NOT Claude Contextual Retrieval results.

Important interpretation: SciFact qrels are document-level, and our relevance is binarized (all positive grades map to relevant), so our NDCG must not be compared directly to BEIR official graded NDCG. An original full document is not an isolated Chunk needing additional contextualization. This benchmark primarily stress-tests the reusable Dataset/Query/Ranking/Evaluation infrastructure on a separate domain.

Future research: original versus contextual scientific *chunks* only after separately validated evidence-span or golden-chunk annotations exist.
