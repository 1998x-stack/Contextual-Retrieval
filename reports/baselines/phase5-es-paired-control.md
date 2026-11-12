# Executed Phase 5 analysis: Elasticsearch metadata-context control

Sources: actual GitHub Actions run https://github.com/1998x-stack/Contextual-Retrieval/actions/runs/43148398222

The compared inputs are the same Anthropic Cookbook corpus (90 original documents / 737 Chunks) and the same 248 queries. E02 indexes original content. E05-control indexes original content plus deterministic source-document metadata; this is NOT Claude-generated context.

| Statistic | Measured value |
|---|---:|
| Baseline E02 ID Recall@20 | 80.8947773% |
| Metadata E05 ID Recall@20 | 80.8947773% |
| Per-query paired mean difference | 0.0000 pp |
| Query-level 95% bootstrap percentile interval | [0.0000, 0.0000] pp |
| Paired sign-flip two-sided p-value | 1.0000 |
| Improved / regressed / unchanged | 0 / 0 / 248 |

The CI uses seed 42, 600 paired Bootstrap resamples, and 600 random sign-flip permutations. All 248 Query-level Recall@20 values match between both conditions. The 0-width interval reflects identical observed scores, not general certainty that metadata can never help.

This finding does not establish whether Claude-generated chunk-specific context improves retrieval. For that, compare E03 to E06 and E09 to E10 with the same source dataset and actual paid model outputs.
