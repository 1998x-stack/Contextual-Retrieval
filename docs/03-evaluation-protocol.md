# Evaluation and interpretation

For query q, let G be the set of golden chunk IDs and R_k the first k unique retrieval IDs.

- hit@k = 1 if G intersects R_k, otherwise 0.
- recall@k = |G intersect R_k| / |G|.
- all_evidence@k = 1 if G is a subset of R_k, otherwise 0.
- reciprocal_rank@k = 1 / rank(first relevant result) or 0.
- ndcg@k = DCG of ranked binary relevance / ideal DCG with min(|G|,k) relevant hits.
- cookbook_pass@k = for each golden **content string**, 1 if a retrieved content matches, divided by number of gold contents, averaged across all input queries. Repeated strings across separate chunks can lead to misleading matches.

Every mean uses the same query universe, and all metrics are in [0,1]. Display percentages only when presentation units are explicit.

Two comparisons are necessary:
1. compatible legacy content-matching score for provenance against official Cookbook;
2. stable-ID recall to avoid unrelated chunks matching solely due to identical content.

For inter-method comparisons, reuse the same Qrels, query ordering, candidate pool size, embedding inputs and cutoff values. Compare fixed-pool reranking separately from Cookbook's k-dependent pool.

Do not claim statistically significant improvements based on aggregate means alone. Future phases will add paired bootstrap confidence intervals and by-type failure analysis.
