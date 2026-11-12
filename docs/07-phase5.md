# Phase 5 — Statistical Benchmark and Error Analysis

## Research objective

Every conclusion must follow from an executed run artifact, not a historical paper reference value, a mocked SDK response or an inferred billing estimate. Analyze the same Query and Golden evidence under different E01–E10 configurations or models, estimate uncertainty and surface concrete regressions.

## Architecture

- analysis/io.py: verifies manifest schema, success status, full Query IDs and Golden Chunk identities, finite and bounded metrics, and aggregate-to-query consistency. A pair must have the same corpus_sha256 and evaluation_sha256 and exactly the same Query text/Golden IDs.
- analysis/statistics.py: seeded paired percentile Bootstrap for the mean Query-level quality difference, paired two-sided random sign-flip test and Holm step-down correction for a specified family of hypotheses. The Bootstrap supports resampling correlated Queries by user-supplied group_id.
- analysis/comparison.py: per-Query delta, winning/losing examples, original and candidate first-gold ranks, category breakdown. No automatic query-type guessing; unknown categories remain unlabeled.
- analysis/reporting.py: result JSON, query-level CSV/JSONL, Markdown, cross-model leaderboard grouped by dataset/Qrels identity, and cost-quality Pareto. Results are immutable by default (existing report paths are rejected).

## Core mathematics

For an identical Query universe Q, every Query contributes a paired difference:

    d(q) = score(candidate,q) - score(reference,q)

The reported paired mean is the arithmetic mean of d(q), expressed in percentage points (pp) when scores are fractions.

**Query-level paired bootstrap**: sample |Q| Query IDs with replacement, compute each resampled mean, repeat B times with a fixed seed, report percentile bounds at 2.5% / 97.5%.

**Cluster bootstrap**: if Queries from the same document or Session are correlated, supply a common group_id. Sample the same number of groups with replacement and take a Query-weighted mean over the resulting members. Resampling may vary the total count of Queries. The cluster CI cannot be interpreted as reliable with very few independent groups; a CI for 1 group is degenerate.

**Paired sign-flip test**: randomly multiply each paired delta (or each cluster's vector of deltas) by ±1 and recompute the mean. Report the fraction of null samples as or more extreme than observed, with a +1 Monte Carlo correction. It requires sign exchangeability within resampling units. A small p-value does not imply meaningful effect size or causality.

**Holm–Bonferroni**: for a fixed comparison family E01 vs E04, E03 vs E06, E09 vs E10 etc., adjust all resulting p-values together; record the number of comparisons. If you repeatedly tune a method against the same held-out Queries, post-hoc significance is not trustworthy.

The Bootstrap interval is a percentile interval rather than BCa. It is a practical baseline for this repository, not an asymptotically exact solution on tiny datasets. For a 248-Query benchmark, report both effect size and uncertainty, not only significance.

## CLI recipes

### Single pair

~~~bash
crlab report paired \
  --baseline runs/e03 \
  --candidate runs/e06 \
  --metric recall@20 \
  --resamples 2000 --permutations 5000 --seed 42 \
  --out reports/generated/e03-e06
~~~

Output has summary.json, per_query_delta.jsonl, per_query_delta.csv and report.md. Each Query records delta, label, first relevant rank (when the selected metric contains @K) and improved/regressed/unchanged state. The ten worst regressions and ten largest gains appear in summary.json.

### Multi-comparison family

~~~bash
crlab report batch \
  --baseline runs/e01 \
  --candidates runs/e03 runs/e04 runs/e07 \
  --metric recall@20 \
  --out reports/generated/multi
~~~

Output has batch.json, report.md and per-candidate folders. All candidates must share the baseline data and labels. The adjusted p-values cover *the candidate family supplied in this invocation*, not every possible search through prior experiments.

### Query annotations (optional, human-supplied)

Write a JSONL annotation file with records like:

~~~json
{"query_id":"q_000000","category":"exact_identifier","group_id":"document_001"}
{"query_id":"q_000001","category":"semantic_paraphrase","group_id":"document_001"}
~~~

Then pass --query-labels path/to/annotations.jsonl to paired or batch. These labels are illustrative and are NOT provided by Anthropic. A correlated Query group such as a single source document, file, Session, or user is the resampling unit when annotations are given. Unlabeled queries remain visible as unlabeled, with their own group.

### Cross-model leaderboard

~~~bash
crlab report leaderboard \
  --runs runs/voyage2-e01 runs/other-embedding-e01 runs/context-voyage-e04 \
  --metric recall@20 --out reports/generated/models
~~~

The tool groups datasets by the full raw Corpus SHA-256 plus Qrels SHA-256. It refuses to combine different Query/Gold definitions even when hashes appear the same. Scores from different corpora are presented in **separate groups**, not ranked together. Calling an available 2026 embedding model on the 2024 data is method reproduction or extension, not exact 2024 model replay.

### Pareto frontier

Create a real billing/accounting ledger with *comparable total USD costs for one defined workload*:

~~~json
{"runs":[
  {"path":"runs/e03","cost_usd":0.48,"source":"operator billing export; full workload"},
  {"path":"runs/e06","cost_usd":1.36,"source":"operator billing export; full workload"}
]}
~~~

All numbers in the example are **fictitious illustrations**, not actual prices. Required path values are interpreted relative to the current working directory. Ledger entries must be complete, finite, non-negative and include provenance. Missing costs cause a hard failure. Zero can only be explicitly assigned by the operator with source evidence.

~~~bash
crlab report pareto \
  --runs runs/e03 runs/e06 \
  --costs costs/my-total-costs.json \
  --metric recall@20 \
  --out reports/generated/cost-quality
~~~

A point A dominates B if A has at least as high quality and no greater cost, with at least one strict improvement. If you account for only Voyage query tokens while omitting context generation and reranking, you have **not** measured end-to-end Pareto cost.

## What does not yet exist

- Chunk-level evidence labels for additional public domains (the BEIR SciFact document-level adapter is implemented and verified; it deliberately refuses Contextual Chunk Qrels).
- A complete real API-billing-to-USD pipeline across Anthropic, Voyage and Cohere; cost ledger is externally supplied.
- Paid model sweep results: a dry-run-safe multi-model planner is implemented, but its actual paid API evaluations still need operator credentials and budget approval.
- Failure-category gold annotations for official 248 Queries.
- Claim of real Claude or Cohere retrieval benefit in CI. CI uses official data but only offline metadata context and BM25.

## CI acceptance

CI runs a real Elasticsearch E02 (original) and E05 (metadata control), then runs paired and batch reports on the two actual artifacts using 600 Bootstrap samples and 600 permutations, plus a same-data leaderboard. This verifies the entire reporting pipeline without credentials. The observed outcome is an empirical control, *not* official Contextual Retrieval accuracy.


## Phase 5 extension: cross-domain and model sweeps

The BEIR adapter accepts a standard corpus.jsonl, queries.jsonl and qrels/test.tsv with a strict document-level label contract. The SciFact download is pinned to the publisher's published MD5, and the actual source archive SHA-256 plus raw data hashes are recorded.

~~~bash
crlab data beir-download --out data/raw/beir
crlab data validate --dataset beir --data-dir data/raw/beir/scifact
crlab eval run --dataset beir --experiment E02 \
  --data-dir data/raw/beir/scifact --out runs/scifact-local-bm25
~~~

Document retrieval uses one full source document as one Chunk object with index 0. Scores named cookbook_pass@K are an implementation artifact for compatibility, not the historical Anthropic scoring of short source Chunks. BEIR grades above 0 become binary relevance for our Recall, NDCG and MRR; published graded BEIR metrics are **not directly comparable** to this binary NDCG.

For Embedding comparisons, the model sweep command creates a unique output directory and Elasticsearch index namespace per model ID:

~~~bash
crlab matrix sweep --models MODEL_A MODEL_B --ids E01 E03 \
  --out-root runs/model-sweep --dataset beir \
  --data-dir data/raw/beir/scifact
~~~

This prints a full plan and marks paid calls blocked until --allow-paid-api is added. The plan is never executed without --execute. Model names and API keys are supplied by the operator; historical model snapshots are not silently substituted.

The runnable SciFact and Anthropic official CI jobs are independent. Do **not** pool the scores or treat the 5,183-document SciFact setting as a Chunk Contextualization test.
