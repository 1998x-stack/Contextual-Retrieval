# BEIR SciFact — document-level retrieval extension

## Source and provenance

Source: BEIR repository (https://github.com/beir-cellar/beir) and dataset wiki (https://github.com/beir-cellar/beir/wiki/Datasets-available).

- Canonical archive: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
- Upstream MD5: 5f7d1de60b170fc8027bb7898e2efca1.
- Observed downloaded ZIP SHA-256: 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165.
- Extracted raw Corpus SHA-256: dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6.
- Queries SHA-256: 8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313.
- test Qrels SHA-256: 0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c.

All hashes and metrics above were observed in GitHub Actions on 2026-11-12. The loader accepts other BEIR-format corpora provided by the user but does not download them automatically.

## Experiment protocol

- Dataset: 5,183 source documents, 300 test Queries with at least one positive judgment.
- Labels: BEIR documents, not source Chunks. We represent each document as exactly one retrieval unit with synthetic index 0. This is a mapping technique, **not** a claim to know which subsection is evidence.
- Evaluation: positive relevance grade maps to binary relevant/not relevant. Our Recall@K and binary NDCG@K differ from original BEIR graded NDCG when relevance grades vary.
- Retriever: local Python BM25 with Unicode regex tokenization, not Anserini or Elasticsearch BM25.
- All corpus documents indexed; unjudged Queries are excluded; Qrels refer to full document IDs.
- The CLI refuses to use a Chunk Context bundle with BEIR, because the available Qrels lack Chunk-level evidence spans.
- Data source files are gitignored; CI uploads result artifacts and the source manifest, not the entire dataset.

## Commands

~~~bash
crlab data beir-download --dataset scifact --out data/raw/beir
crlab data validate --dataset beir --data-dir data/raw/beir/scifact
crlab eval run --dataset beir --experiment E02 \
  --data-dir data/raw/beir/scifact --out runs/beir-scifact-e02
crlab report leaderboard --runs runs/beir-scifact-e02 \
  --metric recall@20 --out reports/generated/beir-scifact
~~~

For multi-model comparison, prepare a Voyage API key and review a dry-run model sweep first:

~~~bash
crlab matrix sweep --models MODEL_A MODEL_B --ids E01 E03 \
  --dataset beir --data-dir data/raw/beir/scifact --out-root runs/beir-sweep
~~~

Add --allow-paid-api and --execute only when paid inference is explicitly approved. Embedding APIs may reject overlength full scientific documents; the adapter does not silently truncate them.

## Reproduction boundary

This is an independent Phase 5 cross-domain experiment and does not reproduce Anthropic's Contextual Retrieval improvements. Chunk-Level Contextualization on scientific papers requires separate evidence-span annotation and is a future research task.
