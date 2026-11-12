# Dataset contracts

## Raw Anthropic format

Corpus is a JSON array of documents:
- doc_id: source-local document identifier;
- original_uuid: stable document UUID for qrels;
- content: full source document;
- chunks: list of objects containing chunk_id, original_index and content.

Evaluation is JSONL:
- query: natural-language query;
- golden_chunk_uuids: sequence of [original_uuid, original_index] pairs;
- golden_documents: inlined evidence snapshots with uuid and chunks[index, content].

## Normalized entities

- Chunk: chunk_id, document_id, index, content.
- Query: query_id, text, golden_chunk_ids, golden_contents.
- Dataset: tuple of chunks, tuple of queries, document count.

chunk_id is constructed as original_uuid + ':' + index. This avoids collisions from duplicate content across documents. The normalized query ID is q_000000 ... by input order; its stable meaning is within a dataset checksum.

## Validation behavior

Strict loader fails on duplicate identities, missing required data, golden references that do not map to corpus chunks, or conflicting content between the corpus and golden snapshot. Compare whitespace-stripped contents for golden snapshots, as in Cookbook scoring.

The source files remain unmodified and large raw data are gitignored. A run manifest records SHA-256 for both raw inputs. Do not leak query text to embedding index construction or use future sessions in historical memory retrieval benchmarks.

## Artifact structure

- manifest.json: method, model, data hashes, config, timestamp;
- metrics.json: per-K averages with explicit metric names;
- per_query.jsonl: query ID, gold chunk IDs, ranked candidates, and per-K metric values.

Data and metric names are versioned contracts. Adding document-level BEIR qrels requires a separate evaluation mode; it must not label every chunk of a relevant document as relevant.
