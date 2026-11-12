"""crlab CLI: download, validate, run, compare."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from crlab import __version__
from crlab.analysis.comparison import compare_runs, load_query_labels
from crlab.analysis.io import load_run
from crlab.analysis.reporting import (
    build_leaderboard, build_pareto, save_batch_report,
    save_json_report, save_paired_report,
)
from crlab.contextualization.artifacts import generate_context_bundle, load_context_bundle
from crlab.contextualization.engine import AnthropicContextualizer, MetadataContextualizer
from crlab.datasets.anthropic import load_dataset
from crlab.datasets.beir import load_dataset as load_beir_dataset, paths as beir_paths
from crlab.datasets.beir_download import download_scifact
from crlab.experiments.runner import file_hash, run, save_run
from crlab.experiments.matrix import (
    PHASE4_EXPERIMENTS, experiment_spec, plan_matrix, plan_model_sweep,
)

OFFICIAL_BASE = (
    "https://raw.githubusercontent.com/anthropics/claude-cookbooks/"
    "ce8b2629efe8b24048b127b4c79af1e9cbab93d4/capabilities/contextual-embeddings/data/"
)
RAW_FILENAMES = ("codebase_chunks.json", "evaluation_set.jsonl")
OFFICIAL_SHA256 = {
    "codebase_chunks.json": "70dbed1cb8a90a107b23742b1537051346860f841ec8b03734c1298fa4a34d46",
    "evaluation_set.jsonl": "7186e2bed15547a2ed178134876b29e4abb206bf0394facbe61d58bf835ee073",
}


def _files(data_dir: Path) -> tuple[Path, Path]:
    return (data_dir / RAW_FILENAMES[0], data_dir / RAW_FILENAMES[1])


def download(data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename in RAW_FILENAMES:
        target = data_dir / filename
        if target.exists() and not force:
            raise FileExistsError(f"{target} exists; use --force to overwrite")
        request = urllib.request.Request(
            OFFICIAL_BASE + filename, headers={"User-Agent": "crlab/0.1"}
        )
        temporary = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as writer:
                    while block := response.read(1024 * 1024):
                        writer.write(block)
            actual_hash = file_hash(temporary)
            if actual_hash != OFFICIAL_SHA256[filename]:
                raise ValueError(f"Official source checksum mismatch for {filename}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        manifest[filename] = {
            "url": OFFICIAL_BASE + filename,
            "sha256": file_hash(target),
            "bytes": target.stat().st_size,
        }
    (data_dir / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def validate(
    data_dir: Path, *, expect_official: bool = False,
    dataset_format: str = "anthropic", beir_split: str = "test",
) -> None:
    if dataset_format == "beir":
        if expect_official:
            raise ValueError("--expect-official is only valid for the Anthropic dataset")
        corpus, queries, qrels = beir_paths(data_dir, split=beir_split)
        dataset = load_beir_dataset(corpus, queries, qrels)
        print(json.dumps({
            "format": "beir_document",
            "label_granularity": "document",
            "documents": dataset.document_count,
            "chunks": len(dataset.chunks),
            "queries": len(dataset.queries),
            "corpus_sha256": file_hash(corpus),
            "query_source_sha256": file_hash(queries),
            "evaluation_sha256": file_hash(qrels),
            "split": beir_split,
        }, indent=2))
        return
    corpus, eval_file = _files(data_dir)
    dataset = load_dataset(corpus, eval_file)
    counts = {
        "documents": dataset.document_count,
        "chunks": len(dataset.chunks),
        "queries": len(dataset.queries),
        "corpus_sha256": file_hash(corpus),
        "evaluation_sha256": file_hash(eval_file),
    }
    if expect_official and (
        counts["documents"], counts["chunks"], counts["queries"]
    ) != (90, 737, 248):
        raise ValueError(f"Official reference count mismatch: {counts}")
    if expect_official and (
        counts["corpus_sha256"] != OFFICIAL_SHA256[RAW_FILENAMES[0]]
        or counts["evaluation_sha256"] != OFFICIAL_SHA256[RAW_FILENAMES[1]]
    ):
        raise ValueError("Official reference data checksum mismatch")
    print(json.dumps(counts, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    if args.experiment is not None:
        spec = experiment_spec(args.experiment)
        if args.method is not None and args.method != spec.method:
            raise ValueError(f"{spec.experiment_id} requires --method {spec.method}")
        if spec.use_context != (args.context_dir is not None):
            raise ValueError(f"{spec.experiment_id}: context bundle required={spec.use_context}")
        if spec.rerank != (args.rerank_model is not None):
            raise ValueError(f"{spec.experiment_id}: reranker required={spec.rerank}")
        method = spec.method
    else:
        if args.method is None:
            raise ValueError("--method or --experiment is required")
        method = args.method
    if args.rerank_model is not None and not args.allow_paid_api:
        raise RuntimeError("Reranking is billable: pass --allow-paid-api")
    if args.rerank_mode == "cookbook" and (args.rerank_model is None or method != "vector"):
        raise ValueError("Cookbook rerank mode needs vector and --rerank-model")
    source_queries_path = None
    if args.dataset == "beir":
        if args.context_dir is not None:
            raise ValueError(
                "BEIR has document-level Qrels; contextual chunk index requires "
                "an evidence-labeled chunk dataset, not fabricated Chunk Qrels"
            )
        corpus, source_queries_path, eval_file = beir_paths(
            Path(args.data_dir), split=args.beir_split
        )
        dataset = load_beir_dataset(corpus, source_queries_path, eval_file)
    else:
        corpus, eval_file = _files(Path(args.data_dir))
        dataset = load_dataset(corpus, eval_file)
    contexts = None
    context_manifest = None
    if args.context_dir is not None:
        bundle = load_context_bundle(
            args.context_dir, dataset, corpus_path=corpus, evaluation_path=eval_file
        )
        contexts = bundle.contexts
        context_manifest = bundle.manifest
    reranker = None
    if args.rerank_model is not None:
        from crlab.reranking.cohere import CohereReranker
        reranker = CohereReranker(args.rerank_model)
    embedder = None
    if method in {"vector", "hybrid"}:
        if not args.allow_paid_api:
            raise RuntimeError(
                "Vector/Hybrid may make billable API requests; pass --allow-paid-api"
            )
        from crlab.providers.voyage import VoyageEmbedder

        embedder = VoyageEmbedder(
            model=args.embedding_model,
            cache_path=args.embedding_cache,
            document_input_type=(
                "document" if args.embedding_input_mode == "asymmetric" else None
            ),
            query_input_type=(
                "query" if args.embedding_input_mode == "asymmetric" else None
            ),
        )
    weights = (args.semantic_weight, args.bm25_weight)
    scores, rows = run(
        dataset,
        method=method,
        embedder=embedder,
        candidate_pool=args.candidate_pool,
        semantic_weight=weights[0],
        bm25_weight=weights[1],
        context_by_id=contexts,
        lexical_backend=args.lexical_backend,
        es_index_name=args.es_index,
        elasticsearch_url=args.es_url,
        es_recreate=args.es_recreate_index,
        reranker=reranker,
        rerank_mode=args.rerank_mode,
    )
    save_run(
        args.out,
        method=method,
        scores=scores,
        per_query=rows,
        corpus_path=corpus,
        evaluation_path=eval_file,
        model_name=args.embedding_model if embedder else None,
        embedding_input_mode=args.embedding_input_mode,
        candidate_pool=args.candidate_pool,
        weights=weights,
        context_manifest=context_manifest,
        lexical_backend=args.lexical_backend,
        es_index_name=args.es_index,
        reranker_model=args.rerank_model,
        rerank_mode=args.rerank_mode,
        experiment_id=args.experiment,
        dataset_kind=("beir_document" if args.dataset == "beir" else "anthropic_chunk"),
        query_source_path=source_queries_path,
    )
    print(json.dumps({"out": args.out, "metrics": scores}, indent=2))


def context_generate(args: argparse.Namespace) -> None:
    corpus, eval_file = _files(Path(args.data_dir))
    if args.provider == "anthropic":
        if not args.allow_paid_api:
            raise RuntimeError("Anthropic generation is billable: add --allow-paid-api")
        if not args.model:
            raise ValueError("--model is required for Anthropic contextualization")
        provider = AnthropicContextualizer(
            args.model,
            max_output_tokens=args.max_output_tokens,
            cache_enabled=not args.no_prompt_cache,
        )
    else:
        provider = MetadataContextualizer()
    manifest = generate_context_bundle(
        corpus, eval_file, args.out, provider, cache_path=args.cache_path
    )
    print(json.dumps({"out": str(args.out), "manifest": manifest}, indent=2))


def context_validate(args: argparse.Namespace) -> None:
    corpus, eval_file = _files(Path(args.data_dir))
    dataset = load_dataset(corpus, eval_file)
    bundle = load_context_bundle(
        args.context_dir, dataset, corpus_path=corpus, evaluation_path=eval_file
    )
    print(json.dumps({
        "status": "valid", "contexts": len(bundle.contexts),
        "strategy": bundle.manifest["strategy"],
        "model": bundle.manifest["model"],
        "cache_hits": bundle.manifest["cache_hits"],
        "cache_misses": bundle.manifest["cache_misses"],
    }, indent=2))


def list_matrix() -> None:
    for spec in PHASE4_EXPERIMENTS.values():
        print(
            f"{spec.experiment_id} "
            f"method={spec.method:<6} context={str(spec.use_context):<5} "
            f"rerank={spec.rerank}"
        )


def run_matrix(args: argparse.Namespace) -> None:
    items = plan_matrix(
        args.ids,
        data_dir=str(args.data_dir),
        out_root=str(args.out_root),
        context_dir=str(args.context_dir) if args.context_dir else None,
        lexical_backend=args.lexical_backend,
        es_url=args.es_url,
        es_recreate_index=args.es_recreate_index,
        embedding_model=args.embedding_model,
        embedding_input_mode=args.embedding_input_mode,
        rerank_model=args.rerank_model,
        rerank_mode=args.rerank_mode,
        allow_paid_api=args.allow_paid_api,
        index_namespace=args.es_index_namespace,
        dataset_format=args.dataset,
        beir_split=args.beir_split,
    )
    print(json.dumps({"execute": args.execute, "experiments": items}, indent=2))
    if not args.execute:
        return
    blocked = [item for item in items if item["blockers"]]
    if blocked:
        raise ValueError(
            "Cannot execute blocked batch: " +
            "; ".join(item["experiment"] + ": " + ", ".join(item["blockers"])
                      for item in blocked)
        )
    for item in items:
        subprocess.run(item["command"], check=True)


def run_model_sweep(args: argparse.Namespace) -> None:
    models = plan_model_sweep(
        args.models, args.ids,
        out_root=str(args.out_root), data_dir=str(args.data_dir),
        context_dir=str(args.context_dir) if args.context_dir else None,
        lexical_backend=args.lexical_backend, es_url=args.es_url,
        rerank_model=args.rerank_model, rerank_mode=args.rerank_mode,
        embedding_input_mode=args.embedding_input_mode,
        allow_paid_api=args.allow_paid_api,
        dataset_format=args.dataset, beir_split=args.beir_split,
    )
    print(json.dumps({"execute": args.execute, "model_sweep": models}, indent=2))
    if not args.execute:
        return
    failures = [
        f"{item['model']} / {exp['experiment']}: {', '.join(exp['blockers'])}"
        for item in models for exp in item["experiments"] if exp["blockers"]
    ]
    if failures:
        raise ValueError("Blocked model sweep: " + "; ".join(failures))
    for item in models:
        for experiment in item["experiments"]:
            subprocess.run(experiment["command"], check=True)


def compare(paths: list[Path]) -> None:
    for path in paths:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        print(
            f"{path}: method={manifest['method']} "
            f"legacy_pass@20={metrics.get('cookbook_pass@20', float('nan')):.4f} "
            f"id_recall@20={metrics.get('recall@20', float('nan')):.4f}"
        )


def report_analysis(args: argparse.Namespace) -> None:
    if args.report_cmd == "compare":
        compare(args.runs)
        return
    if args.report_cmd == "paired":
        baseline = load_run(args.baseline)
        candidate = load_run(args.candidate)
        labels = load_query_labels(args.query_labels) if args.query_labels else None
        summary, rows = compare_runs(
            baseline, candidate, metric=args.metric,
            resamples=args.resamples, permutations=args.permutations,
            seed=args.seed, labels=labels,
        )
        save_paired_report(args.out, summary, rows)
        print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))
    elif args.report_cmd == "batch":
        baseline = load_run(args.baseline)
        candidates = [load_run(path) for path in args.candidates]
        labels = load_query_labels(args.query_labels) if args.query_labels else None
        summary = save_batch_report(
            baseline, candidates, metric=args.metric, output=args.out,
            resamples=args.resamples, permutations=args.permutations,
            seed=args.seed, labels=labels,
        )
        print(json.dumps({"out": str(args.out), "batch": summary}, indent=2))
    elif args.report_cmd == "leaderboard":
        data = build_leaderboard([load_run(path) for path in args.runs],
                                 metric=args.metric)
        save_json_report(args.out, data, kind="leaderboard")
        print(json.dumps({"out": str(args.out), "leaderboard": data}, indent=2))
    elif args.report_cmd == "pareto":
        data = build_pareto(
            [load_run(path) for path in args.runs],
            metric=args.metric, costs_path=args.costs,
        )
        save_json_report(args.out, data, kind="pareto")
        print(json.dumps({"out": str(args.out), "pareto": data}, indent=2))
    else:
        raise ValueError(f"Unsupported report command {args.report_cmd}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crlab", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="namespace", required=True)

    data = subs.add_parser("data", help="Download or validate raw official data")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    download_parser = data_sub.add_parser("download")
    download_parser.add_argument("--out", type=Path, default=Path("data/raw/anthropic"))
    download_parser.add_argument("--force", action="store_true")
    beir_download_parser = data_sub.add_parser("beir-download")
    beir_download_parser.add_argument("--dataset", choices=["scifact"], default="scifact")
    beir_download_parser.add_argument("--out", type=Path, default=Path("data/raw/beir"))
    validate_parser = data_sub.add_parser("validate")
    validate_parser.add_argument("--data-dir", type=Path, default=Path("data/raw/anthropic"))
    validate_parser.add_argument("--expect-official", action="store_true")
    validate_parser.add_argument("--dataset", choices=["anthropic", "beir"], default="anthropic")
    validate_parser.add_argument("--beir-split", default="test")

    context = subs.add_parser("context", help="Generate or validate chunk contexts")
    context_sub = context.add_subparsers(dest="context_cmd", required=True)
    generate_parser = context_sub.add_parser("generate")
    generate_parser.add_argument("--data-dir", type=Path, default=Path("data/raw/anthropic"))
    generate_parser.add_argument("--out", type=Path, required=True)
    generate_parser.add_argument("--provider", choices=["metadata", "anthropic"], default="metadata")
    generate_parser.add_argument("--model", default=None)
    generate_parser.add_argument("--cache-path", type=Path, default=Path("data/cache/contexts.sqlite3"))
    generate_parser.add_argument("--max-output-tokens", type=int, default=2048)
    generate_parser.add_argument("--no-prompt-cache", action="store_true")
    generate_parser.add_argument("--allow-paid-api", action="store_true")
    context_validate_parser = context_sub.add_parser("validate")
    context_validate_parser.add_argument("--data-dir", type=Path, default=Path("data/raw/anthropic"))
    context_validate_parser.add_argument("--context-dir", type=Path, required=True)

    evaluation = subs.add_parser("eval", help="Run benchmark methods")
    eval_sub = evaluation.add_subparsers(dest="eval_cmd", required=True)
    run_parser = eval_sub.add_parser("run")
    run_parser.add_argument("--data-dir", default="data/raw/anthropic")
    run_parser.add_argument("--dataset", choices=["anthropic", "beir"], default="anthropic")
    run_parser.add_argument("--beir-split", default="test")
    run_parser.add_argument("--method", choices=["bm25", "vector", "hybrid"])
    run_parser.add_argument("--experiment", choices=sorted(PHASE4_EXPERIMENTS))
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--embedding-model", default="voyage-2")
    run_parser.add_argument("--embedding-input-mode", choices=["legacy", "asymmetric"], default="legacy")
    run_parser.add_argument("--context-dir", type=Path, default=None)
    run_parser.add_argument("--embedding-cache", default="data/cache/embeddings.json")
    run_parser.add_argument("--allow-paid-api", action="store_true")
    run_parser.add_argument("--candidate-pool", type=int, default=150)
    run_parser.add_argument("--semantic-weight", type=float, default=0.8)
    run_parser.add_argument("--bm25-weight", type=float, default=0.2)
    run_parser.add_argument("--lexical-backend", choices=["local", "elasticsearch"], default="local")
    run_parser.add_argument("--es-url", default="http://localhost:9200")
    run_parser.add_argument("--es-index", default=None)
    run_parser.add_argument("--es-recreate-index", action="store_true")
    run_parser.add_argument("--rerank-model", default=None)
    run_parser.add_argument("--rerank-mode", choices=["fixed", "cookbook"], default="fixed")

    matrix_parser = subs.add_parser("matrix", help="List E01-E10 experiment recipes")
    matrix_sub = matrix_parser.add_subparsers(dest="matrix_cmd", required=True)
    matrix_sub.add_parser("list")
    matrix_run = matrix_sub.add_parser("run", help="Plan E01-E10; execute only with --execute")
    matrix_run.add_argument("--ids", nargs="+",
                            choices=sorted(PHASE4_EXPERIMENTS),
                            default=list(PHASE4_EXPERIMENTS))
    matrix_run.add_argument("--data-dir", type=Path, default=Path("data/raw/anthropic"))
    matrix_run.add_argument("--dataset", choices=["anthropic", "beir"], default="anthropic")
    matrix_run.add_argument("--beir-split", default="test")
    matrix_run.add_argument("--es-index-namespace", default="crlab_phase4")
    matrix_run.add_argument("--out-root", type=Path, default=Path("runs/phase4"))
    matrix_run.add_argument("--context-dir", type=Path, default=None)
    matrix_run.add_argument("--lexical-backend", choices=["local", "elasticsearch"],
                            default="elasticsearch")
    matrix_run.add_argument("--es-url", default="http://localhost:9200")
    matrix_run.add_argument("--es-recreate-index", action="store_true")
    matrix_run.add_argument("--embedding-model", default="voyage-2")
    matrix_run.add_argument("--embedding-input-mode", choices=["legacy", "asymmetric"], default="legacy")
    matrix_run.add_argument("--rerank-model", default=None)
    matrix_run.add_argument("--rerank-mode", choices=["fixed", "cookbook"], default="fixed")
    matrix_run.add_argument("--allow-paid-api", action="store_true")
    matrix_run.add_argument("--execute", action="store_true")

    sweep = matrix_sub.add_parser(
        "sweep", help="Plan isolated multi-embedding-model experiments; --execute to run"
    )
    sweep.add_argument("--models", nargs="+", required=True)
    sweep.add_argument("--embedding-input-mode", choices=["legacy", "asymmetric"], default="legacy")
    sweep.add_argument("--ids", nargs="+", choices=sorted(PHASE4_EXPERIMENTS),
                       default=["E01", "E03"])
    sweep.add_argument("--data-dir", type=Path, default=Path("data/raw/anthropic"))
    sweep.add_argument("--dataset", choices=["anthropic", "beir"], default="anthropic")
    sweep.add_argument("--beir-split", default="test")
    sweep.add_argument("--out-root", type=Path, default=Path("runs/model-sweep"))
    sweep.add_argument("--context-dir", type=Path)
    sweep.add_argument("--lexical-backend", choices=["local", "elasticsearch"],
                       default="elasticsearch")
    sweep.add_argument("--es-url", default="http://localhost:9200")
    sweep.add_argument("--rerank-model")
    sweep.add_argument("--rerank-mode", choices=["fixed", "cookbook"], default="fixed")
    sweep.add_argument("--allow-paid-api", action="store_true")
    sweep.add_argument("--execute", action="store_true")

    report = subs.add_parser("report", help="Compare executed runs")
    report_sub = report.add_subparsers(dest="report_cmd", required=True)
    compare_parser = report_sub.add_parser("compare")
    compare_parser.add_argument("--runs", nargs="+", type=Path, required=True)

    paired = report_sub.add_parser("paired", help="Paired CI, p-value and query deltas")
    paired.add_argument("--baseline", type=Path, required=True)
    paired.add_argument("--candidate", type=Path, required=True)
    batch = report_sub.add_parser("batch", help="Paired CIs with Holm correction")
    batch.add_argument("--baseline", type=Path, required=True)
    batch.add_argument("--candidates", nargs="+", type=Path, required=True)
    for statistical in (paired, batch):
        statistical.add_argument("--metric", default="recall@20")
        statistical.add_argument("--out", type=Path, required=True)
        statistical.add_argument("--query-labels", type=Path, default=None)
        statistical.add_argument("--resamples", type=int, default=2000)
        statistical.add_argument("--permutations", type=int, default=5000)
        statistical.add_argument("--seed", type=int, default=42)

    leaderboard = report_sub.add_parser("leaderboard")
    leaderboard.add_argument("--runs", nargs="+", type=Path, required=True)
    leaderboard.add_argument("--metric", default="recall@20")
    leaderboard.add_argument("--out", type=Path, required=True)

    pareto = report_sub.add_parser("pareto")
    pareto.add_argument("--runs", nargs="+", type=Path, required=True)
    pareto.add_argument("--metric", default="recall@20")
    pareto.add_argument("--costs", type=Path, required=True)
    pareto.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.namespace == "data":
            if args.data_cmd == "download":
                download(args.out, args.force)
            elif args.data_cmd == "beir-download":
                print(json.dumps(download_scifact(args.out), indent=2))
            else:
                validate(args.data_dir, expect_official=args.expect_official,
                         dataset_format=args.dataset, beir_split=args.beir_split)
        elif args.namespace == "context":
            if args.context_cmd == "generate":
                context_generate(args)
            else:
                context_validate(args)
        elif args.namespace == "eval":
            evaluate(args)
        elif args.namespace == "matrix":
            if args.matrix_cmd == "list":
                list_matrix()
            elif args.matrix_cmd == "run":
                run_matrix(args)
            else:
                run_model_sweep(args)
        else:
            report_analysis(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
