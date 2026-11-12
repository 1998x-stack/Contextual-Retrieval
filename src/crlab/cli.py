"""crlab CLI: download, validate, run, compare."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from crlab import __version__
from crlab.datasets.anthropic import load_dataset
from crlab.experiments.runner import file_hash, run, save_run

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


def validate(data_dir: Path, *, expect_official: bool = False) -> None:
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
    corpus, eval_file = _files(Path(args.data_dir))
    dataset = load_dataset(corpus, eval_file)
    embedder = None
    if args.method in {"vector", "hybrid"}:
        if not args.allow_paid_api:
            raise RuntimeError(
                "Vector/Hybrid may make billable API requests; pass --allow-paid-api"
            )
        from crlab.providers.voyage import VoyageEmbedder

        embedder = VoyageEmbedder(
            model=args.embedding_model,
            cache_path=args.embedding_cache,
        )
    weights = (args.semantic_weight, args.bm25_weight)
    scores, rows = run(
        dataset,
        method=args.method,
        embedder=embedder,
        candidate_pool=args.candidate_pool,
        semantic_weight=weights[0],
        bm25_weight=weights[1],
    )
    save_run(
        args.out,
        method=args.method,
        scores=scores,
        per_query=rows,
        corpus_path=corpus,
        evaluation_path=eval_file,
        model_name=args.embedding_model if embedder else None,
        candidate_pool=args.candidate_pool,
        weights=weights,
    )
    print(json.dumps({"out": args.out, "metrics": scores}, indent=2))


def compare(paths: list[Path]) -> None:
    for path in paths:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        print(
            f"{path}: method={manifest['method']} "
            f"legacy_pass@20={metrics.get('cookbook_pass@20', float('nan')):.4f} "
            f"id_recall@20={metrics.get('recall@20', float('nan')):.4f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crlab", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="namespace", required=True)

    data = subs.add_parser("data", help="Download or validate raw official data")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    download_parser = data_sub.add_parser("download")
    download_parser.add_argument("--out", type=Path, default=Path("data/raw/anthropic"))
    download_parser.add_argument("--force", action="store_true")
    validate_parser = data_sub.add_parser("validate")
    validate_parser.add_argument("--data-dir", type=Path, default=Path("data/raw/anthropic"))
    validate_parser.add_argument("--expect-official", action="store_true")

    evaluation = subs.add_parser("eval", help="Run benchmark methods")
    eval_sub = evaluation.add_subparsers(dest="eval_cmd", required=True)
    run_parser = eval_sub.add_parser("run")
    run_parser.add_argument("--data-dir", default="data/raw/anthropic")
    run_parser.add_argument("--method", choices=["bm25", "vector", "hybrid"], required=True)
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--embedding-model", default="voyage-2")
    run_parser.add_argument("--embedding-cache", default="data/cache/embeddings.json")
    run_parser.add_argument("--allow-paid-api", action="store_true")
    run_parser.add_argument("--candidate-pool", type=int, default=150)
    run_parser.add_argument("--semantic-weight", type=float, default=0.8)
    run_parser.add_argument("--bm25-weight", type=float, default=0.2)

    report = subs.add_parser("report", help="Compare executed runs")
    report_sub = report.add_subparsers(dest="report_cmd", required=True)
    compare_parser = report_sub.add_parser("compare")
    compare_parser.add_argument("--runs", nargs="+", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.namespace == "data":
            if args.data_cmd == "download":
                download(args.out, args.force)
            else:
                validate(args.data_dir, expect_official=args.expect_official)
        elif args.namespace == "eval":
            evaluate(args)
        else:
            compare(args.runs)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
