"""Auditable reporting over already-executed benchmark artifacts only."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from crlab.analysis.comparison import compare_runs
from crlab.analysis.io import Run, assert_pairable
from crlab.analysis.statistics import holm_adjust


def _prepare(path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report {path}")
    path.mkdir(parents=True)
    return path


def _write(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8",
    )


def _percent(number: float) -> str:
    return f"{number * 100:+.4f} pp"


def save_paired_report(
    output: str | Path,
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    root = _prepare(output)
    _write(root / "summary.json", summary)
    with (root / "per_query_delta.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fields = (
        "query_id", "category", "group_id", "baseline_score",
        "candidate_score", "delta", "change",
    )
    with (root / "per_query_delta.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    lines = [
        "# Paired retrieval comparison",
        "",
        f"- Baseline: {summary['baseline']}",
        f"- Candidate: {summary['candidate']}",
        f"- Metric: {summary['metric']}",
        f"- Query count: {summary['query_count']}",
        f"- Mean difference (candidate minus baseline): {_percent(summary['mean_delta'])}",
        f"- 95% paired bootstrap CI: "
        f"[{_percent(summary['ci_low'])}, {_percent(summary['ci_high'])}]",
        f"- Two-sided sign-flip p-value: {summary['pvalue']:.6f}",
        f"- Unit: {summary['resampling_unit']} / {summary['num_groups']} groups",
        f"- Improved / regressed / unchanged: "
        f"{summary['improved']} / {summary['regressed']} / {summary['unchanged']}",
        "",
        "## Query groups",
        "| Category | Count | Mean difference |",
        "|---|---:|---:|",
    ]
    for category, item in summary["category_breakdown"].items():
        lines.append(
            f"| {category.replace('|', '/')} | {item['query_count']} "
            f"| {_percent(item['mean_delta'])} |"
        )
    lines.extend(["", "## Largest regressions", ""])
    for row in summary["top_regressions"]:
        lines.append(
            f"- {row['query_id']}: {_percent(row['delta'])}: "
            f"{row['query'].replace(chr(10), ' ')[:150]}"
        )
    if not summary["top_regressions"]:
        lines.append("No negative query deltas observed.")
    lines.extend(["", summary["caveat"], ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def save_batch_report(
    baseline: Run, candidates: Sequence[Run], *,
    metric: str, output: str | Path,
    resamples: int = 2000, permutations: int = 5000, seed: int = 42,
    labels: Mapping[str, Mapping[str, str]] | None = None,
) -> dict:
    if not candidates:
        raise ValueError("Batch report must include candidates")
    records: list[dict] = []
    per_query: list[list[dict]] = []
    for candidate in candidates:
        summary, rows = compare_runs(
            baseline, candidate, metric=metric, resamples=resamples,
            permutations=permutations, seed=seed, labels=labels,
        )
        records.append(summary)
        per_query.append(rows)
    adjusted = holm_adjust([row["pvalue"] for row in records])
    for item, adjusted_p in zip(records, adjusted, strict=True):
        item["holm_adjusted_pvalue"] = adjusted_p
        item["holm_family_size"] = len(records)
    root = _prepare(output)
    for i, (item, rows) in enumerate(zip(records, per_query, strict=True), start=1):
        save_paired_report(root / f"candidate-{i:02d}", item, rows)
    result = {
        "schema_version": 1, "metric": metric,
        "baseline": str(baseline.path), "family_size": len(records),
        "results": records,
        "warning": "Holm correction applies only to this specified comparison family.",
    }
    _write(root / "batch.json", result)
    lines = [
        "# Multi-run ablation comparisons", "",
        "| Candidate | Delta | CI low | CI high | p | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            f"| {item['candidate']} | {_percent(item['mean_delta'])} | "
            f"{_percent(item['ci_low'])} | {_percent(item['ci_high'])} | "
            f"{item['pvalue']:.6f} | {item['holm_adjusted_pvalue']:.6f} |"
        )
    (root / "report.md").write_text(
        "\n".join(lines + ["", result["warning"], ""]), encoding="utf-8"
    )
    return result


def build_leaderboard(runs: Sequence[Run], *, metric: str) -> dict:
    if not runs:
        raise ValueError("Leaderboard needs one or more runs")
    groups: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        if metric not in run.metrics:
            raise ValueError(f"Metric {metric} unavailable in {run.path}")
        groups.setdefault(run.dataset_key, []).append(run)
    result = []
    for key, group in sorted(groups.items()):
        for run in group[1:]:
            assert_pairable(group[0], run, metric)
        ordered = sorted(group, key=lambda run: (-run.metrics[metric], str(run.path)))
        result.append({
            "corpus_sha256": key[0], "evaluation_sha256": key[1],
            "query_count": len(group[0].queries),
            "rankings": [
                {
                    "rank": i, "path": str(run.path),
                    "score": run.metrics[metric],
                    "experiment": run.manifest.get("experiment_id"),
                    "model": run.manifest.get("vector_model"),
                    "method": run.manifest.get("method"),
                    "context": run.manifest.get("context"),
                    "reranker": run.manifest.get("reranker_model"),
                }
                for i, run in enumerate(ordered, 1)
            ],
        })
    return {
        "schema_version": 1, "metric": metric, "datasets": result,
        "warning": "Different datasets are grouped separately; scores are not pooled.",
    }


def _load_costs(path: str | Path) -> dict[str, dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("runs"), list):
        raise ValueError("Cost ledger requires a runs array")
    costs: dict[str, dict] = {}
    for item in obj["runs"]:
        key = str(Path(item["path"]).resolve())
        amount = item.get("cost_usd")
        source = item.get("source")
        if key in costs:
            raise ValueError(f"Duplicate cost entry: {key}")
        if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError(f"Invalid or missing total USD cost for {key}")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Missing cost provenance for {key}")
        costs[key] = {"cost_usd": float(amount), "source": source}
    return costs


def build_pareto(runs: Sequence[Run], *, metric: str, costs_path: str | Path) -> dict:
    if not runs:
        raise ValueError("Pareto needs one or more runs")
    reference = runs[0]
    if metric not in reference.metrics:
        raise ValueError(f"Unknown quality metric {metric}")
    for run in runs[1:]:
        assert_pairable(reference, run, metric)
    ledger = _load_costs(costs_path)
    points = []
    for run in runs:
        key = str(run.path.resolve())
        if key not in ledger:
            raise ValueError(f"Unknown run cost for {run.path} must not be set to zero")
        points.append({
            "path": str(run.path),
            "experiment": run.manifest.get("experiment_id"),
            "score": run.metrics[metric],
            "cost_usd": ledger[key]["cost_usd"],
            "cost_source": ledger[key]["source"],
        })
    for point in points:
        point["dominated_by"] = [
            other["path"] for other in points
            if other is not point and other["score"] >= point["score"]
            and other["cost_usd"] <= point["cost_usd"]
            and (other["score"] > point["score"]
                 or other["cost_usd"] < point["cost_usd"])
        ]
        point["pareto_optimal"] = not point["dominated_by"]
    return {
        "schema_version": 1, "metric": metric,
        "corpus_sha256": reference.dataset_key[0],
        "evaluation_sha256": reference.dataset_key[1],
        "points": points,
        "frontier": [p["path"] for p in points if p["pareto_optimal"]],
        "warning": (
            "Compare end-to-end workload cost on the same query set, including "
            "context generation and model calls where applicable. Unknown is not zero."
        ),
    }


def save_json_report(output: str | Path, result: dict, *, kind: str) -> None:
    if kind not in ("leaderboard", "pareto"):
        raise ValueError("Invalid report kind")
    root = _prepare(output)
    _write(root / f"{kind}.json", result)
    lines = [f"# {kind.title()} report", ""]
    if kind == "leaderboard":
        for group in result["datasets"]:
            lines.extend([
                "## Dataset " + group["corpus_sha256"][:12], "",
                "| Rank | Run | Score | Model |",
                "|---:|---|---:|---|",
            ])
            for item in group["rankings"]:
                lines.append(
                    f"| {item['rank']} | {item['path']} | {item['score']:.6f} "
                    f"| {item['model'] or 'n/a'} |"
                )
            lines.append("")
    else:
        lines.extend([
            "| Run | Score | Cost (USD) | On Pareto frontier |",
            "|---|---:|---:|---|",
        ])
        for item in result["points"]:
            lines.append(
                f"| {item['path']} | {item['score']:.6f} | "
                f"{item['cost_usd']:.6f} | {item['pareto_optimal']} |"
            )
    lines.extend(["", result.get("warning", ""), ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
