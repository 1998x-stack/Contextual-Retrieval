"""Phase 5 statistical research regression tests using synthetic offline run artifacts."""

import json
from pathlib import Path

import pytest

from crlab.analysis.comparison import compare_runs, load_query_labels
from crlab.analysis.io import IncompatibleRuns, load_run
from crlab.analysis.reporting import (
    build_leaderboard, build_pareto, save_batch_report, save_paired_report,
)
from crlab.analysis.statistics import (
    holm_adjust, paired_bootstrap, paired_signflip_pvalue,
)
from crlab.cli import main
from crlab.evaluation.metrics import aggregate, evaluate_query
from crlab.experiments.runner import save_run
from crlab.schemas import SearchHit


@pytest.fixture
def synthetic_runs(sample_files, dataset, tmp_path):
    """Two per-query benchmark runs where candidate retrieves gold at rank one.

    These are *test fixtures*, not claimed benchmark measurements.
    """
    content = {chunk.chunk_id: chunk.content for chunk in dataset.chunks}
    original_order = ["doc-A:1", "doc-B:0", "doc-A:0"]
    improved_orders = [
        ["doc-A:0", "doc-A:1", "doc-B:0"],
        ["doc-B:0", "doc-A:1", "doc-A:0"],
    ]
    cutoffs = (1, 2, 3)

    def save(name, orders):
        rows = []
        for query, order in zip(dataset.queries, orders, strict=True):
            hits = [
                SearchHit(chunk_id=id, score=float(3 - rank), rank=rank + 1,
                          source="synthetic_fixture")
                for rank, id in enumerate(order)
            ]
            scores = evaluate_query(query, hits, content, cutoffs)
            rows.append({
                "query_id": query.query_id,
                "query": query.text,
                "gold_chunk_ids": list(query.golden_chunk_ids),
                "rankings": [
                    {"chunk_id": hit.chunk_id, "score": hit.score,
                     "rank": hit.rank, "source": hit.source}
                    for hit in hits
                ],
                "metrics": scores,
            })
        root = tmp_path / name
        save_run(
            root, method="bm25",
            scores=aggregate([row["metrics"] for row in rows]),
            per_query=rows, corpus_path=sample_files[0],
            evaluation_path=sample_files[1],
        )
        return root
    return (
        save("baseline", [original_order, original_order]),
        save("candidate", improved_orders),
        save("duplicate", [original_order, original_order]),
    )


def test_deterministic_bootstrap_and_permutation():
    samples = [0.0, 0.25, 0.75, 1.0]
    assert paired_bootstrap(samples, resamples=250, seed=17) == paired_bootstrap(
        samples, resamples=250, seed=17
    )
    low, high = paired_bootstrap(samples, resamples=250, seed=17)
    assert 0 <= low <= high <= 1
    assert paired_bootstrap([0, 0], resamples=100) == (0.0, 0.0)
    assert paired_bootstrap([1, 1, 1], resamples=100) == (1, 1)
    assert paired_signflip_pvalue([0, 0], permutations=100) == 1
    p = paired_signflip_pvalue(samples, permutations=200, seed=17)
    assert 0 < p <= 1


def test_cluster_units_and_validation():
    deltas = [1.0, 1.0, -0.5]
    ci = paired_bootstrap(deltas, resamples=100, groups=["a", "a", "b"])
    assert len(ci) == 2 and ci[0] <= ci[1]
    p = paired_signflip_pvalue(
        deltas, permutations=100, groups=["a", "a", "b"],
    )
    assert 0 < p <= 1
    with pytest.raises(ValueError, match="label count"):
        paired_bootstrap(deltas, resamples=100, groups=["a"])
    with pytest.raises(ValueError, match=">= 100"):
        paired_bootstrap(deltas, resamples=1)
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap(deltas, resamples=100, confidence=1.0)


def test_holm_stepdown_adjustment():
    values = holm_adjust([0.01, 0.04, 0.03])
    assert values == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError, match="P-values"):
        holm_adjust([float("nan")])


def test_run_loader_requires_consistent_artifacts(synthetic_runs):
    left, right, _ = synthetic_runs
    assert len(load_run(left).queries) == 2
    assert load_run(left).metrics["recall@1"] == 0
    assert load_run(right).metrics["recall@1"] == 1
    assert load_run(left).dataset_key == load_run(right).dataset_key


def test_paired_comparison_observes_actual_query_deltas(synthetic_runs):
    left, right, _ = synthetic_runs
    summary, rows = compare_runs(
        load_run(left), load_run(right), metric="recall@1",
        resamples=100, permutations=100, seed=3,
    )
    assert summary["mean_delta"] == 1
    assert summary["ci_low"] == 1 and summary["ci_high"] == 1
    assert summary["improved"] == 2
    assert summary["regressed"] == 0
    assert all(row["candidate_first_gold_rank"] == 1 for row in rows)
    assert all(row["baseline_first_gold_rank"] is None for row in rows)
    assert summary["resampling_unit"] == "query"
    assert summary["pvalue"] > 0


def test_identical_runs_have_zero_difference(synthetic_runs):
    left, _, duplicate = synthetic_runs
    summary, rows = compare_runs(
        load_run(left), load_run(duplicate), metric="recall@1",
        resamples=100, permutations=100,
    )
    assert summary["mean_delta"] == 0
    assert summary["pvalue"] == 1.0
    assert summary["unchanged"] == len(rows)


def test_mismatch_source_hash_is_rejected(synthetic_runs):
    left, right, _ = synthetic_runs
    manifest = right / "manifest.json"
    data = json.loads(manifest.read_text())
    data["evaluation_sha256"] = "not-the-same"
    manifest.write_text(json.dumps(data))
    with pytest.raises(IncompatibleRuns, match="SHA-256"):
        compare_runs(load_run(left), load_run(right), metric="recall@1",
                     resamples=100, permutations=100)


def test_mismatch_golden_labels_is_rejected(synthetic_runs):
    left, right, _ = synthetic_runs
    rows_path = right / "per_query.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["gold_chunk_ids"] = ["incorrect"]
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    # The stronger loader now rejects the contradiction even before pair matching.
    with pytest.raises(ValueError, match="Ranking and ID metric disagree"):
        load_run(right)


def test_tampered_aggregate_is_rejected(synthetic_runs):
    left, _, _ = synthetic_runs
    f = left / "metrics.json"
    metrics = json.loads(f.read_text())
    metrics["recall@1"] = 1.0
    f.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Aggregate and query"):
        load_run(left)


def test_query_labels_breakdown_and_clustered_resampling(synthetic_runs, tmp_path):
    left, right, _ = synthetic_runs
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        json.dumps({"query_id": "q_000000", "category": "exact_identifier",
                    "group_id": "doc-A"}) + "\n" +
        json.dumps({"query_id": "q_000001", "category": "semantic",
                    "group_id": "doc-A"}) + "\n"
    )
    labels = load_query_labels(labels_path)
    summary, _ = compare_runs(
        load_run(left), load_run(right), metric="recall@1",
        resamples=100, permutations=100, labels=labels,
    )
    assert summary["resampling_unit"] == "group_id"
    assert summary["num_groups"] == 1
    assert set(summary["category_breakdown"]) == {"semantic", "exact_identifier"}
    with pytest.raises(ValueError, match="unknown query IDs"):
        compare_runs(
            load_run(left), load_run(right), metric="recall@1",
            resamples=100, permutations=100,
            labels={"nonexistent": {"group_id": "x"}},
        )


def test_outputs_are_immutable_and_machine_readable(synthetic_runs, tmp_path):
    left, right, _ = synthetic_runs
    summary, rows = compare_runs(
        load_run(left), load_run(right), metric="recall@1",
        resamples=100, permutations=100,
    )
    folder = tmp_path / "paired-report"
    save_paired_report(folder, summary, rows)
    assert (folder / "summary.json").exists()
    assert len((folder / "per_query_delta.jsonl").read_text().splitlines()) == 2
    assert "Largest regressions" in (folder / "report.md").read_text()
    with pytest.raises(FileExistsError):
        save_paired_report(folder, summary, rows)


def test_batch_holm_correction_and_leaderboard(synthetic_runs, tmp_path):
    left, right, duplicate = map(load_run, synthetic_runs)
    output = tmp_path / "multi"
    batch = save_batch_report(
        left, [right, duplicate], metric="recall@1",
        output=output, resamples=100, permutations=100,
    )
    assert batch["family_size"] == 2
    assert 0 <= batch["results"][0]["holm_adjusted_pvalue"] <= 1
    assert (output / "candidate-01" / "summary.json").exists()
    assert (output / "candidate-02" / "report.md").exists()
    board = build_leaderboard([left, right, duplicate], metric="recall@1")
    assert len(board["datasets"]) == 1
    assert board["datasets"][0]["rankings"][0]["score"] == 1.0


def test_cost_pareto_requires_complete_provenance(synthetic_runs, tmp_path):
    left, right, duplicate = map(load_run, synthetic_runs)
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"runs": [
        {"path": str(left.path), "cost_usd": 0.1, "source": "fixture-price"},
        {"path": str(right.path), "cost_usd": 0.3, "source": "fixture-price"},
        {"path": str(duplicate.path), "cost_usd": 0.4, "source": "fixture-price"},
    ]}))
    frontier = build_pareto(
        [left, right, duplicate], metric="recall@1", costs_path=path,
    )
    assert set(frontier["frontier"]) == {str(left.path), str(right.path)}
    assert frontier["points"][2]["dominated_by"]
    path.write_text(json.dumps({"runs": [
        {"path": str(left.path), "cost_usd": 0.0, "source": "fixture-price"},
    ]}))
    with pytest.raises(ValueError, match="Unknown run cost"):
        build_pareto([left, right], metric="recall@1", costs_path=path)


def test_cli_paired_report_end_to_end(synthetic_runs, tmp_path, capsys):
    left, right, _ = synthetic_runs
    out = tmp_path / "cli"
    main([
        "report", "paired", "--baseline", str(left), "--candidate", str(right),
        "--metric", "recall@1", "--resamples", "100", "--permutations", "100",
        "--out", str(out),
    ])
    assert json.loads(capsys.readouterr().out)["summary"]["mean_delta"] == 1
    assert (out / "report.md").exists()


def test_cli_batch_and_pareto_end_to_end(synthetic_runs, tmp_path, capsys):
    left, right, duplicate = synthetic_runs
    out = tmp_path / "cli-batch"
    main([
        "report", "batch", "--baseline", str(left),
        "--candidates", str(right), str(duplicate),
        "--metric", "recall@1", "--resamples", "100",
        "--permutations", "100", "--out", str(out),
    ])
    assert (out / "batch.json").exists()
    capsys.readouterr()
    board = tmp_path / "leaderboard"
    main([
        "report", "leaderboard", "--runs", str(left), str(right),
        "--metric", "recall@1", "--out", str(board),
    ])
    assert (board / "leaderboard.json").exists()



def test_tampered_query_ranking_is_detected_even_if_scores_look_consistent(synthetic_runs):
    left, _, _ = synthetic_runs
    ranking_file = left / "per_query.jsonl"
    rows = [json.loads(line) for line in ranking_file.read_text().splitlines()]
    rows[0]["rankings"][0]["chunk_id"] = "doc-A:0"  # now gold at rank 1
    ranking_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Ranking and ID metric disagree"):
        load_run(left)



def test_pairing_rejects_changed_query_text_even_with_valid_scores(synthetic_runs):
    left, right, _ = synthetic_runs
    location = right / "per_query.jsonl"
    rows = [json.loads(line) for line in location.read_text().splitlines()]
    rows[0]["query"] = "Question modified after benchmark"
    location.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(IncompatibleRuns, match="Query text"):
        compare_runs(load_run(left), load_run(right), metric="recall@1",
                     resamples=100, permutations=100)
