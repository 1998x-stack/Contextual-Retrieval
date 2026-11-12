"""Matrix planning must be deterministic and never charge APIs without explicit flags."""

import json

import pytest

from crlab.cli import main
from crlab.experiments.matrix import plan_matrix


def test_default_plan_blocks_paid_and_contextual_experiments(tmp_path):
    plan = plan_matrix(
        [f"E{i:02d}" for i in range(1, 11)],
        data_dir="data/raw/anthropic",
        out_root=str(tmp_path),
    )
    assert len(plan) == 10
    assert plan[1]["status"] == "ready"  # E02: original BM25
    assert plan[4]["status"] == "blocked"  # E05: requires context
    assert plan[7]["status"] == "blocked"  # E08: context + reranker + paid API
    assert all(
        entry["command"][1:5] == ["-m", "crlab.cli", "eval", "run"]
        for entry in plan
    )
    assert not any((tmp_path / f"e{i:02d}").exists() for i in range(1, 11))


def test_plan_enables_all_when_explicit_parameters_are_supplied(tmp_path):
    plan = plan_matrix(
        ["E04", "E05", "E08", "E09", "E10"],
        data_dir="fixture",
        out_root=str(tmp_path),
        context_dir="/path/to/contexts",
        rerank_model="user-chosen-rerank-model",
        allow_paid_api=True,
    )
    assert all(item["status"] == "ready" for item in plan)
    assert "--allow-paid-api" in plan[0]["command"]
    assert "--es-index" in plan[1]["command"]
    assert plan[3]["experiment"] == "E09"
    assert "--rerank-model" in plan[-1]["command"]


def test_plan_rejects_invalid_rerank_mode_and_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="vector-only"):
        plan_matrix(
            ["E10"], data_dir="fixture", out_root=str(tmp_path),
            context_dir="/contexts", rerank_model="model",
            rerank_mode="cookbook", allow_paid_api=True,
        )
    with pytest.raises(ValueError, match="Duplicate experiment"):
        plan_matrix(["E02", "e02"], data_dir="fixture", out_root=str(tmp_path))


def test_cli_matrix_dry_run_prints_blockers_without_executing(tmp_path, capsys):
    main(["matrix", "run", "--ids", "E02", "E08",
          "--lexical-backend", "local", "--out-root", str(tmp_path)])
    results = json.loads(capsys.readouterr().out)
    assert not results["execute"]
    assert results["experiments"][0]["status"] == "ready"
    assert results["experiments"][1]["status"] == "blocked"
    assert not (tmp_path / "e02").exists()


def test_cli_batch_executes_only_explicit_offline_case(sample_files, tmp_path):
    path = tmp_path / "runs"
    main(["matrix", "run", "--ids", "E02",
          "--lexical-backend", "local",
          "--data-dir", str(sample_files[0].parent),
          "--out-root", str(path), "--execute"])
    manifest = json.loads((path / "e02" / "manifest.json").read_text())
    assert manifest["experiment_id"] == "E02"
    assert manifest["lexical_engine"] == "local_python_bm25"


def test_cli_batch_fails_preflight_without_running_any_part(sample_files, tmp_path):
    root = tmp_path / "outputs"
    with pytest.raises(SystemExit) as exc:
        main(["matrix", "run", "--ids", "E02", "E04",
              "--lexical-backend", "local",
              "--data-dir", str(sample_files[0].parent),
              "--out-root", str(root), "--execute"])
    assert exc.value.code == 2
    assert not (root / "e02").exists()
