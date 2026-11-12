"""Multi-model experiments must have isolated artifacts and ES index namespaces."""

import json

import pytest

from crlab.cli import main
from crlab.experiments.matrix import plan_matrix, plan_model_sweep


def test_model_sweep_uses_isolated_output_and_index_namespaces(tmp_path):
    sweep = plan_model_sweep(
        ["voyage-2", "voyage-large-2"],
        ["E01", "E03"],
        data_dir="data/raw/anthropic",
        out_root=str(tmp_path / "models"),
        allow_paid_api=True,
    )
    assert len(sweep) == 2
    assert all(item["status"] == "ready" for item in sweep)
    assert sweep[0]["run_namespace"] != sweep[1]["run_namespace"]
    paths = [
        item["experiments"][0]["command"][item["experiments"][0]["command"].index("--out") + 1]
        for item in sweep
    ]
    assert paths[0] != paths[1]
    es_indexes = []
    for item in sweep:
        command = item["experiments"][1]["command"]
        es_indexes.append(command[command.index("--es-index") + 1])
    assert es_indexes[0] != es_indexes[1]
    assert all(index.startswith("crlab_") for index in es_indexes)


def test_sweep_default_does_not_authorize_paid_calls(tmp_path):
    sweep = plan_model_sweep(
        ["model-x"], ["E01"], out_root=str(tmp_path), data_dir="data/raw/anthropic",
    )
    assert sweep[0]["status"] == "blocked"
    assert "requires --allow-paid-api" in sweep[0]["experiments"][0]["blockers"]


def test_sweep_rejects_duplicate_model_and_bm25_only(tmp_path):
    params = {"out_root": str(tmp_path), "data_dir": "fixture"}
    with pytest.raises(ValueError, match="distinct"):
        plan_model_sweep(["same", "same"], ["E01"], **params)
    with pytest.raises(ValueError, match="vector or hybrid"):
        plan_model_sweep(["model"], ["E02"], **params)


def test_sweep_cross_dataset_beir_commands_keep_label_granularity(tmp_path):
    sweep = plan_model_sweep(
        ["model-x"], ["E01", "E03"],
        out_root=str(tmp_path), data_dir="data/raw/beir/scifact",
        dataset_format="beir", allow_paid_api=True,
    )
    for exp in sweep[0]["experiments"]:
        cmd = exp["command"]
        assert cmd[cmd.index("--dataset") + 1] == "beir"
        assert cmd[cmd.index("--beir-split") + 1] == "test"


def test_beir_context_experiments_block_at_preflight(tmp_path):
    items = plan_matrix(
        ["E01", "E04"], out_root=str(tmp_path), data_dir="beir/scifact",
        dataset_format="beir", context_dir="somewhere",
        allow_paid_api=True,
    )
    assert items[0]["status"] == "ready"
    assert items[1]["status"] == "blocked"
    assert any("document-level Qrels" in x for x in items[1]["blockers"])


def test_sweep_cli_is_dry_run_until_explicit_execute(tmp_path, capsys):
    main([
        "matrix", "sweep", "--models", "voyage-2", "voyage-large-2",
        "--ids", "E01", "E03", "--out-root", str(tmp_path / "runs"),
    ])
    response = json.loads(capsys.readouterr().out)
    assert response["execute"] is False
    assert len(response["model_sweep"]) == 2
    assert all(item["status"] == "blocked" for item in response["model_sweep"])
    assert not (tmp_path / "runs").exists()


def test_sweep_execute_preflights_all_models_first(tmp_path):
    output = tmp_path / "runs"
    with pytest.raises(SystemExit) as error:
        main([
            "matrix", "sweep", "--models", "voyage-2", "voyage-large-2",
            "--ids", "E01", "--out-root", str(output), "--execute",
        ])
    assert error.value.code == 2
    assert not output.exists()
