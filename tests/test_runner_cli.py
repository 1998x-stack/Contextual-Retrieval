import json

import pytest

from crlab.cli import main
from crlab.experiments.runner import run, save_run


class FakeEmbedder:
    """Deterministic offline stub, never used as an official retrieval model."""

    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if "memory" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]

    def embed_queries(self, texts):
        return [
            [1.0, 0.0] if "memory" in text.lower()
            else [0.0, 1.0]
            for text in texts
        ]


@pytest.mark.parametrize("method", ["bm25", "vector", "hybrid"])
def test_benchmark_produces_real_rankings(dataset, method):
    embedder = FakeEmbedder() if method != "bm25" else None
    scores, rows = run(
        dataset, method=method, embedder=embedder,
        cutoffs=(1, 2), candidate_pool=3,
    )
    assert len(rows) == 2
    assert scores["recall@2"] >= 0
    assert all(row["rankings"] for row in rows)
    assert all(row["query_id"].startswith("q_") for row in rows)


def test_run_writes_manifest_without_credentials(sample_files, dataset, tmp_path):
    scores, rows = run(dataset, method="bm25", cutoffs=(1,), candidate_pool=3)
    out = tmp_path / "run"
    save_run(out, method="bm25", scores=scores, per_query=rows,
             corpus_path=sample_files[0], evaluation_path=sample_files[1])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["query_count"] == 2
    assert manifest["lexical_engine"] == "local_python_bm25"
    assert len((out / "per_query.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        save_run(out, method="bm25", scores=scores, per_query=rows,
                 corpus_path=sample_files[0], evaluation_path=sample_files[1])


def test_cli_local_baseline(sample_files, tmp_path, capsys):
    data_dir = sample_files[0].parent
    main(["data", "validate", "--data-dir", str(data_dir)])
    counts = json.loads(capsys.readouterr().out)
    assert counts["chunks"] == 3
    run_dir = tmp_path / "benchmark-run"
    main(["eval", "run", "--data-dir", str(data_dir),
          "--method", "bm25", "--out", str(run_dir)])
    assert (run_dir / "metrics.json").exists()
    main(["report", "compare", "--runs", str(run_dir)])
    assert "id_recall@20" in capsys.readouterr().out


def test_vector_requires_explicit_permission(sample_files, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "run", "--data-dir", str(sample_files[0].parent),
              "--method", "vector", "--out", str(tmp_path / "paid")])
    assert exc.value.code == 2
