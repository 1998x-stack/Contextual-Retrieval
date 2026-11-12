"""E01-E10 predefined full-factorial-style retrieval/rerank ablation recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    method: str
    use_context: bool
    rerank: bool

    def to_dict(self) -> dict:
        return asdict(self)


PHASE4_EXPERIMENTS = {
    spec.experiment_id: spec
    for spec in (
        ExperimentSpec("E01", "vector", False, False),
        ExperimentSpec("E02", "bm25", False, False),
        ExperimentSpec("E03", "hybrid", False, False),
        ExperimentSpec("E04", "vector", True, False),
        ExperimentSpec("E05", "bm25", True, False),
        ExperimentSpec("E06", "hybrid", True, False),
        ExperimentSpec("E07", "vector", False, True),
        ExperimentSpec("E08", "vector", True, True),
        ExperimentSpec("E09", "hybrid", False, True),
        ExperimentSpec("E10", "hybrid", True, True),
    )
}


def experiment_spec(experiment_id: str) -> ExperimentSpec:
    try:
        return PHASE4_EXPERIMENTS[experiment_id.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment {experiment_id}; expected E01-E10") from exc



def plan_matrix(
    experiment_ids: list[str],
    *,
    data_dir: str,
    out_root: str,
    context_dir: str | None = None,
    lexical_backend: str = "elasticsearch",
    es_url: str = "http://localhost:9200",
    es_recreate_index: bool = False,
    embedding_model: str = "voyage-2",
    rerank_model: str | None = None,
    rerank_mode: str = "fixed",
    allow_paid_api: bool = False,
) -> list[dict]:
    """Make a safe, explicit command plan without running any models.

    A blocked plan can be displayed, but batch execution must reject it
    *before* any external API or Elasticsearch indexing side effect.
    """
    from pathlib import Path
    import sys

    if lexical_backend not in {"local", "elasticsearch"}:
        raise ValueError("lexical_backend must be local or elasticsearch")
    if rerank_mode not in {"fixed", "cookbook"}:
        raise ValueError("rerank_mode must be fixed or cookbook")
    if not experiment_ids:
        raise ValueError("Select one or more experiment IDs")
    if len({item.upper() for item in experiment_ids}) != len(experiment_ids):
        raise ValueError("Duplicate experiment IDs in batch")
    plans: list[dict] = []
    for item in experiment_ids:
        spec = experiment_spec(item)
        blockers: list[str] = []
        if spec.use_context and not context_dir:
            blockers.append("missing --context-dir")
        if spec.rerank and not rerank_model:
            blockers.append("missing --rerank-model")
        paid = spec.method in {"vector", "hybrid"} or spec.rerank
        if paid and not allow_paid_api:
            blockers.append("requires --allow-paid-api")
        if spec.rerank and rerank_mode == "cookbook" and spec.method != "vector":
            blockers.append("Cookbook rerank mode is vector-only")
        destination = Path(out_root) / spec.experiment_id.lower()
        if destination.exists() and any(destination.iterdir()):
            blockers.append(f"output exists: {destination}")

        command = [
            sys.executable, "-m", "crlab.cli", "eval", "run",
            "--experiment", spec.experiment_id,
            "--data-dir", data_dir,
            "--out", str(destination),
            "--embedding-model", embedding_model,
        ]
        if spec.use_context and context_dir:
            command.extend(["--context-dir", context_dir])
        if spec.method in {"bm25", "hybrid"}:
            command.extend(["--lexical-backend", lexical_backend])
            if lexical_backend == "elasticsearch":
                command.extend([
                    "--es-url", es_url,
                    "--es-index", f"crlab_phase4_{spec.experiment_id.lower()}",
                ])
                if es_recreate_index:
                    command.append("--es-recreate-index")
        if spec.rerank and rerank_model:
            command.extend(["--rerank-model", rerank_model, "--rerank-mode", rerank_mode])
        if paid and allow_paid_api:
            command.append("--allow-paid-api")
        plans.append({
            "experiment": spec.experiment_id,
            "status": "blocked" if blockers else "ready",
            "needs_paid_api": paid,
            "blockers": blockers,
            "command": command,
        })
    return plans
