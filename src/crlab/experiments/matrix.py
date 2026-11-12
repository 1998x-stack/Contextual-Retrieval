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
    index_namespace: str = "crlab_phase4",
    dataset_format: str = "anthropic",
    beir_split: str = "test",
) -> list[dict]:
    """Make a safe, explicit command plan without running any models.

    A blocked plan can be displayed, but batch execution must reject it
    *before* any external API or Elasticsearch indexing side effect.
    """
    from pathlib import Path
    import re
    import sys

    if not re.fullmatch(r"crlab_[a-z0-9_-]{1,80}", index_namespace):
        raise ValueError("Elasticsearch namespace must match crlab_[a-z0-9_-]+")
    if dataset_format not in {"anthropic", "beir"}:
        raise ValueError("Unsupported dataset format")
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
        if dataset_format == "beir" and spec.use_context:
            blockers.append("BEIR has document-level Qrels; contextual Chunk experiments are unavailable")
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
            "--dataset", dataset_format,
        ]
        if dataset_format == "beir":
            command.extend(["--beir-split", beir_split])
        if spec.use_context and context_dir:
            command.extend(["--context-dir", context_dir])
        if spec.method in {"bm25", "hybrid"}:
            command.extend(["--lexical-backend", lexical_backend])
            if lexical_backend == "elasticsearch":
                command.extend([
                    "--es-url", es_url,
                    "--es-index", f"{index_namespace}_{spec.experiment_id.lower()}",
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



def plan_model_sweep(
    models: list[str],
    experiment_ids: list[str],
    *,
    out_root: str,
    data_dir: str,
    context_dir: str | None = None,
    lexical_backend: str = "elasticsearch",
    es_url: str = "http://localhost:9200",
    rerank_model: str | None = None,
    rerank_mode: str = "fixed",
    allow_paid_api: bool = False,
    dataset_format: str = "anthropic",
    beir_split: str = "test",
) -> list[dict]:
    """One fully isolated run namespace per Embedding model (dry run only).

    API credentials are never resolved here; execution is performed by CLI
    after global preflight and requires --allow-paid-api plus --execute.
    """
    import hashlib
    import re
    from pathlib import Path

    if not models or len(set(models)) != len(models):
        raise ValueError("Provide distinct nonempty embedding model IDs")
    if any(not name.strip() for name in models):
        raise ValueError("Empty embedding model ID")
    if not experiment_ids or all(
        experiment_spec(item).method == "bm25" for item in experiment_ids
    ):
        raise ValueError("Model sweep requires vector or hybrid experiments")
    results: list[dict] = []
    for model in models:
        safe = re.sub(r"[^a-z0-9_-]+", "-", model.lower()).strip("-")[:36] or "model"
        suffix = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
        label = f"{safe}-{suffix}"
        plans = plan_matrix(
            experiment_ids,
            data_dir=data_dir,
            out_root=str(Path(out_root) / label),
            context_dir=context_dir,
            lexical_backend=lexical_backend,
            es_url=es_url,
            embedding_model=model,
            rerank_model=rerank_model,
            rerank_mode=rerank_mode,
            allow_paid_api=allow_paid_api,
            dataset_format=dataset_format,
            beir_split=beir_split,
            index_namespace=f"crlab_sweep_{safe}_{suffix}",
        )
        results.append({
            "model": model,
            "run_namespace": label,
            "status": "ready" if all(p["status"] == "ready" for p in plans) else "blocked",
            "experiments": plans,
        })
    return results
