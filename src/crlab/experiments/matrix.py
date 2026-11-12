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
