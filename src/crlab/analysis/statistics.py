"""Pure-Python, seeded paired resampling for a fixed query universe."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import mean


def _validated(values: Sequence[float], count: int, name: str) -> None:
    if not values:
        raise ValueError("At least one paired observation is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("All observations must be finite")
    if count < 100:
        raise ValueError(f"{name} must be >= 100 to avoid very coarse resampling")


def _units(values: Sequence[float], groups: Sequence[str] | None) -> list[list[float]]:
    """Cluster is resampled as the unit while aggregate remains query-weighted."""
    if groups is None:
        return [[value] for value in values]
    if len(groups) != len(values):
        raise ValueError("Group label count must match query count")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Group labels must be nonempty strings")
    units: dict[str, list[float]] = defaultdict(list)
    for label, value in zip(groups, values, strict=True):
        units[label].append(value)
    return list(units.values())


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lo, hi = int(position), min(int(position) + 1, len(sorted_values) - 1)
    weight = position - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
    groups: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Percentile CI for mean(candidate-baseline); not a causal claim.

    If groups are supplied, whole groups are sampled with replacement.
    Resampled group sizes may differ, but means stay query-weighted.
    """
    _validated(deltas, resamples, "resamples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    units = _units(deltas, groups)
    if all(delta == 0 for delta in deltas):
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(units)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        count = 0
        for _ in range(n):
            selected = units[rng.randrange(n)]
            total += sum(selected)
            count += len(selected)
        means.append(total / count)
    means.sort()
    alpha = (1 - confidence) / 2
    return (_percentile(means, alpha), _percentile(means, 1 - alpha))


def paired_signflip_pvalue(
    deltas: Sequence[float],
    *,
    permutations: int = 5000,
    seed: int = 42,
    groups: Sequence[str] | None = None,
) -> float:
    """Two-sided Monte-Carlo paired sign-flip test (or cluster sign-flip).

    P-value uses the +1 correction; nonzero p is never rounded to zero.
    Assumes within-unit difference distribution is sign-exchangeable.
    """
    _validated(deltas, permutations, "permutations")
    units = _units(deltas, groups)
    observed = abs(mean(deltas))
    if observed == 0:
        return 1.0
    totals = [sum(unit) for unit in units]
    denominator = sum(map(len, units))
    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(permutations):
        synthetic = sum(
            total if rng.getrandbits(1) else -total for total in totals
        ) / denominator
        if abs(synthetic) >= observed - 1e-12:
            count_extreme += 1
    return (count_extreme + 1) / (permutations + 1)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down correction preserving original order."""
    n = len(pvalues)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("P-values must be in [0,1]")
    adjusted = [1.0] * n
    max_adjusted = 0.0
    for rank, index in enumerate(sorted(range(n), key=lambda i: pvalues[i])):
        max_adjusted = max(max_adjusted, min(1.0, (n - rank) * pvalues[index]))
        adjusted[index] = max_adjusted
    return adjusted
