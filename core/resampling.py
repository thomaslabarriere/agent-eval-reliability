"""Paired resampling for comparing two agent versions on the same cases.

Windmill runs an eval as a flow over a dataset and reports the pass-rate delta
between two agent versions as a raw subtraction. Because both versions are scored
on the *same* cases, the comparison is paired, and a raw delta says nothing about
whether it would survive re-running. Two tools here answer that:

- `paired_bootstrap_ci`: a confidence interval on the mean per-case delta, by
  resampling cases with replacement (captures between-case variability).
- `paired_permutation_test`: a p-value for "no difference", by randomly swapping
  the A/B label within each case (the exact null for a paired comparison).

Both take one pass rate per case per version (each typically the mean of N
repeated runs), keep the pairing, are seeded, and use only the stdlib.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


@dataclass(frozen=True)
class DeltaCI:
    delta: float  # mean(B) - mean(A) over cases
    low: float
    high: float


def _check_pairs(a: list[float], b: list[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired inputs must be equal length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("need at least one case")


def _check_resampling(iters: int, alpha: float) -> None:
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def paired_bootstrap_ci(
    rates_a: list[float],
    rates_b: list[float],
    iters: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> DeltaCI:
    """Percentile bootstrap CI on the mean per-case delta (B - A).

    Resamples the *cases* with replacement, keeping each case's (A, B) pair
    together so the pairing is preserved.
    """
    _check_pairs(rates_a, rates_b)
    _check_resampling(iters, alpha)
    n = len(rates_a)
    observed = _mean(rates_b) - _mean(rates_a)
    rng = random.Random(seed)
    deltas = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(rates_a[i] for i in idx) / n
        db = sum(rates_b[i] for i in idx) / n
        deltas.append(db - da)
    deltas.sort()
    lo = deltas[int((alpha / 2.0) * iters)]
    hi = deltas[min(iters - 1, int((1.0 - alpha / 2.0) * iters))]
    return DeltaCI(delta=observed, low=lo, high=hi)


@dataclass(frozen=True)
class PermutationResult:
    delta: float
    p_value: float
    significant: bool

    def verdict(self, alpha: float = 0.05) -> str:  # pragma: no cover - display
        if self.p_value <= alpha:
            return f"significant (delta={self.delta:+.1%}, p={self.p_value:.3f})"
        return f"not distinguishable from noise (delta={self.delta:+.1%}, p={self.p_value:.3f})"


def paired_permutation_test(
    rates_a: list[float],
    rates_b: list[float],
    iters: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PermutationResult:
    """Two-sided paired permutation test on the mean per-case delta.

    Under the null (the two versions are interchangeable) swapping A and B within
    a case is equally likely, so we flip each case's sign independently and count
    how often the permuted |mean delta| reaches the observed one. The observed
    assignment is included, so the p-value is never 0.
    """
    _check_pairs(rates_a, rates_b)
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(rates_a)
    diffs = [b - a for a, b in zip(rates_a, rates_b)]
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)
    at_least_as_extreme = 1  # count the observed assignment itself
    for _ in range(iters):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed - 1e-12:
            at_least_as_extreme += 1
    p_value = at_least_as_extreme / (iters + 1)
    delta = sum(diffs) / n
    return PermutationResult(delta=delta, p_value=p_value, significant=p_value <= alpha)
