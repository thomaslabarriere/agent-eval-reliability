"""Turn repeated run outcomes into an auditable reliability report.

Input is the raw per-run pass/fail for two agent versions on the same cases,
each case run several times. Output is a plain data structure: per-case Wilson
intervals and a flakiness flag, and an overall pass rate per version with a
Wilson interval, plus a bootstrap CI and a permutation p-value on the delta.

The verdict is computed here, from the numbers, so it can be recomputed and
checked; nothing is taken on trust from a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .resampling import DeltaCI, PermutationResult, paired_bootstrap_ci, paired_permutation_test
from .wilson import Interval, Z_95, wilson_interval

# A case's outcomes = the ordered pass/fail of each repeated run.
CaseOutcomes = dict[str, list[bool]]


@dataclass(frozen=True)
class CaseReport:
    case_id: str
    interval_a: Interval
    interval_b: Interval
    flaky_a: bool  # some but not all runs passed for version A
    flaky_b: bool


@dataclass(frozen=True)
class ReliabilityReport:
    version_a: str
    version_b: str
    runs_per_case: int
    overall_a: Interval
    overall_b: Interval
    delta_ci: DeltaCI
    test: PermutationResult
    cases: list[CaseReport] = field(default_factory=list)

    @property
    def flaky_cases(self) -> list[str]:
        return [c.case_id for c in self.cases if c.flaky_a or c.flaky_b]


def _is_flaky(outcomes: list[bool]) -> bool:
    return 0 < sum(outcomes) < len(outcomes)


def _aligned_case_ids(a: CaseOutcomes, b: CaseOutcomes) -> list[str]:
    if a.keys() != b.keys():
        only_a = sorted(a.keys() - b.keys())
        only_b = sorted(b.keys() - a.keys())
        raise ValueError(f"case ids differ between versions; only in A: {only_a}, only in B: {only_b}")
    if not a:
        raise ValueError("no cases")
    return sorted(a.keys())


def build_report(
    outcomes_a: CaseOutcomes,
    outcomes_b: CaseOutcomes,
    version_a: str = "A",
    version_b: str = "B",
    z: float = Z_95,
    iters: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> ReliabilityReport:
    """Build the full reliability report from repeated run outcomes.

    `outcomes_a`/`outcomes_b` map a case id to the list of pass/fail booleans
    from that case's repeated runs. Both versions must cover the same case ids.
    """
    case_ids = _aligned_case_ids(outcomes_a, outcomes_b)

    runs = len(outcomes_a[case_ids[0]])
    for cid in case_ids:
        if len(outcomes_a[cid]) == 0 or len(outcomes_b[cid]) == 0:
            raise ValueError(f"case {cid} has no runs for one version")

    cases: list[CaseReport] = []
    rates_a: list[float] = []
    rates_b: list[float] = []
    passes_a = trials_a = passes_b = trials_b = 0
    for cid in case_ids:
        oa, ob = outcomes_a[cid], outcomes_b[cid]
        ia = wilson_interval(sum(oa), len(oa), z)
        ib = wilson_interval(sum(ob), len(ob), z)
        cases.append(CaseReport(cid, ia, ib, _is_flaky(oa), _is_flaky(ob)))
        rates_a.append(sum(oa) / len(oa))
        rates_b.append(sum(ob) / len(ob))
        passes_a += sum(oa); trials_a += len(oa)
        passes_b += sum(ob); trials_b += len(ob)

    overall_a = wilson_interval(passes_a, trials_a, z)
    overall_b = wilson_interval(passes_b, trials_b, z)
    delta_ci = paired_bootstrap_ci(rates_a, rates_b, iters=iters, seed=seed, alpha=alpha)
    test = paired_permutation_test(rates_a, rates_b, iters=iters, seed=seed, alpha=alpha)

    return ReliabilityReport(
        version_a=version_a,
        version_b=version_b,
        runs_per_case=runs,
        overall_a=overall_a,
        overall_b=overall_b,
        delta_ci=delta_ci,
        test=test,
        cases=cases,
    )
