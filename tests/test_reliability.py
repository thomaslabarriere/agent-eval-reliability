import pytest

from core.reliability import build_report


def _outcomes(pattern, n):
    """Helper: build {case_id: [bool]*n} from a dict of case_id -> pass count."""
    return {cid: [True] * k + [False] * (n - k) for cid, k in pattern.items()}


def test_big_delta_can_still_be_noise():
    # The whole point: a large raw delta on few, high-variance cases is not
    # necessarily real. Version B "wins" 3 of 4 cases but each case ran twice.
    a = _outcomes({"c1": 2, "c2": 0, "c3": 2, "c4": 0}, n=2)
    b = _outcomes({"c1": 2, "c2": 2, "c3": 0, "c4": 2}, n=2)
    rep = build_report(a, b, iters=5000, seed=1)
    assert abs(rep.delta_ci.delta) >= 0.2  # a visible raw delta
    assert rep.test.significant is False   # but not distinguishable from noise


def test_small_consistent_delta_is_significant():
    # B beats A on every one of many cases by a small, consistent margin.
    a = _outcomes({f"c{i}": 3 for i in range(20)}, n=10)
    b = _outcomes({f"c{i}": 6 for i in range(20)}, n=10)
    rep = build_report(a, b, iters=5000, seed=1)
    assert rep.test.significant is True
    assert rep.delta_ci.low > 0  # CI on the delta excludes zero


def test_flaky_cases_are_flagged():
    a = _outcomes({"stable": 5, "flaky": 2}, n=5)
    b = _outcomes({"stable": 5, "flaky": 5}, n=5)
    rep = build_report(a, b, iters=1000, seed=0)
    assert rep.flaky_cases == ["flaky"]


def test_overall_rate_pools_all_runs():
    a = _outcomes({"c1": 5, "c2": 5}, n=10)  # 10/20
    b = _outcomes({"c1": 10, "c2": 10}, n=10)  # 20/20
    rep = build_report(a, b, iters=500, seed=0)
    assert rep.overall_a.point == pytest.approx(0.5)
    assert rep.overall_b.point == pytest.approx(1.0)
    assert rep.runs_per_case == 10


def test_mismatched_case_ids_rejected():
    a = _outcomes({"c1": 1}, n=2)
    b = _outcomes({"c2": 1}, n=2)
    with pytest.raises(ValueError):
        build_report(a, b, iters=100)
