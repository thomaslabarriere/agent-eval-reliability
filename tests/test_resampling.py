import pytest

from core.resampling import paired_bootstrap_ci, paired_permutation_test


def test_identical_arms_are_not_significant():
    rates = [0.6, 0.4, 0.8, 0.5, 0.7]
    res = paired_permutation_test(rates, list(rates), iters=2000, seed=1)
    assert res.delta == 0.0
    assert res.p_value > 0.5
    assert res.significant is False


def test_large_consistent_gap_is_significant():
    # B beats A on every case by a wide, consistent margin.
    a = [0.1, 0.2, 0.0, 0.15, 0.05, 0.1, 0.2, 0.0]
    b = [0.9, 0.95, 1.0, 0.85, 0.9, 0.95, 1.0, 0.9]
    res = paired_permutation_test(a, b, iters=5000, seed=1)
    assert res.delta > 0.6
    assert res.p_value < 0.05
    assert res.significant is True


def test_permutation_pvalue_never_zero():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0]
    res = paired_permutation_test(a, b, iters=100, seed=0)
    assert res.p_value > 0.0  # observed assignment is always counted


def test_bootstrap_ci_brackets_observed_delta():
    a = [0.2, 0.3, 0.25, 0.35, 0.4]
    b = [0.5, 0.55, 0.6, 0.5, 0.65]
    ci = paired_bootstrap_ci(a, b, iters=3000, seed=2)
    assert ci.low <= ci.delta <= ci.high


def test_bootstrap_ci_narrows_with_more_cases():
    # Same effect size and (non-zero) between-case spread, more cases -> tighter
    # interval. The per-case deltas here are 0.1 and 0.5, so there is real
    # variance for the standard error to shrink as n grows.
    small_a = [0.2, 0.2]
    small_b = [0.3, 0.7]
    big_a = [0.2, 0.2] * 25
    big_b = [0.3, 0.7] * 25
    wide = paired_bootstrap_ci(small_a, small_b, iters=3000, seed=3)
    narrow = paired_bootstrap_ci(big_a, big_b, iters=3000, seed=3)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_seed_is_deterministic():
    a = [0.3, 0.5, 0.2, 0.8]
    b = [0.6, 0.4, 0.5, 0.9]
    r1 = paired_permutation_test(a, b, iters=1000, seed=7)
    r2 = paired_permutation_test(a, b, iters=1000, seed=7)
    assert r1.p_value == r2.p_value


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([0.1, 0.2], [0.3], iters=10)
