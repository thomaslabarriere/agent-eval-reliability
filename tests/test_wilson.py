import math

import pytest

from core.wilson import Z_95, wilson_interval


def test_matches_wilson_formula_not_normal_approx():
    # 8/10 at 95%. Wilson center/margin, computed independently here.
    n, k, z = 10, 8, Z_95
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z2 / (4 * n)) / n) / denom
    iv = wilson_interval(k, n)
    assert iv.point == pytest.approx(0.8)
    assert iv.low == pytest.approx(center - margin, abs=1e-9)
    assert iv.high == pytest.approx(center + margin, abs=1e-9)
    # The naive normal approx would be symmetric around phat; Wilson is not.
    normal_low = phat - z * math.sqrt(phat * (1 - phat) / n)
    assert iv.low != pytest.approx(normal_low, abs=1e-3)


def test_interval_is_asymmetric_toward_center():
    # For a lopsided proportion, the Wilson point is not the midpoint of [low, high].
    iv = wilson_interval(9, 10)
    midpoint = (iv.low + iv.high) / 2
    assert midpoint < iv.point  # pulled toward 0.5


def test_stays_inside_unit_interval_at_extremes():
    # A normal approximation gives 0 width (and can exceed 1) at k == n; Wilson does not.
    iv = wilson_interval(10, 10)
    assert iv.point == 1.0
    assert iv.high <= 1.0
    assert iv.low < 1.0  # real lower uncertainty, unlike the normal approx's 0 width


def test_zero_trials_is_full_uncertainty():
    iv = wilson_interval(0, 0)
    assert (iv.low, iv.high) == (0.0, 1.0)


def test_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
