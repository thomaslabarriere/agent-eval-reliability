"""Wilson score interval for a binomial proportion.

Windmill's eval UI shows a pass rate (passed / scored) as a bare point. With a
finite number of scored runs that point has real uncertainty, and the naive
normal approximation (phat +/- z*sqrt(phat*(1-phat)/n)) is badly behaved near 0
and 1 and can even leave [0, 1]. The Wilson interval stays inside [0, 1], is
asymmetric toward 0.5, and is well-behaved for small n, which is exactly the
regime an eval dataset lives in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z for a 95% two-sided interval. Kept explicit so callers can widen/narrow it.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    point: float  # observed proportion (passed / n)
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for `successes` out of `n` trials.

    `point` is the plain observed proportion; `low`/`high` are the Wilson bounds,
    clamped to [0, 1]. For n == 0 the interval is the whole [0, 1] with an
    undefined point reported as 0.0, since nothing is known yet.
    """
    if successes < 0 or n < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes}, n={n}")
    if n == 0:
        return Interval(point=0.0, low=0.0, high=1.0)

    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return Interval(point=phat, low=low, high=high)
