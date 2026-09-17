"""Windmill script: is the pass-rate delta between two agent versions real?

Drop-in runnable for a Windmill workspace. It takes the recorded pass/fail of
each case's repeated runs for two agent versions and returns, as structured
output, a Wilson interval on each version's pass rate, a bootstrap CI and a
permutation p-value on the delta, and the list of flaky cases.

It reads recorded outcomes rather than calling an LLM, so a reviewer can re-run
it and get the same verdict. Wiring it to live Windmill runs is a matter of
feeding real per-run outcomes into the same `main`.

Local demo (no Windmill, no API key needed):
    python -m windmill.delta_reliability
"""

from __future__ import annotations

import json
import os
import sys

# Make `core` importable when run locally as `python -m windmill.x`. Inside a
# Windmill workspace `core` is not on this path; because the core is pure stdlib,
# the intended path there is to sync it as workspace scripts (wmill / git sync)
# or paste it inline. This shim is a local-run convenience, not the workspace
# mechanism.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reliability import ReliabilityReport, build_report  # noqa: E402


def _as_bool(x: object) -> bool:
    """Coerce a JSON run outcome to bool, rejecting anything ambiguous.

    A dataset column may arrive as the string "false" or "0"; `bool("false")` is
    True, which would silently count a failure as a pass and corrupt every number
    downstream. Accept only genuine booleans or the ints 0/1.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and x in (0, 1):
        return bool(x)
    raise ValueError(f"run outcome must be a boolean (or 0/1), got {x!r}")


def _coerce_outcomes(outcomes: object, name: str) -> dict:
    if not isinstance(outcomes, dict):
        raise ValueError(f"{name} must be a dict of case_id -> list, got {type(outcomes).__name__}")
    coerced = {}
    for cid, runs in outcomes.items():
        if not isinstance(runs, list):
            raise ValueError(f"{name}[{cid!r}] must be a list of run outcomes, got {type(runs).__name__}")
        coerced[cid] = [_as_bool(x) for x in runs]
    return coerced


def _report_to_dict(rep: ReliabilityReport) -> dict:
    return {
        "versions": {"a": rep.version_a, "b": rep.version_b},
        "runs_per_case": rep.runs_per_case,
        "overall": {
            "a": {"pass_rate": rep.overall_a.point, "ci": [rep.overall_a.low, rep.overall_a.high]},
            "b": {"pass_rate": rep.overall_b.point, "ci": [rep.overall_b.low, rep.overall_b.high]},
        },
        "delta": {
            "value": rep.delta_ci.delta,
            "ci": [rep.delta_ci.low, rep.delta_ci.high],
            "p_value": rep.test.p_value,
            "significant": rep.test.significant,
        },
        "flaky_cases": rep.flaky_cases,
        "cases": [
            {
                "case_id": c.case_id,
                "a": {"pass_rate": c.interval_a.point, "ci": [c.interval_a.low, c.interval_a.high]},
                "b": {"pass_rate": c.interval_b.point, "ci": [c.interval_b.low, c.interval_b.high]},
                "flaky": c.flaky_a or c.flaky_b,
            }
            for c in rep.cases
        ],
    }


def main(
    outcomes_a: dict,
    outcomes_b: dict,
    version_a: str = "A",
    version_b: str = "B",
    iters: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Windmill entrypoint. `outcomes_a`/`outcomes_b`: {case_id: [true/false, ...]}."""
    rep = build_report(
        _coerce_outcomes(outcomes_a, "outcomes_a"),
        _coerce_outcomes(outcomes_b, "outcomes_b"),
        version_a=version_a,
        version_b=version_b,
        iters=iters,
        seed=seed,
        alpha=alpha,
    )
    return _report_to_dict(rep)


def _render(result: dict) -> str:
    d = result["delta"]
    verdict = "SIGNIFICANT" if d["significant"] else "not distinguishable from noise"
    lines = [
        f"{result['versions']['a']} vs {result['versions']['b']}"
        f"  ({result['runs_per_case']} runs/case)",
        f"  {result['versions']['a']}: {result['overall']['a']['pass_rate']:.1%}"
        f"  {result['versions']['b']}: {result['overall']['b']['pass_rate']:.1%}",
        f"  delta {d['value']:+.1%}  95% CI [{d['ci'][0]:+.1%}, {d['ci'][1]:+.1%}]"
        f"  p={d['p_value']:.3f}  -> {verdict}",
        f"  flaky cases: {result['flaky_cases'] or 'none'}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    fixture = os.path.join(here, "..", "examples", "runs_v1_vs_v2.json")
    with open(fixture) as fh:
        data = json.load(fh)
    out = main(data["outcomes_a"], data["outcomes_b"], data.get("version_a", "v1"), data.get("version_b", "v2"))
    print(_render(out))
