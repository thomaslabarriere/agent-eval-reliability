"""Windmill script: how well does your Agent-scorer judge match humans?

Drop-in runnable for a Windmill workspace. Given gold pass/fail labels and the
judge's raw scores per case, it reports the judge's agreement, precision/recall,
confusion matrix, and Cohen's kappa at the current `pass_if` threshold, and
recommends the `pass_if` that best matches the humans.

Reads recorded judge scores rather than calling the judge, so the verdict is
reproducible. Feeding live Agent-scorer scores into the same `main` calibrates a
real judge.

Local demo (no Windmill, no API key needed):
    python -m windmill.judge_calibration
"""

from __future__ import annotations

import json
import os
import sys

# Local-run convenience so `python -m windmill.x` finds `core`. In a Windmill
# workspace `core` is synced as workspace scripts or pasted inline (it is pure
# stdlib); this shim is not the workspace mechanism.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import (  # noqa: E402
    JudgeReport,
    cross_validated_agreement,
    recommend_threshold,
    score_judge,
)


def _as_bool(x: object) -> bool:
    """Coerce a JSON gold label to bool, rejecting anything ambiguous.

    `bool("false")` is True, so a label arriving as a string would silently flip.
    Accept only genuine booleans or the ints 0/1.
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, int) and x in (0, 1):
        return bool(x)
    raise ValueError(f"gold label must be a boolean (or 0/1), got {x!r}")


def _report_to_dict(rep: JudgeReport) -> dict:
    c = rep.confusion
    return {
        "confusion": {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn},
        "agreement": rep.agreement,
        "precision": rep.precision,
        "recall": rep.recall,
        "f1": rep.f1,
        "kappa": rep.kappa,
    }


def main(
    gold_labels: dict,
    judge_scores: dict,
    current_pass_if: float = 0.5,
    objective: str = "agreement",
) -> dict:
    """Windmill entrypoint.

    `gold_labels`: {case_id: true/false} human labels.
    `judge_scores`: {case_id: float in [0,1]} the judge's raw score per case.
    """
    if not isinstance(gold_labels, dict) or not isinstance(judge_scores, dict):
        raise ValueError("gold_labels and judge_scores must both be dicts of case_id -> value")
    if gold_labels.keys() != judge_scores.keys():
        raise ValueError("gold_labels and judge_scores must cover the same case ids")
    if not 0.0 <= current_pass_if <= 1.0:
        raise ValueError(f"current_pass_if must be in [0, 1], got {current_pass_if}")
    case_ids = sorted(gold_labels)
    gold = [_as_bool(gold_labels[c]) for c in case_ids]
    scores = []
    for c in case_ids:
        s = float(judge_scores[c])
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"judge score for {c!r} must be in [0, 1], got {s}")
        scores.append(s)

    current = score_judge(gold, [s >= current_pass_if for s in scores])
    best = recommend_threshold(gold, scores, objective=objective)
    # The recommended threshold is fit on these same labels, so its in-sample
    # score is optimistic. Report a cross-validated agreement so the number that
    # travels is not the one the threshold was tuned on.
    cv = cross_validated_agreement(gold, scores, objective=objective)

    return {
        "n": len(case_ids),
        "current": {"pass_if": current_pass_if, **_report_to_dict(current)},
        "recommended": {
            "pass_if": best.pass_if,
            "objective": best.objective,
            **_report_to_dict(best.report),
        },
        "cross_validated_agreement": {
            "mean": cv.mean_agreement,
            "k": cv.k,
            "folds": cv.fold_agreements,  # spread; the mean is noisy on small sets
        },
    }


def _render(r: dict) -> str:
    cur, rec = r["current"], r["recommended"]

    def line(tag, d):
        c = d["confusion"]
        return (
            f"  {tag:<11} (pass_if={d['pass_if']:.2f}): agreement {d['agreement']:.0%}, "
            f"kappa {d['kappa']:.2f}, precision {d['precision']:.0%}, recall {d['recall']:.0%}, "
            f"fp {c['fp']} fn {c['fn']}"
        )

    cv = r["cross_validated_agreement"]
    return "\n".join([
        f"Judge calibration over {r['n']} labeled cases",
        line("current", cur),
        line("recommended", rec),
        f"  cross-validated agreement (k={cv['k']}): {cv['mean']:.0%}"
        " (held-out; the recommended threshold's in-sample number is optimistic)",
    ])


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "examples", "judge_labeled.json")) as fh:
        data = json.load(fh)
    out = main(data["gold_labels"], data["judge_scores"], data.get("current_pass_if", 0.5))
    print(_render(out))
