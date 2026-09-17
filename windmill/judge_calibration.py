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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import JudgeReport, recommend_threshold, score_judge  # noqa: E402


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
    if gold_labels.keys() != judge_scores.keys():
        raise ValueError("gold_labels and judge_scores must cover the same case ids")
    case_ids = sorted(gold_labels)
    gold = [bool(gold_labels[c]) for c in case_ids]
    scores = [float(judge_scores[c]) for c in case_ids]

    current = score_judge(gold, [s >= current_pass_if for s in scores])
    best = recommend_threshold(gold, scores, objective=objective)

    return {
        "n": len(case_ids),
        "current": {"pass_if": current_pass_if, **_report_to_dict(current)},
        "recommended": {
            "pass_if": best.pass_if,
            "objective": best.objective,
            **_report_to_dict(best.report),
        },
    }


def _render(r: dict) -> str:
    cur, rec = r["current"], r["recommended"]

    def line(tag, d):
        c = d["confusion"]
        return (
            f"  {tag} (pass_if={d['pass_if']:.2f}): agreement {d['agreement']:.0%}, "
            f"kappa {d['kappa']:.2f}, precision {d['precision']:.0%}, recall {d['recall']:.0%}, "
            f"fp {c['fp']} fn {c['fn']}"
        )

    return "\n".join([
        f"Judge calibration over {r['n']} labelled cases",
        line("current", cur),
        line("recommended", rec),
    ])


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "examples", "judge_labeled.json")) as fh:
        data = json.load(fh)
    out = main(data["gold_labels"], data["judge_scores"], data.get("current_pass_if", 0.5))
    print(_render(out))
