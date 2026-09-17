"""Calibrate an LLM judge against human gold labels.

Windmill's Agent scorer lets an LLM decide pass/fail, with a `pass_if` threshold
turning a score into a boolean, but nothing measures whether that judge agrees
with a human. This module does: given gold labels and the judge's verdicts, it
reports agreement, precision/recall, the confusion matrix, and Cohen's kappa
(agreement above chance). Given the judge's raw scores, it recommends the
`pass_if` threshold that best matches the humans.

"pass" is treated as the positive class throughout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Confusion:
    tp: int  # gold pass, judge pass
    fp: int  # gold fail, judge pass  (judge too lenient)
    fn: int  # gold pass, judge fail  (judge too strict)
    tn: int  # gold fail, judge fail

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def confusion_matrix(gold: list[bool], judge: list[bool]) -> Confusion:
    if len(gold) != len(judge):
        raise ValueError(f"gold and judge must align, got {len(gold)} and {len(judge)}")
    if not gold:
        raise ValueError("need at least one labelled case")
    tp = fp = fn = tn = 0
    for g, j in zip(gold, judge):
        if g and j:
            tp += 1
        elif not g and j:
            fp += 1
        elif g and not j:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, fn, tn)


def agreement(c: Confusion) -> float:
    """Fraction of cases where judge and gold agree (a.k.a. accuracy)."""
    return (c.tp + c.tn) / c.n


def precision(c: Confusion) -> float:
    """Of the cases the judge passed, how many the humans also passed."""
    denom = c.tp + c.fp
    return c.tp / denom if denom else 0.0


def recall(c: Confusion) -> float:
    """Of the cases the humans passed, how many the judge also passed."""
    denom = c.tp + c.fn
    return c.tp / denom if denom else 0.0


def f1(c: Confusion) -> float:
    p, r = precision(c), recall(c)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def cohens_kappa(gold: list[bool], judge: list[bool]) -> float:
    """Cohen's kappa: agreement corrected for what chance alone would give.

    1.0 is perfect, 0.0 is chance-level, negative is worse than chance. A judge
    can have high raw agreement yet a low kappa when the labels are imbalanced,
    which is precisely when raw agreement is misleading.
    """
    c = confusion_matrix(gold, judge)
    n = c.n
    po = agreement(c)
    p_gold_pass = (c.tp + c.fn) / n
    p_judge_pass = (c.tp + c.fp) / n
    pe = p_gold_pass * p_judge_pass + (1 - p_gold_pass) * (1 - p_judge_pass)
    if pe >= 1.0:
        # No variability to disagree over (everything one class): define as perfect.
        return 1.0
    return (po - pe) / (1 - pe)


@dataclass(frozen=True)
class JudgeReport:
    confusion: Confusion
    agreement: float
    precision: float
    recall: float
    f1: float
    kappa: float


def score_judge(gold: list[bool], judge: list[bool]) -> JudgeReport:
    c = confusion_matrix(gold, judge)
    return JudgeReport(
        confusion=c,
        agreement=agreement(c),
        precision=precision(c),
        recall=recall(c),
        f1=f1(c),
        kappa=cohens_kappa(gold, judge),
    )


@dataclass(frozen=True)
class ThresholdChoice:
    pass_if: float  # recommended threshold: judge passes when score >= pass_if
    objective: str
    value: float  # objective value at that threshold
    report: JudgeReport


def recommend_threshold(
    gold: list[bool],
    scores: list[float],
    objective: str = "agreement",
) -> ThresholdChoice:
    """Recommend the `pass_if` threshold that best matches the gold labels.

    Sweeps every candidate boundary (each distinct score, plus one above the max
    so "reject all" is reachable), applying judge_pass = score >= threshold, and
    picks the threshold maximizing `objective` ("agreement", "f1", or "kappa").
    Ties break toward the higher (stricter) threshold, which is the safer default
    for a gate. This maps directly onto Windmill's `ScorerDef` `pass_if` field.
    """
    if len(gold) != len(scores):
        raise ValueError(f"gold and scores must align, got {len(gold)} and {len(scores)}")
    if not gold:
        raise ValueError("need at least one labelled case")
    objectives = {"agreement": agreement, "f1": f1, "kappa": None}
    if objective not in objectives:
        raise ValueError(f"unknown objective {objective!r}, expected one of {sorted(objectives)}")

    candidates = sorted(set(scores) | {max(scores) + 1e-9})
    best: ThresholdChoice | None = None
    for thr in candidates:
        judge = [s >= thr for s in scores]
        rep = score_judge(gold, judge)
        value = rep.kappa if objective == "kappa" else objectives[objective](rep.confusion)
        if best is None or value > best.value or (value == best.value and thr > best.pass_if):
            best = ThresholdChoice(pass_if=thr, objective=objective, value=value, report=rep)
    assert best is not None
    return best
