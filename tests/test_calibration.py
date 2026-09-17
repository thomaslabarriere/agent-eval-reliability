import pytest

from core.calibration import (
    cohens_kappa,
    confusion_matrix,
    f1,
    precision,
    recall,
    recommend_threshold,
    score_judge,
)


def test_confusion_counts_and_metrics():
    gold = [True, True, True, False, False]
    judge = [True, False, True, True, False]  # 2 tp, 1 fn, 1 fp, 1 tn
    c = confusion_matrix(gold, judge)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)
    assert precision(c) == pytest.approx(2 / 3)
    assert recall(c) == pytest.approx(2 / 3)
    assert f1(c) == pytest.approx(2 / 3)


def test_false_positive_vs_false_negative_are_distinct():
    # A lenient judge (passes a case humans failed) vs a strict one (fails a
    # case humans passed) must not look identical.
    lenient = confusion_matrix([False], [True])   # fp
    strict = confusion_matrix([True], [False])    # fn
    assert (lenient.fp, lenient.fn) == (1, 0)
    assert (strict.fp, strict.fn) == (0, 1)


def test_kappa_below_agreement_on_imbalanced_labels():
    # 18/20 pass. A judge that blindly says "pass" scores 90% agreement but adds
    # nothing over chance -> kappa near 0. This is the trap raw agreement hides.
    gold = [True] * 18 + [False] * 2
    judge = [True] * 20
    rep = score_judge(gold, judge)
    assert rep.agreement == pytest.approx(0.9)
    assert rep.kappa < 0.1


def test_kappa_perfect_when_judge_matches_gold():
    gold = [True, False, True, False, True]
    assert cohens_kappa(gold, list(gold)) == pytest.approx(1.0)


def test_recommend_threshold_picks_best_separator():
    # Scores clearly separate pass (>=0.6) from fail (<0.6).
    scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.1]
    gold = [True, True, True, False, False, False]
    choice = recommend_threshold(gold, scores, objective="agreement")
    assert choice.value == pytest.approx(1.0)  # perfect separation is achievable
    assert 0.4 < choice.pass_if <= 0.7         # boundary sits between the classes


def test_recommend_threshold_ties_break_stricter():
    # Two thresholds both give perfect agreement; the stricter (higher) wins.
    scores = [0.9, 0.8, 0.2, 0.1]
    gold = [True, True, False, False]
    choice = recommend_threshold(gold, scores, objective="agreement")
    assert choice.pass_if == pytest.approx(0.8)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        confusion_matrix([True, False], [True])
