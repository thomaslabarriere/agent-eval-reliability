import pytest

from core.calibration import (
    cohens_kappa,
    confusion_matrix,
    cross_validated_agreement,
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


def test_recommend_threshold_genuine_tie_breaks_stricter():
    # Both candidate thresholds give identical agreement (0.5); the stricter
    # (higher) one must win. Reversing the tie-break makes this fail.
    choice = recommend_threshold([True, False], [0.5, 0.5], objective="agreement")
    assert choice.pass_if > 0.5


def test_recommend_threshold_kappa_objective():
    # Exercises the kappa branch of the objective dispatch.
    scores = [0.9, 0.8, 0.2, 0.1]
    gold = [True, True, False, False]
    choice = recommend_threshold(gold, scores, objective="kappa")
    assert choice.objective == "kappa"
    assert choice.report.kappa == pytest.approx(1.0)


def test_cross_validation_does_not_leak_test_into_train():
    # Signal-free labels: a single threshold can overfit the noise in-sample, but
    # it must not generalize. Honest CV collapses near/below chance; if a fold's
    # training set leaked the held-out rows, the threshold would be fit on the
    # test rows too and CV would rebound to the in-sample optimum. The gap is the
    # test: it fails if the function trains on all the data (measured honest CV
    # 0.375 vs leaked 0.583 on this fixture).
    import random as _random

    n = 24
    scores = [(i + 1) / (n + 1) for i in range(n)]  # distinct, sorted
    rng = _random.Random(42)
    gold = [rng.random() < 0.5 for _ in range(n)]

    in_sample = recommend_threshold(gold, scores, "agreement").value
    cv = cross_validated_agreement(gold, scores, k=6, seed=0).mean_agreement
    assert in_sample >= 0.55          # in-sample can overfit the noise
    assert cv <= 0.45                 # honest held-out is near/below chance
    assert cv < in_sample - 0.1       # a train/test leak would erase this gap


def test_cross_validation_is_deterministic():
    gold = [True, False] * 6
    scores = [i / 12 for i in range(12)]
    a = cross_validated_agreement(gold, scores, k=4, seed=0).mean_agreement
    b = cross_validated_agreement(gold, scores, k=4, seed=0).mean_agreement
    assert a == b


def test_cross_validation_caps_k_at_n():
    gold = [True, False, True, False]
    scores = [0.9, 0.1, 0.8, 0.2]
    cv = cross_validated_agreement(gold, scores, k=999)
    assert cv.k == len(gold)  # capped to leave-one-out


def test_cross_validated_agreement_high_on_separable_data():
    gold = [True] * 8 + [False] * 8
    scores = [0.9] * 8 + [0.1] * 8  # perfectly separable
    cv = cross_validated_agreement(gold, scores, k=4, seed=1)
    assert cv.mean_agreement == pytest.approx(1.0)


def test_cross_validation_needs_enough_cases():
    with pytest.raises(ValueError):
        cross_validated_agreement([True], [0.5], k=5)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        confusion_matrix([True, False], [True])
