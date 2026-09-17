import pytest

from windmill.delta_reliability import main as delta_main
from windmill.judge_calibration import main as judge_main


def test_delta_rejects_stringly_typed_outcomes():
    # The whole point: "false" must not silently count as a pass (bool("false") is True).
    with pytest.raises(ValueError):
        delta_main({"c1": ["false", "false"]}, {"c1": [True, True]}, iters=100)


def test_delta_accepts_zero_one_ints():
    out = delta_main({"c1": [1, 0]}, {"c1": [1, 1]}, iters=200, seed=0)
    assert out["overall"]["a"]["pass_rate"] == pytest.approx(0.5)
    assert out["overall"]["b"]["pass_rate"] == pytest.approx(1.0)


def test_delta_rejects_non_list_value():
    with pytest.raises(ValueError):
        delta_main({"c1": True}, {"c1": [True]}, iters=100)


def test_judge_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        judge_main({"c1": True, "c2": False}, {"c1": 50, "c2": 0.1})


def test_judge_reports_cross_validated_agreement():
    gold = {f"c{i}": (i % 2 == 0) for i in range(8)}
    scores = {f"c{i}": (0.9 if i % 2 == 0 else 0.1) for i in range(8)}
    out = judge_main(gold, scores)
    assert "cross_validated_agreement" in out
    assert 0.0 <= out["cross_validated_agreement"]["mean"] <= 1.0
