"""The standalone Windmill files must run with no repo imports and stay in sync
with the tested core (rebuilt by scripts/build_standalone.py)."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_builder():
    spec = importlib.util.spec_from_file_location("builder", ROOT / "scripts" / "build_standalone.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exec_standalone(name: str) -> dict:
    path = ROOT / "windmill" / "standalone" / f"{name}_standalone.py"
    ns: dict = {}
    # __name__ is not "__main__", so the demo block does not run on import.
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns


def test_standalone_files_are_in_sync_with_core():
    builder = _load_builder()
    for name, spec in builder.TARGETS.items():
        regenerated = builder.build(name, spec)
        committed = (ROOT / "windmill" / "standalone" / f"{name}_standalone.py").read_text()
        assert regenerated == committed, (
            f"{name}_standalone.py is stale; run `python scripts/build_standalone.py`"
        )


def test_delta_standalone_runs_without_repo():
    ns = _exec_standalone("delta_reliability")
    out = ns["main"]({"c1": [True, False], "c2": [True, True]},
                     {"c1": [True, True], "c2": [True, True]}, iters=200, seed=0)
    assert "delta" in out and "significant" in out["delta"]
    assert set(out["overall"]) == {"a", "b"}


def test_judge_standalone_runs_without_repo():
    ns = _exec_standalone("judge_calibration")
    out = ns["main"]({"c1": True, "c2": False}, {"c1": 0.9, "c2": 0.2})
    assert "recommended" in out and "cross_validated_agreement" in out


def test_standalone_has_no_repo_imports():
    for name in ("delta_reliability", "judge_calibration"):
        text = (ROOT / "windmill" / "standalone" / f"{name}_standalone.py").read_text()
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("from core"), line
            assert not stripped.startswith("from ."), line
            assert not stripped.startswith("import core"), line
