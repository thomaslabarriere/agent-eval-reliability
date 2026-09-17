# agent-eval-reliability

A small, dependency-free reliability layer for LLM-agent evals: is a pass-rate
difference between two agent versions real, and does your LLM judge actually
agree with a human? It is built to sit on top of Windmill's agent-evals, and
ships as two Windmill-ready runnables plus a pure, tested core.

## Why it exists

Windmill already has a solid agent-eval system (datasets, cases, script/agent
scorers, pass rate and a delta between versions, all run as a generated flow).
Reading that code, two gaps stand out, and they are the two that decide whether
a number you look at is trustworthy:

1. **A pass-rate delta has no error bar.** An eval run executes each case once
   (`backend/windmill-api/src/ai_evals/run.rs` builds a `forloopflow` over the
   cases), and the UI shows the delta between two versions as a raw subtraction
   (`frontend/src/lib/components/aiEvals/EvalsPane.svelte`). A non-deterministic
   agent scored once per case can show a +7% "improvement" that is pure noise.
2. **The judge is never calibrated.** An Agent scorer (`ScorerDef::Agent` in
   `ai_evals/scorers.rs`) lets an LLM decide pass/fail with a `pass_if`
   threshold, but nothing measures whether that judge agrees with a human, or
   what threshold would make it agree best.

This repo fills both, without touching their Svelte UI.

## What it does

### Brick A: is the delta real? (`windmill/delta_reliability.py`)

Runs each case several times, then reports a Wilson interval on each version's
pass rate, a paired bootstrap CI and a paired permutation p-value on the delta,
and which cases are flaky. Example on the bundled fixture:

```
v1 vs v2  (5 runs/case)
  v1: 56.7%  v2: 63.3%
  delta +6.7%  95% CI [-6.7%, +20.0%]  p=0.689  -> not distinguishable from noise
  flaky cases: [duplicate_order, escalate_fraud, partial_refund, ...]
```

The +6.7% would read as a green win in the current UI. It is noise.

### Brick B: does the judge match humans? (`windmill/judge_calibration.py`)

Given human gold labels and the judge's scores, reports agreement,
precision/recall, the confusion matrix, and Cohen's kappa, and recommends the
`pass_if` threshold that best matches the humans. Example on the bundled fixture:

```
Judge calibration over 24 labelled cases
  current     (pass_if=0.50): agreement 83%, kappa 0.67, precision 73%, recall 100%, fp 4 fn 0
  recommended (pass_if=0.57): agreement 96%, kappa 0.92, precision 92%, recall 100%, fp 1 fn 0
```

The recommended `pass_if` drops from 4 false positives to 1, and maps straight
onto Windmill's `ScorerDef` `pass_if` field.

## Quickstart (offline, no API key)

```bash
python -m windmill.delta_reliability     # Brick A on examples/runs_v1_vs_v2.json
python -m windmill.judge_calibration     # Brick B on examples/judge_labeled.json
python -m pytest -q                      # the tests
```

Both scripts read recorded outcomes, so anyone can reproduce the exact verdict.
Wiring them to live Windmill runs means feeding real per-run outcomes / judge
scores into the same `main`.

## How it maps to Windmill

- Each script is a plain `def main(...)` returning structured output, so it drops
  into a workspace as a Python script (auto-generated UI from the signature).
- Brick B's output is a `pass_if` value, directly usable in an Agent scorer.
- Natural next step inside their product: surface Brick A's interval where the
  delta is rendered (`EvalsPane.svelte`), so a delta ships with its error bar.

## Layout

```
core/          pure, tested statistics (no I/O, no LLM)
  wilson.py        Wilson score interval
  resampling.py    paired bootstrap CI + paired permutation test
  reliability.py   run outcomes -> full report + verdict
  calibration.py   confusion, precision/recall, kappa, threshold sweep
windmill/      the two runnables (def main), portable into a workspace
examples/      recorded fixtures so the demos run offline
tests/         mutation-proof tests (revert a fix and a test goes red)
```

## Scope

This is an analysis layer, deliberately small. It does not run agents itself and
does not modify Windmill's UI. The offline fixtures are illustrative, not a
benchmark of any real agent. The value is the method: a verdict computed from
recorded facts, reproducible and auditable, rather than a raw number taken on
trust.

## License

MIT.
