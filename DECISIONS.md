# Design decisions

Short record of the non-obvious choices, so they can be argued in an interview.

## Wilson interval, not the normal approximation
The normal approximation (`phat +/- z*sqrt(phat*(1-phat)/n)`) collapses to zero
width at `phat = 0` or `1` and can leave `[0, 1]`. Eval datasets are small and
pass rates often near the extremes, exactly where it fails. Wilson stays inside
`[0, 1]`, is asymmetric toward 0.5, and behaves for small `n`. See
`core/wilson.py`; a test asserts the interval is not the symmetric normal one.

## Paired permutation test, not McNemar or two-proportion z
Both agent versions are scored on the *same* cases, so the comparison is paired.
McNemar is the classic paired test but assumes one observation per unit; here
each case is run N times, giving a per-case pass rate, not a single bit. A
two-proportion z-test would ignore the pairing and treat all runs as
independent, understating uncertainty. A sign-flip permutation test over the
per-case deltas is the exact null for "the two versions are interchangeable on
this case" and needs no distributional assumption. See `core/resampling.py`.

## Bootstrap over cases for the delta CI
The uncertainty we care about is "would this delta hold on other cases like
these", so we resample the cases (with the A/B pair kept together), not the runs
within a case. Percentile interval, seeded for reproducibility.

## p-value is never zero
The observed assignment is counted in the permutation null (`(hits+1)/(iters+1)`),
so a p-value is a proper bound, never a misleading exact 0.

## Cohen's kappa alongside agreement
Raw agreement is inflated when labels are imbalanced: a judge that always says
"pass" scores high agreement on a mostly-passing dataset while adding nothing.
Kappa corrects for chance agreement and exposes that. A test locks this in.

## Threshold ties break stricter
When several `pass_if` thresholds score equally, the higher (stricter) one is
chosen: for a gate, a false pass is usually worse than a false fail, so we prefer
the more conservative boundary. Callers can optimize agreement, F1, or kappa.

## The recommended threshold is fit in-sample, so it is cross-validated
`recommend_threshold` picks `pass_if` on the same labels it then scores, so its
in-sample number is optimistic and will look better here than on the next batch.
Rather than pretend that away, `cross_validated_agreement` refits the threshold
on k-1 folds and measures agreement on the held-out fold, and the judge script
reports that held-out number alongside the in-sample one. On a small label set,
treat the recommendation as a suggestion and trust the cross-validated figure.

## The overall pass-rate interval is a display approximation
The per-version overall Wilson interval pools every run as if independent, but
repeated runs of the same case are correlated, so that interval is narrower than
the true uncertainty. It is kept as a display summary; the delta CI and p-value,
which are what a version comparison actually turns on, use the paired case-level
resampling that avoids this pooling. Stated here so it is a choice, not an
oversight.

## Within-case variance is collapsed to a per-case rate
Both resampling tools reduce each case's N runs to one pass rate, so they capture
between-case variability, not the noise inside a case's own rate estimate. This
is the right question for "would this delta hold on other cases like these", but
when N per case is small the delta CI and p-value are mildly optimistic, since
per-case rates are themselves noisy estimates treated as fixed.

## Offline-first
Everything runs from recorded outcomes with no network and no API key, so the
verdicts are reproducible and reviewable. Live use feeds real outcomes into the
same functions; nothing in the core changes.

## Pure standard library
The core has no third-party dependency, so a script pastes into a Windmill Python
runnable and runs with no dependency resolution. `pytest` is a dev-only dependency
for the tests.
