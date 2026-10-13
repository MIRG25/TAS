# Versioned survival endpoint correction — 2026-10-09

## Scope and evidence

This is a training-objective correctness correction, not a new architecture or
proof of improved discrimination. Preserve the seed-2026 voxel + spatial baseline.
Do not restart Q/K, SAM, covariance, compositional banks, or other retired variants.

The old loss uses the same `searchsorted(..., side="right")` interval for events
and censors and includes that entire interval for a censor. Under the project's
finite-endpoint prediction interpretation, a patient censored at day 200 therefore
supervises survival through day 365. We do not observe that outcome.

For hazards `[0.1, 0.8, 0.2]` and finite endpoints `[180, 365]`:

| Observation | Legacy NLL | Endpoint-v2 NLL |
|---|---:|---:|
| Censored at 100 | -log(0.9) | 0 |
| Censored at 200 | -log(0.9 × 0.2) | -log(0.9) |
| Censored at 365 | -log(0.9 × 0.2 × 0.8) | -log(0.9 × 0.2) |
| Event at 180 | -log(0.9 × 0.8) | -log(0.1) |
| Event at 200 | -log(0.9 × 0.8) | unchanged |

The local 494-case manifest contains 247 censored cases, all off the configured
cutpoints; one lies beyond the last finite endpoint. No observed event is exactly
on a cutpoint. These local counts do not establish a cause for any test-score gap.

## Exact convention

With finite endpoints c1 < ... < cK and K+1 hazard outputs, intervals are
`[0,c1], (c1,c2], ..., (cK,+infinity)`.

- An event at t uses the first endpoint >= t (`side="left"`), plus survival
  through the preceding intervals. Events after cK retain the existing open-tail
  event category; it is not a finite-time prediction endpoint.
- A censor at t contributes only finite intervals whose endpoints are <= t
  (`side="right"` gives their count). The open tail is never fully survived.
- A censor before c1 contributes zero NLL/gradient. It remains in the valid-patient
  batch mean, not silently dropped or renormalized. This sacrifices within-bin
  information: it is a discrete-grid/coarsened-data likelihood, not an exact
  continuous-time likelihood. A piecewise continuous model would be a separate
  experiment, not silently substituted here.
- Missing labels remain masked. Unmasked invalid times/events fail closed in v2.
- The logits, architecture, optimizer, crop policy, seed, label budget and
  validation selection metric do not change.

The event-up/censor-down convention is consistent with the author-maintained
[pycox discretization implementation](https://github.com/havakv/pycox/blob/master/pycox/preprocessing/discretization.py)
(`DiscretizeUnknownC`, inspected 2026-10-09). Our open-tail and before-first-endpoint
handling are explicit project adaptations; this is not a claim of identical grids.

## Compatibility and metrics

`model.survival_loss_version` accepts `legacy-v1` (default if absent) or
`endpoint-v2`. Old configs and checkpoints remain legacy; there is no global
default/YAML change. New source/target checkpoints record the version. Test-only
loading rejects a config/marker mismatch. Changing a checkpoint config does not
make old weights newly trained.

The evaluation module is unchanged: same C-index definition/risk horizon, iAUC,
IBS, KM/log-rank, and interpolation/endpoint lookup. For identical predictions,
old and new loss versions produce identical metrics. In particular, historical
off-grid lookups still use the next finite endpoint; we do not change that metric
approximation in this control. Results must not be improved by changing scoring.

Only IDH+WHO -> survival can reuse the existing original source checkpoint for
this first control, because its source training did not use survival labels/loss.
For directions with survival among sources, a different configured loss version
is rejected instead of claiming the source was corrected. Correcting those source
stages would require separately authorized/reported source retraining.

## One bounded server experiment

Upload the provided archive to the project `outputs/` folder. In project root:

```bash
tar -xzf outputs/survival_endpoint_update_20261009.tar.gz &&
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u -m scripts.run_survival_endpoint_control --device cuda
```

The launcher reads the completed original voxel Fold0 survival `best.pt` embedded
configuration, not mutable YAML inheritance. It validates protocol, fit/validation/
test patients, original source tensors, cached files and baseline sidecars before
training. Only the loss version and output/experiment identifiers change. It does
not select settings from test results, change source weights, run fresh, or refit.
It refuses to overwrite any target run. `--dry-run` is read-only and loads no MRI.

Outputs: `runs/source_initialized_survival_endpoint_v2/`, including best.pt,
effective config, patient split, history, test_metrics.json, predictions.csv, and
hashes of baseline/source/protocol artifacts. The training entry automatically
evaluates the validation-selected model on Test and prints all survival metrics.
Only the selected checkpoint is saved under the matched best-only policy.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.run_survival_endpoint_control --report-only
```

No new inference in report-only mode. Do not treat one-fold differences as
five-fold evidence or a guarantee of C-index >= 0.70. Prior repeated inspection of
these test sets makes this an exploratory control, not untouched confirmation.

## Verification

CPU tests cover hand-derived NLLs and gradients at/in/before/after intervals,
invalid data, missing labels, legacy serialization, unchanged metric mapping,
and real tiny-cache source -> target -> automatic Test -> test-only replay.
The runner's actual child process is exercised; no clinical GPU training is
performed locally. Test outcomes are recorded in the supplied verification log.

Verified locally: full suite **579 passed, 3 skipped, 2 subtests passed**;
11,123 library/deprecation warnings (not a warning-free run). The initial full
run exposed nine legacy-source compatibility failures; these were fixed rather
than skipped. The final preflight hardening was additionally checked with the
three endpoint-control tests, all passing. Independent read-only review found no
blocking issue. CPU/tiny-cache correctness does not validate clinical performance
or guarantee numerical identity across servers.
