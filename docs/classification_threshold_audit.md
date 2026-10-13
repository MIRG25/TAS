# Frozen classification-threshold diagnostic (2026-10-07)

## Scope and prespecified rule

Reuse all 20 already validation-selected checkpoints: voxel/SAM x IDH/WHO x
Fold0-4, source-shared, 100% target-label budget. This is an evaluation-only
diagnostic of decision thresholds, NOT new training or probability calibration.
No MRI weights, architecture, source heads, YAML settings or original results
are changed. Survival is excluded because a binary decision threshold cannot
improve its C-index. No fresh runs are launched.

For each selected checkpoint, use its original target-validation patients and
deterministic inference configuration. Maximize validation balanced accuracy
`(sensitivity + specificity)/2`, equivalent to Youden's J. Candidate thresholds
are distinct validation class-1 scores, 0.5, and the endpoint immediately below
the smallest validation score. Predict class 1 iff `p1 > threshold`; equality
is class 0, matching binary argmax at 0.5. Objective comparisons use exact
integer confusion-count numerators. Ties: closest to 0.5, then larger threshold.
The same rule is used for both methods/tasks, without looking at Test scores.

Class 1 means IDH mutant or WHO grade 4, respectively. WHO class 0 is grades
2/3, as defined in the current project, not grade 2 versus grades 3/4.

## Execution and safety

The runner preflights all requested checkpoints and patient-disjoint protocols
first. Voxel/SAM must have matching splits, source provenance and configurations
after removing only SAM settings and experiment name/output root. Missing or
duplicate runs fail explicitly, rather than silently selecting a favorable run.
Only trusted project best.pt files are loaded. No original MAE/source weight
files are needed because adapted best.pt includes the complete model state.

Each model then runs validation inference. Default metrics must reproduce the
checkpoint's original validation metrics within 1e-6, otherwise evaluation stops.
Missing labels or single-class validation cannot silently fall back to Test.
Threshold candidates, validation predictions and threshold_lock.json are saved
BEFORE test inference. Test scores cannot select thresholds. Test inference is
recomputed with frozen weights; default metrics must reproduce the original
test_metrics.json within 1e-6. This detects changed cache/checkpoint/code instead
of attributing replay differences to thresholds. Two reported AUCs must match.
Model tensors, checkpoint file and locked-threshold hashes are checked unchanged.

Upload `outputs/classification_threshold_audit_update_20261007.tar.gz` to the
SERVER project root first; the local archive is not automatically uploaded.
The small incremental package assumes the already-used SAM update is installed.

```bash
cd /root/autodl-tmp/project/glioma_meta_learning
test -f classification_threshold_audit_update_20261007.tar.gz && \
tar -xzf classification_threshold_audit_update_20261007.tar.gz && \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.audit_classification_thresholds --device cuda
```

This performs only forward passes on validation and Test, not 20 new training
runs. Each job prints both sets of test_metrics automatically. No new .pt files
are written. Original run directories and old 0.5 metrics remain untouched.

Optional metadata-only preflight: add `--dry-run`.
Explicit small subset: add `--fold 2 --task idh` (still compares both methods).
Path overrides: `--voxel-root`, `--sam-root`, `--folds`, `--protocol`, `--cache-dir`.
Only the cache location or worker count can be overridden, not preprocessing,
crop size, inference batch size, checkpoint epoch or threshold selection rule.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.audit_classification_thresholds --report-only
```

Report output: `runs/classification_threshold_audit/summary.csv` and terminal
tables with fold-wise thresholds, ACC/AUC/SEN/SPEC/F1/BACC, means/sample SDs and
same-checkpoint threshold-minus-default deltas. It also reports SAM-minus-voxel
deltas under each fixed decision rule, only for verified matched pairs.
Missing folds are not zeros and
partial means are not complete five-fold results. No per-fold winner selection.

Per-model directory: `runs/classification_threshold_audit/{voxel,sam}/{idh,who}_foldN/`

- audit_plan.json: rule, checkpoint hash and patient split.
- threshold_lock.json: validation-selected threshold and selection provenance.
- validation_threshold_candidates.csv: validation-only threshold search.
- validation_predictions.csv / test_predictions.csv: probabilities, labels,
  patient IDs, both decisions, selected epoch and checkpoint hash.
- threshold_metrics.json: confusion matrices, both validation/Test metric sets,
  replay and immutable-weight checks, file hashes, explicit diagnostic caveats.

Rerunning the same command skips verified completed jobs and prints their Test
metrics again. A partially written job is never overwritten: use a new
`--output-dir` with an explicit task/fold/method subset to recover it. No large
checkpoint copies are needed. `--report-only` reads audits and refreshes only
their summary.csv; it does not access MRI or evaluate new patients.

## Interpretation limits

The validation set was already used for checkpoint selection. Its tuned metrics
are optimization-set values, not an independent validation estimate. In small
validation cohorts (e.g. only 20 IDH-positive patients), threshold estimates can
be noisy; no Test gain is guaranteed. Increased sensitivity can trade off with
specificity or ACC. AUC is unchanged because patient scores are unchanged.
The already repeatedly inspected outer tests are exploratory/development data;
this diagnostic must not be presented as untouched confirmatory evidence.
No claim is made that it fixes survival ranking, all task performance, or the
scientific transfer advantage. Later independent/locked comparisons are needed.

Reference: https://scikit-learn.org/stable/modules/classification_threshold.html
