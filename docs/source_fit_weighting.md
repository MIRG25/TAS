# Source-fit classification weighting control

This is an opt-in source-only experiment on the **new WHO endpoint**
(I/II=0, III/IV=1). It does not restart the paused target-only weighting ablation.
The old default and saved results remain unchanged.

## Hypothesis, not an established explanation

Source weighted CE currently divides by the sum of weights in each minibatch.
With small homogeneous batches this cancels class weighting. This is valid
PyTorch behavior, but differs from a fixed per-patient weighted objective.
The new mode tests that difference; it does not establish that this caused the
observed performance drop or guarantee improved stability/transfer.

Use the existing source-fit class weights and class-weight powers without tuning.
The current effective powers are IDH=1.0 and WHO=0.5; they are not increased.
For source-fit class frequencies p and raw weights w, normalize once:

    fixed_weight[c] = w[c] / sum_c(p[c] * w[c])
    source_classification_loss = mean_i(fixed_weight[y_i] * CE_i)

Thus E_source-fit[fixed_weight]=1. Counts exclude missing/masked labels and come
only from source fit, never validation/test. Uniform shuffled sampling, PCGrad,
gradient clipping and optimizer remain unchanged. These downstream operations
can still affect gradient magnitudes; this is not a guarantee of equal updates
or reduced stochastic variance. Source full-budget refit is disallowed in this
mode to avoid silently changing the weight reference.

Only source IDH/WHO classification loss changes. Source survival loss, target
losses, architecture, source checkpoint selection, target selection, patients,
seed2026, and the label-seed schedule2028+fold stay unchanged. Source heads remain
preserved during target adaptation. No fresh runs or new network modules.

## Run after uploading the update to the project root

Do not replace shared Python files while an old training job is running.

```bash
tar -xzf outputs/source_fit_weighting_update_20261010.tar.gz
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m scripts.run_cross_depth_transfer --who-low-high --source-fit-weighting --fold-indices 0 1 2 3 4 --device cuda --dry-run
python -u -m scripts.run_cross_depth_transfer --who-low-high --source-fit-weighting --fold-indices 0 1 2 3 4 --device cuda
python -m scripts.run_cross_depth_transfer --who-low-high --source-fit-weighting --report-only
```

All three source/held-out-task directions are included. Default without an
explicit fold list remains the existing Fold0–1 pilot. A new source model is
trained for each direction/fold, followed by target adaptation and automatic
outer-Test evaluation of its validation-selected checkpoint. Completed jobs are
verified and skipped; incomplete directories are not overwritten or silently resumed.
Fixed-mode source checkpoints fingerprint the manifest at training start; reuse
is rejected if the manifest later changes, even if patients and class counts
remain identical. This digest is integrity provenance, not a training input.

New roots:

- `runs/source_learning_fit_weighting`: source checkpoints and completion markers.
- `runs/source_cross_depth_fit_weighting`: target checkpoints, Test metrics and logs.

The source config snapshot lives under the target root's `_inputs` directory;
no extra YAML or alteration of legacy defaults is needed. Checkpoints and final
target results include `source_classification` counts, normalized weights and
`weight_reference=source_fit_only`. Target config still records
`classification_loss_reduction=batch-weight-mean`.
The report also prints the selected source validation metrics, separately from
target Test metrics, to make source improvement distinguishable from transfer gain.

## Interpretation

Compare against the **same new-WHO** cross-depth experiment, not the former
II/III-vs-IV endpoint. Source validation AUC, class-specific recall and downstream
Test metrics must all be examined. Higher source validation alone is not proof
of better transfer. Repeatedly inspected Test results are exploratory; do not
select per-fold methods, seeds or epochs from Test. No missing-fold imputation,
no averaged log-rank p values. A single seed cannot establish statistical stability.
