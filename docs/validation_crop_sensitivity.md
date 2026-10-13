# Frozen validation crop-sensitivity audit (2026-10-08)

This is **not another training candidate**. It investigates the completed
source-shared + voxel tokenizer + spatial adapter baseline, all three targets
and five folds. Covariance is not stacked on it; fresh is not evaluated.

## Fixed protocol

- Load each existing validation-selected `best.pt` and its embedded source and
  target weights. No source checkpoint/MAE file is needed for reconstruction.
- Preserve exactly the saved target-validation IDs, target label budget,
  preprocessing, input size and metrics. Read test IDs only for exclusion;
  never evaluate test labels, load test MRI or read saved test predictions.
- Seven prespecified same-size crops: original deterministic crop, then starts
  shifted by -16/+16 voxels separately along cached D, H and W. These are **not**
  anatomical left/right directions. At the 1-mm cache spacing this is 16 mm.
  Shift the already-clipped production start, clip again at cache boundaries.
  No random draws, resizing, padding, augmentation or retention-based retry.
- First reproduce the saved center-validation AUC/ACC/F1/SEN/SPEC (classification)
  or C-index (survival), absolute tolerance 1e-5. If this fails, stop that model
  before shifted inference, save `reproduction_failed.json`, and investigate
  cache/checkpoint/code/runtime consistency. Do not relax it based on Test.
- Report every view's metrics. No optimal crop, threshold, epoch, method,
  averaged prediction or ensemble is selected. Survival IPCW references use
  **target-fit labels only**, not fit MRI, validation or test censoring estimates.
- Record requested/actual positions, boundary clipping, tumor retention and
  foreground fraction. Empty tumor masks have undefined retention (CSV blank).
  Deduplicate actual crop starts for per-patient dispersion and flip summaries.
- Forward FP32, original batch size, eval mode, no gradients/optimizer. Verify
  every model state tensor and the input checkpoint SHA256 after inference.

## Server instructions

Upload the incremental archive into the project root and extract there:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning
tar -xzf validation_crop_audit_update_20261008.tar.gz
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u -m scripts.audit_crop_sensitivity --device cuda
```

Defaults: input `runs/source_initialized_adaptation_voxel_tokenizer`, output
`runs/voxel_validation_crop_audit`, all 15 existing models. There are **zero
training epochs and zero outer tests**. Each model needs seven validation
passes; progress prints during each pass. It produces small CSV/JSON files
and validation KM plots, no new `.pt` or image caches.

Optional read-only preflight (no inference/output writes):

```bash
python -m scripts.audit_crop_sensitivity --dry-run
```

If checkpoint identity validation fails, inspect **all** checkpoint metadata
once, without needing MRI, a GPU or even the local protocol files:

```bash
python -m scripts.audit_crop_sensitivity --inspect-checkpoints
```

The compatibility/diagnostic fix (v2, 2026-10-08) prints the exact checkpoint,
task/fold/epoch and every mismatched field with actual/expected values. The
normal runner also names the offending file. It excludes explicitly consistent
matched-fresh controls in the same root; unknown or inconsistent source-shared
checkpoints still fail. It does **not** guess missing seeds, rewrite config,
trust directory names instead of saved metadata, or relax model/split checks.
The original generic traceback alone does not establish which field failed.

If paths changed on another server, use `--root`, `--cache-dir`, `--folds`,
`--protocol`, `--output-root`. Config is taken from the checkpoint, not a newly
edited YAML. A subset can use `--fold 3 4 --target-task survival idh`; this does
not select different crop policies. CPU is allowed explicitly with `--device cpu`.

```bash
python -m scripts.report_crop_sensitivity
```

The report runs automatically at completion and can also read partial progress.
Completed audits are skipped only if the checkpoint/request and saved output
hashes match. The request also records the manifest SHA256 and validation-cache
file sizes/mtimes (cheap change detection, not full MRI content checksums).
An incomplete directory is not overwritten; choose a new output
root or explicitly preserve/move that incomplete audit before restarting.

## Outputs and interpretation

Under `foldN_TASK/`:

- `crop_sensitivity.json`: protocol, checkpoint hash, verification, metrics of
  all seven crops, score dispersion, rank correlations, limitation statements.
- `validation_metrics_by_crop.csv`: all metrics in a fixed view order.
- `validation_predictions_all_crops.csv`: predictions and crop geometry for
  every validation patient/view.
- `patient_sensitivity.csv`: unique-crop count, score range/SD, max deviation
  from center, retention, and classification flip flag.

Classification uses positive-class probability; survival uses the project's
fixed-horizon risk. The within/between SD ratio normalizes mean within-patient
score SD by the between-patient center-score SD. It is not a statistical test
or a calibrated acceptance threshold. Spearman correlations measure the rank
stability of the same cohort between center and shifted views; constant scores
give an undefined correlation, not a fabricated zero.

Changes associated with a lower tumor-retention fraction may reflect missing
image content. Changes when retention remains similar may reflect positional
sensitivity or different context. Neither proves the causal source of a fold
gap. Stable predictions would weaken this hypothesis; do **not** automatically
launch a multi-crop training sweep. This validation set already selected the
epoch, so these are exploratory diagnostics, not independent validation.

No `test_metrics` are fabricated or reprinted: `outer_test_evaluated=false`.
The existing outer-test results remain untouched.
