# Target-fit classification weighting ablation (2026-10-08)

**Status: PAUSED by the user's subsequent request. Do not launch this experiment
or deploy its prepared archive.** The active reference is voxel tokenizer +
spatial adapter with training seed 2026 and legacy batch-weight-mean CE.
The optional implementation is retained for provenance, not enabled by default.

The two-seed audit shows both repeated mistakes and training variability, not
a unique cause or proof of source bias. This experiment tests a specific loss
semantics issue, not a new architecture or a claimed cure for survival errors.

## One requested training change

Legacy PyTorch CE: `sum(w[y] * CE) / sum(w[y])` within each minibatch.
For a homogeneous-class batch, the class weight cancels; this is correct
PyTorch behavior, not a library defect. It does not equal the expectation of
a fixed per-patient weighted objective for very small random batches.

New target option: `adaptation.classification_loss_reduction: fit-patient-mean`.
Let fit class frequencies be `p[c]` and the EXISTING configured weights be
`w[c] = (1 / (K * p[c])) ** class_weight_power`.
Normalize once: `w_fixed[c] = w[c] / sum(p[c] * w[c])`.
Minibatch loss: `mean(w_fixed[y_i] * CE_i)` over valid labelled patients.
Hence `E_fit[w_fixed[y]] = 1`; homogeneous minority batches retain their
relative weight. No minibatch frequency estimation, validation/Test-derived
weights, new power, threshold change, resampling, or additional labels.
Mask-false/missing labels are excluded, never converted to negative labels.

Default `batch-weight-mean` takes exactly the previous loss path. Survival
NLL and source training are unchanged. New normalization is currently guarded
against combination with SAM, head warmstart, and full-budget refitting.
The existing IDH/WHO heads/selection metrics/argmax decisions are unchanged.

This changes the objective and gradient distribution. Improved sensitivity,
ACC, AUC, or stability is NOT guaranteed; these outcomes can trade off.
The ablation is not proposed as a paper's algorithmic novelty.

## Matched execution

The runner reads each original completed voxel run's `best.pt`,
`effective_config.json`, `data_split.json`, and `test_metrics.json` from
`runs/source_initialized_adaptation_voxel_tokenizer`. It never reconstructs
the baseline from a possibly edited current YAML. It requires training seed
2026 and holds that seed fixed; this is NOT another seed repeat.

Preflight checks all selected jobs before any GPU training: config differs
only in the reduction field; same source tensors embedded in baseline best.pt;
same protocol, patient IDs/order and label budget; unchanged source task/held-out
target identity; all required caches exist. Child training repeats the
config/source/patient check after reconstructing its actual split.

Source checkpoints are reused. Only IDH and WHO target adaptation runs, all
five folds, with 100% budget. Survival target runs would be unchanged and are
therefore not repeated. This does not establish improvement across all three
tasks or in the low-label regime.

Outputs: `runs/source_initialized_adaptation_fit_patient_mean` (separate root).
Each target saves best-only checkpoint, fit-derived weight counts/scales,
effective config, split, training history, final outer-Test metrics and patient
predictions. Normalized weights/reference and baseline/source hashes are in
the checkpoint/result/split provenance. Validation selects epoch; final Test
does not select checkpoint, thresholds, or per-fold methods.

Run from the SERVER project root after uploading the archive into `outputs/`:

```bash
tar -xzf outputs/classification_weighting_update_20261008.tar.gz &&
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.run_classification_weighting --device cuda
```

Optional preflight only: append `--dry-run` (no writes/training/inference).
Runner prints/logs each target's `test_metrics` and automatically reports at
the end. While running or after completion:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.report_classification_weighting
```

The same run command skips only fully completed runs with matching inputs and
artifact hashes. Partial runs are not overwritten or silently resumed; use a
new output root for a deliberately restarted selection. Minimum 2 GiB free is
checked before each job. No runs, predictions, or existing checkpoints deleted.
The standalone report also verifies completion hashes and the original baseline
result hash. A final JSON without its completion marker is reported as pending,
not silently included. Altered metrics/predictions/checkpoints fail closed.

## Interpretation

Use paired IDH/WHO results across all five folds. Record AUC, ACC, sensitivity,
specificity and F1 together; do not call sensitivity-only gains an overall win.
Do not pick methods per fold, change thresholds against Test, average missing
folds as zeros, or claim significance from fold SD. Existing repeatedly viewed
Test folds remain exploratory. Cross-server/runtime changes can confound even
a matched configuration and seed. No new request for patient logs is needed.
