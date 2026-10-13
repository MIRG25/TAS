# Target-only SAM on the current voxel/spatial transfer model

Status 2026-10-07: all 15 experiments have been reported. Results were mixed;
SAM is not an overall replacement for voxel. Do not rerun the training commands
below as the next experiment. Current next step is the evaluation-only
[classification threshold audit](classification_threshold_audit.md).

This is one new, prespecified optimization candidate. It is **not an established
performance improvement** and is not a new meta-learning algorithm. Original
SAM: https://arxiv.org/abs/2010.01412 . We do not claim invention of SAM.

## Fixed comparison

Reference: `source_initialized_voxel_tokenizer_v1`, the completed 15 experiments.
Candidate: `source_initialized_voxel_sam_v1`.

- Same original source checkpoints under `runs/source_initialized_learning`.
- Source-shared initialization, voxel residual tokenizer, spatial adapters,
  mean pooling, two target-private blocks, identical architecture and losses.
- Same patient-disjoint protocol, 100% target-label budget (including validation),
  seed rule, class weighting, MRI crops, learning rates, early stopping and metrics.
- SAM rho=0.05 across all three targets and all five folds; no test-based tuning.
- No EMA, warmstart, patient contrast, SKB, retention, MAE bypass or source refit.
- No fresh experiments in this round. Do not replace these weights with a
  previous patient-contrast or token-retention source checkpoint.

Only `adaptation.sam` and the experiment name/output root differ from the
voxel baseline. The old YAML/default single-pass path remains available.

## Update semantics and safety

Every fit batch is loaded/cropped once. SAM computes the unclipped clean task
gradient, temporarily adds its L2-normalized perturbation to **trainable target
parameters only**, then recomputes the task gradient on the same batch. Torch
randomness is replayed for dropout, without consuming an additional training
random stream. The current model uses GN/LN, not stateful BatchNorm.

Original parameters are copied back exactly before a single AdamW descent
step, including on exceptions or AMP overflow. Weight decay and gradient
clipping apply at descent only. GradScaler.unscale_ is called once per optimizer
update; nonfinite scaled gradients cause an explicit skipped update and scale
backoff. Two passes mean increased training cost, not increased inference cost.

The source model, source task heads and MAE are never optimized. The existing
source-state equality guard remains active. No extra target/source labels are
provided to the model, and neither SAM pass uses validation/test patients.

All checkpoints are saved **after restoring the normal weights**. By default
only `best.pt` is retained, without optimizer/scaler state; this is not a
resumable-training checkpoint. The selected model is tested automatically.
`scripts.test_source_guided` can reload it independently, without SAM or a
source checkpoint file, and verifies the recorded SAM metadata.

## Server commands

Upload `outputs/source_voxel_sam_update_20261006.tar.gz` into the **server project
root** first. The local archive is not automatically present on the server.
The archive contains project code/config/tests/docs, not datasets or weights.

```bash
cd /root/autodl-tmp/project/glioma_meta_learning
test -f source_voxel_sam_update_20261006.tar.gz && \
tar -xzf source_voxel_sam_update_20261006.tar.gz && \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.run_sam_transfer --device cuda
```

This audits all 15 original sources first, then runs survival/WHO/IDH for
Fold0–4 sequentially. Every target automatically prints/saves `test_metrics`.
Training stdout/stderr is already saved in the new root's `logs` directory.
No extra `tee`, MAE training, source training or fresh run is needed.

Output root: `runs/source_initialized_adaptation_voxel_sam`.
Run suffix: `_spatial3d_voxel_sam`. Existing directories are never overwritten.
Use `--fold 3 4 --target-task survival` only to select explicitly unstarted
jobs after an interruption; incomplete jobs are not automatically resumed.

Read-only preflight, without training/test evaluation:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.run_sam_transfer --device cuda --dry-run
```

Partial/final summary, no training or new testing:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.report_sam_results
```

The report separates tasks and folds, prints mean/sample SD/min/max/range,
does not fill missing results with zeros and never averages log-rank p-values.
It also prints SAM-minus-voxel deltas **only** when protocol, source provenance,
actual fit/validation/test IDs and effective settings match. Lower IBS is better.
If the old root was relocated, pass `--baseline-root PATH`.

Inspect per-epoch `SAM ... updates=... skipped_clean=... skipped_perturbed=...`.
Checkpoint/prediction/result metadata contains version, rho and target-only scope.
The dedicated report accepts the prespecified rho=0.05 only; do not pool other
rho values into this experiment. Generic `adapt_source_guided` accepts a boolean
`adaptation.sam.enabled` switch, but the SAM runner deliberately rejects disabled
SAM or fresh initialization to avoid mislabeled experiments.

## Interpretation

Evaluate mean performance **together with** dispersion and fold-wise changes:
SD falling only because strong folds deteriorate is not success. Include AUC,
ACC and sensitivity/specificity for classification, and C-index/iAUC/IBS for
survival. Do not splice different methods' best test folds into one result.
Single-seed five-fold results do not by themselves establish seed robustness.
These repeatedly inspected outer folds are exploratory development results,
not an untouched confirmation cohort. Future low-label/fresh comparisons and
independent validation are still necessary for the research claims.
