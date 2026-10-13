# 100% labels: all-block target adaptation

## One hypothesis, not a demonstrated improvement

The source-shared baseline updates 12 target adapters and only the final two
Transformer blocks. This experiment tests whether restricting updates to those
two blocks limits 100%-label target performance. It is NOT the proposed low-rank
few-label experiment, a new foundation pretraining run, or a new clinical result.

`configs/foundation/source_initialized_all_blocks.yaml` changes the target's
copied/trainable block count from 2 to 12. The patch embedding, CLS/position
parameters and final normalization still belong to the frozen encoder. Therefore
call this **all-Transformer-block adaptation**, not full-encoder fine-tuning.

The target starts with the same source-shared representation as the old baseline.
Source-shared adapters are copied, not tied. The copied blocks start from MAE
weights, as they did in the original two-block target. The original MAE, source
adapters, source-private adapters and source heads remain frozen; old tasks stay
available. Input remains four MRI modalities, with no clinical covariates or
cross-task ground-truth labels at inference.

## Fixed comparison conditions

- Original guided crop probability 0.5, mean pooling, no spatial adapter, no
  attention pooling, no MAE bypass and no source token retention.
- Same source checkpoints under `runs/source_initialized_learning`, same
  patient-disjoint protocol, label seed 2028 (+ fold), and 100% target budget.
- Target block LR 1e-5; adapter/head LR 1e-4; batch size 2, AMP, gradient
  checkpointing, weight decay 0.05, 200-epoch cap and unchanged early stopping.
- No source/target full-budget refit. Validation selects the checkpoint;
  training automatically loads that checkpoint and prints/saves test metrics.
- WHO remains grade II/III versus IV, not II versus III/IV.
- Apply the same architecture to all five folds in each of the three directions.
  Do not combine old and new per-fold best test scores into a five-fold average.

This changes capacity and optimization freedom, and may increase overfitting.
It cannot guarantee better C-index/ACC/AUC, or prove low-label transfer benefits.
Repeatedly inspected outer-test folds are development evidence, not untouched
confirmatory evidence. Do not select hyperparameters or epochs with test metrics.

## Disk use and compatibility

`adaptation.checkpoint_policy: best-only` saves the complete model state,
configuration, split, selected epoch and validation evidence in `best.pt`, but
omits optimizer/scaler state and does not save `last.pt` or epoch snapshots.
Automatic test and `scripts.test_source_guided` still work without an external
source checkpoint at inference. There is no exact training-resume support.
Metrics, training history, patient predictions and KM plots are still saved.
Legacy configs default to `checkpoint_policy: all` and retain their old saving
behavior. No historical run is deleted or overwritten.

Each best checkpoint still includes frozen source weights so that old tasks are
self-contained. Source training, if a source checkpoint is missing, keeps its
existing best/last policy but only saves a periodic snapshot at epoch 200.

## Server command

Upload the update package into the server project root first. The package
contains code/config/tests/docs only, not MRI caches or model weights.

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf source_initialized_all_blocks_update_20261002.tar.gz || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u -m scripts.run_all_blocks_transfer --device cuda
```

The launcher runs all 15 source-shared target experiments. It reuses existing
source best checkpoints and trains only absent source runs. A partial source
directory or an existing selected target directory is rejected rather than
overwritten. Child errors propagate; the next experiment is not silently started
after failure. Disk availability is checked before each training process.
The default output root is `runs/source_initialized_adaptation_all_blocks`.
Console logs are saved automatically. Do not add `--validation-only`.

For command inspection without starting training:

```bash
python -m scripts.run_all_blocks_transfer --dry-run
```

To run a precisely specified remaining subset, for example only Fold4 survival:

```bash
python -u -m scripts.run_all_blocks_transfer --fold 4 --target-task survival --device cuda
```

An interrupted target's best.pt is not interpreted as completed training.
Keep the interrupted run and explicitly choose a different `--output-dir` if
restarting it. Subsets are for execution recovery, not choosing favorable folds.

## Verification scope

`tests/test_all_blocks_transfer.py` covers gradients in all 12 copied blocks,
equal initial predictions versus the two-block baseline, immutable source
parameters/predictions, all three target losses with automatic test and test-only
reload, and identical numerical results between full and best-only saving.
Launcher tests exercise command planning, missing-source handling, fail-fast
behavior, disk checks and refusal to overwrite results. These are CPU functional
tests, not a 3090 Ti memory benchmark or evidence of improved clinical performance.
