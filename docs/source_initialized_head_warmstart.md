# 100% labels: head warm start, then two-block target adaptation

## Scope and hypothesis

This experiment returns to the original source-shared architecture with mean
pooling, twelve copied target adapters and two copied Transformer blocks. It
tests one training change: fit the randomly initialized target head before
allowing it to update the inherited representation. It does not add attention
pooling, spatial adapters, token retention, MAE bypass or source residual gates.
It is a testable optimization hypothesis, not a guaranteed performance gain or
a claim that a five-epoch head has converged.

The original source adapters, source-private branches, source heads and MAE
encoder remain frozen and unchanged. Existing tasks remain available. Only T1,
T2, T1c and FLAIR enter the model; there are no clinical inputs or other-task
ground-truth labels at inference.

## Exact two-stage schedule

1. **Head-only warm start: five fixed epochs.** Only the new target head is
   trainable. The inherited representation runs in evaluation mode; the target
   fit patients retain the existing random MRI crop/augmentation pipeline.
   Head LR is constant at 1e-4. This phase never uses validation or test labels
   for training, checkpoint selection, stopping, or duration selection.
2. **Target fine-tuning: up to 200 additional epochs.** Keep the fitted head;
   unfreeze the target adapters and two copied target blocks. Recreate the
   optimizer and AMP scaler, retaining all fitted model weights. Adapter/head
   LR is 1e-4, block LR is 1e-5, with the existing five-epoch cosine LR warmup
   and weight decay 0.05. Validate each epoch and use the original early
   stopping rule (patience 30, minimum 30 fine-tuning epochs).

`head_warmstart_epochs: 5` is not `warmup_epochs: 5`: the latter is the learning
rate ramp **inside stage 2**, while the former is a preceding head-only phase.
`selected_epoch` remains the stage-2 epoch, so selected epoch 10 means 5 + 10
epochs of target training. Checkpoints and final results carry
`adaptation_schedule` with `head_warmstart_epochs`, `finetune_max_epochs`,
`selected_finetune_epoch`, `total_epochs_at_selection` and optimizer-reset
provenance. Head-stage history is saved separately in JSON/CSV.

Head-stage random number use is isolated, including the training-loader random
generator. Stage 2 starts with the baseline sampling/random-state stream, not
one advanced by five extra passes. The fitted head is retained; the source model
is never updated. Configurations without this option default to zero head-only
epochs and retain their original training path.

## Fixed comparison conditions

- Reuse the original `runs/source_initialized_learning` best checkpoints.
- Original crop probability 0.5, mean pooling, batch size 2, AMP and encoder
  gradient checkpointing. No source token-retention term.
- Same patient-disjoint protocol and 100% target label budget, including the
  inner validation subset. Only target fit patients contribute gradients.
- Label seed 2028 + outer fold. No source or target full-budget refit.
- Classification checkpoint selection uses validation AUC; survival uses
  validation C-index. Test metrics are not selection criteria.
- WHO remains grade II/III versus IV, not II versus III/IV.
- Use this same schedule for all five folds of all three transfer directions;
  do not select a different architecture for each favorable test fold.

Repeatedly examined outer-test folds are exploratory development evidence,
not an untouched final confirmation set. Completing this experiment cannot by
itself establish low-label transfer benefits; those require matched low-label
comparisons after fixing the method.

## Saving and disk safety

The target saves only the validation-selected `best.pt`, without optimizer,
scaler, `last.pt` or periodic epoch copies. It remains a self-contained model for
automatic test and independent `scripts.test_source_guided` evaluation. This
policy does **not** support exact training resumption. Metrics, patient-level
predictions, histories and KM plots are retained. Historical runs are neither
deleted nor overwritten.

Existing source checkpoints are reused. Missing source runs are trained using
the unchanged source objective; source training keeps its existing best/last
policy, with periodic snapshots only at epoch 200. The launcher rejects partial
source directories and existing selected target directories. Child errors stop
the grid immediately. At least 2 GiB free is checked before every child process;
this is a guard, not a guarantee that the entire grid fits on the disk.

## Run on the server

Upload the update package to the server project root first. It contains code,
configuration, tests and documentation only, not images or model weights.

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf source_initialized_head_warmstart_update_20261003.tar.gz || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u -m scripts.run_head_warmstart_transfer --device cuda
```

This runs 15 **source-shared** adaptations: IDH + WHO to survival, IDH + survival
to WHO, and WHO + survival to IDH, on folds 0–4. Each automatically loads its
validation-selected model, tests once, prints `test_metrics` and saves metrics
and patient predictions. Do not add `--validation-only`.

Output root: `runs/source_initialized_adaptation_head_warmstart`. Console logs
are saved automatically in its `logs` directory. Check the complete planned
commands without training or writing files with:

```bash
python -m scripts.run_head_warmstart_transfer --dry-run
```

For an explicitly specified remaining subset, for example Fold4 survival:

```bash
python -u -m scripts.run_head_warmstart_transfer --fold 4 --target-task survival --device cuda
```

An interrupted target directory is not treated as completed. Keep it and use
an explicit new `--output-dir` when restarting that subset. Do not use subsets
to assemble favorable per-fold results. The launcher also permits an explicitly
requested `--target-initialization fresh` matched control in a **new** output
directory; that is not the default experiment.

## Verification scope

CPU tests cover all three target losses, head-only freezing, retained head
weights, stage transition, random-state isolation, old-path compatibility,
automatic testing and standalone checkpoint reload. Launcher tests exercise the
15-job mapping, missing-source reuse, profile validation and refusal to overwrite
old results; shared launcher tests cover disk/error propagation. These are
functional checks, not evidence of clinical improvement or a GPU memory benchmark.
