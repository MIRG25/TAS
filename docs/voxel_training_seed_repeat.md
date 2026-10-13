# Fixed-split target training-seed repeat (2026-10-08)

## Decision after the completed crop audit

All 15 frozen baseline models reproduced their original center-validation
metrics. Survival folds 3/4 have minimum center-vs-shift rank correlations
0.9796/0.9800, higher than folds 0/1 (0.8636/0.7343). IDH fold 1/3 validation
AUC remains 0.9102-0.9508 / 0.9553-0.9886 across views, despite their original
Test AUC 0.8010/0.8144. Crop sensitivity exists but is not established as the
dominant cause of these gaps. Do not repeat this audit or launch multi-crop
training automatically. Keep the voxel + spatial reference; the mean survival
Test C-index remains 0.6779, not the selected-validation values.

The next controlled experiment asks whether weak-fold behavior changes under
target-training randomness with patients and sources fixed. It is NOT an
architectural improvement, a seed search, or an inference ensemble. Two seeds
cannot establish statistical stability, a unique cause, or a guaranteed gain.

## What changes / what does not

- Reuse the completed baseline training seed **2026**; train one additional
  prespecified seed **2027** for all three targets and folds 0-4. No fresh arm,
  source training, or MAE training. The seed is exact, not seed + fold.
- `--training-seed` controls new target head/stem initialization, crop/augmentation,
  training sample order and model training randomness. Source-shared weights
  remain copied from the exact same original source checkpoint.
- `--label-seed 2028` is unchanged (saved effective label seed 2028 + fold).
  Target budget, fit, validation, test IDs and original protocol do not change.
- Use **each baseline's saved effective_config.json**, not the latest YAML.
  The launcher and child both reject configuration differences beyond
  `training.seed`; compare source tensors against those embedded in old best.pt.
- Same model, losses, learning rates, epoch budget, early stopping, validation
  rule and single center-crop evaluation. No SAM/covariance/retention/warmstart.
- New path suffix `_trainseed2027`; old paths and default CLI behavior remain.
  Save only best.pt (no optimizer/last/periodic files) plus normal reports and
  predictions. Checkpoint, results, split and predictions record training seed;
  checkpoints/results record pairing provenance and current runtime versions.
- Legacy baseline files lack runtime information; matching config and tensors
  cannot prove the old server, libraries, or MRI cache content were identical.
  Use the same environment/caches. If they changed, disclose this confounder;
  do not attribute all differences exclusively to the training seed.

## Server execution

Upload `outputs/voxel_training_seed_repeat_update_20261008_v2.tar.gz` from the local
project into the server project's **outputs/** directory, then:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning &&
tar -xzf outputs/voxel_training_seed_repeat_update_20261008_v2.tar.gz &&
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.run_voxel_seed_repeat --device cuda
```

The **v2** package fixes the initial update's saved-JSON loading error. The
launcher used JSON, whereas the child used the YAML loader, which interpreted
JSON `1e-05` as a string. The exact-config guard therefore stopped the first
child before training. JSON inputs now use the JSON parser; YAML behavior and
the strict pairing checks remain unchanged. Preflight uses the same config
loader as training. No hand edits to effective_config.json/checkpoints/YAML
or deletion of runs is needed for that failure. Its existing log is appended
on restart. The 15 `RUN` lines preceding `PLAN` list planned jobs, not completed
training. If a different run has already created partial model artifacts, the
existing no-overwrite guard still applies.

The first phase verifies ALL selected inputs before the first training starts.
Any missing/wrong source, baseline, sidecar, patient split, or cache stops the
run; it does not silently retrain another method. It then runs 15 target jobs
sequentially with automatic final Test and logging. Reserve about **5 GiB** of
free space (new best checkpoints plus the existing 2-GiB per-job free-space
guard). No files are deleted. Actual checkpoint sizes depend on configuration.

Defaults:

- baseline: `runs/source_initialized_adaptation_voxel_tokenizer`
- new output: `runs/source_initialized_adaptation_voxel_seed2027`
- folds: `artifacts/dataset_who_binary/folds.json`
- original source path/config/protocol: read from each baseline checkpoint.

`--dry-run` only validates/plans (no writes, training, MRI inference or Test).
`--fold 3 4` / `--target-task survival` restrict execution if needed; they do not
change defaults or permit mixing per-fold winners. Completed fingerprint-verified
jobs are skipped on a rerun; incomplete jobs are not overwritten/resumed. If a
run fails or is interrupted before its completion marker, preserve that directory
and explicitly restart the missing task in a new `--output-dir`.

## Results

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.report_voxel_seed_repeat
```

This prints all current repeat Test metrics (including IBS and fold-specific
log-rank), selected epochs, paired old/new validation and Test primary metrics,
paired metric differences and mean absolute differences. Partial results are
not complete five-fold estimates. No average p-value, winner selection,
imputed missing metric, or pooled-prediction ensemble is computed.

Changing scores under fixed patients demonstrates sensitivity to the changed
training run; consistently weak folds across two seeds instead makes that
particular seed an insufficient explanation. Neither result establishes a
unique cause. Do not pick a better seed for each fold or use these development
tests to tune further seeds. Later robustness claims require more independent
replicates and an evaluation not repeatedly used for method development.
