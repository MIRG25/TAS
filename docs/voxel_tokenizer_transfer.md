# Voxel residual input adapter (2026-10-06)

This is an unvalidated architectural experiment, not an established improvement.
It extends the original spatial-adapter baseline, not patient contrast, SKB,
attention pooling, all-block adaptation, warmstart, or token retention.

## Architecture and unchanged conditions

- MRI only: T1/T2/T1c/FLAIR, same 128-cubed crop. No source-task labels,
  clinical variables, segmentation channels, or extra MRI crops enter the model.
- Retain the frozen MAE 16-cubed patch projection. Add a target-private
  residual stem: 3x3x3 stride-two Conv3d + GroupNorm(4) + GELU, channels
  8/16/32/32, then zero-output 1x1x1 Conv3d to 384 channels.
- Add residuals only to the 512 spatial tokens, preserving the original CLS
  and position embedding. Extra parameters: 58,256 per target.
- The source-shared adapters are still copied from the original supervised
  source model; two private Transformer blocks and spatial adapters stay as
  before. Gradients reach the stem through frozen blocks, without optimizing
  their weights. Old source tasks bypass this stem entirely.
- No source/MAE retraining. No additional loss, pooling, crop or learning-rate
  change. New stem uses the existing adapter learning rate (1e-4).
- Constructor RNG is isolated. The zero projection preserves the starting
  function; this is NOT a guarantee of equal optimization trajectories or
  non-decreasing final scores. First-step stem gradients before the zero
  projection are zero; they become nonzero after the projection learns.
- Tiny tests also allow smaller isotropic power-of-two patch sizes. Production
  remains 16. Old checkpoints default to the feature being disabled.

## Server commands

Run from `/root/autodl-tmp/project/glioma_meta_learning` after extracting the
update there. Stop any old patient-contrast launcher in its own terminal first;
this program will not kill existing processes.

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u -m scripts.run_voxel_tokenizer_transfer --device cuda
```

Default: folds 0-4 and all three directions, 100% labels, seed 2028 + fold.
Only original checkpoints under `runs/source_initialized_learning` are reused.
All selected source metadata/splits are audited before any job starts. Missing
sources cause an error, never an automatic source retrain. The runner rejects
contrast/retention sources, crop mismatch, refits, existing target directories,
and mixed architecture profiles. It writes stdout/stderr to per-run log files.
Each target automatically evaluates its validation-selected best.pt on the
outer test and prints `test_metrics`; no separate test command is needed.

Optional bounded run (not required before the default command):

```bash
python -u -m scripts.run_voxel_tokenizer_transfer --fold 0 --device cuda
```

Plan without training/writing: add `--dry-run`. For remaining jobs, specify
`--fold 1 2 3 4` and/or `--target-task survival who idh`; an incomplete run is not
automatically resumed or overwritten. Use a separate output root for a restart.

## Storage and reporting

New root: `runs/source_initialized_adaptation_voxel_tokenizer`. Each directory
ends in `_spatial3d_voxel`. Preserve `best.pt`, `effective_config.json`,
`data_split.json`, history, final `test_metrics.json`, and patient predictions.
Best-only weights omit optimizer/scaler state: no periodic or last.pt files,
and no exact optimizer resume. Writes are atomic; at least 2 GiB free is checked
before every job. No old experiments/caches/checkpoints are deleted.

```bash
python -m scripts.report_voxel_tokenizer_results
```

This prints all three tasks, fold progress, Test mean/sample SD, survival IBS,
and per-fold log-rank statistics. Missing folds are never zero or a complete
five-fold result. No training or testing is triggered by the report command.

Matched fresh arm (NOT launched with the primary command):

```bash
python -u -m scripts.run_voxel_tokenizer_transfer --target-initialization fresh --device cuda
python -m scripts.report_voxel_tokenizer_results --target-initialization fresh
```

Fresh still uses the same frozen MAE and architecture, but initializes the
target adapters without source-shared weights. It is not random-backbone
training. Both arms use the same patients, label budgets, initial head/stem,
capacity, optimizer, and selection rules. The runner currently locks 100%;
low-label budgets remain supported by `scripts.adapt_source_guided` directly.

## Research interpretation

Small-convolution stems are established techniques, not a novelty claim here.
This targets an untested input-side limitation; current subgroup errors do not
prove patch compression is the cause. Extra early-layer gradients may increase
compute/memory despite few added parameters; RTX 3090 24GB suitability must be
checked on the actual server. These outer folds have already informed design
choices: subsequent results remain exploratory, not untouched confirmation.
Only training/validation may choose epochs; do not choose per-fold architectures
or thresholds using Test. No C-index/ACC improvement is promised.

Local implementation validation: 329 tests plus two subtests passed in the
existing cmh3.10 environment (PyTorch 2.6.0+cu118), including three-task tiny-cache
training, automatic testing, identical saved/reloaded predictions, strict source
parameter preservation, backward compatibility, and matched fresh capacity.
A separate synthetic 2x4x128x128x128 full-depth FP16 backward check passed on an
8GB RTX 3070 Ti Laptop GPU. Default GradScaler skipped one initial overflowing
head update and backed off from 65536 to 32768, then two finite updates succeeded;
the input stem received nonzero gradients after the zero projection updated.
This verifies code execution, not clinical performance or server runtime.

## Previous failed direction

Do not continue patient-contrast or reuse its source weights in this experiment.
Completed survival folds 0-2: contrast mean C-index 0.6595 vs the same original
spatial-adapter folds 0.7121. These are partial exploratory results, not a full
five-fold significance claim. Keep the completed files as a negative result.
