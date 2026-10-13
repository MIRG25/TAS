# Cross-depth residual readout pilot (2026-10-10)

## New WHO endpoint: explicit opt-in

The legacy commands below still mean II/III versus IV. To use I/II=0 and
III/IV=1 **with the same cross-depth structure**, install
`outputs/cross_depth_who_low_high_update_20261010.tar.gz`, then run:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [ ! -d artifacts/dataset_who_low_high ]; then
  python -m scripts.prepare_binary_who_artifacts || exit 1
fi
if [ ! -f artifacts/dataset_who_low_high/transfer_protocol.json ]; then
  python -m scripts.prepare_transfer_protocol --folds artifacts/dataset_who_low_high/folds.json || exit 1
fi
python -u -m scripts.run_cross_depth_transfer --who-low-high --device cuda --dry-run && \
python -u -m scripts.run_cross_depth_transfer --who-low-high --device cuda
python -m scripts.run_cross_depth_transfer --who-low-high --report-only
```

Stop old training before installing; do not subsequently unpack the old update
over this launcher. Default remains Fold0–1 / three directions. To run all five
folds, append `--fold-indices 0 1 2 3 4` to both dry-run and training commands.
The launcher trains missing new-protocol sources under
`runs/source_initialized_learning_who_low_high`, then adapts and automatically
tests under `runs/source_initialized_adaptation_cross_depth_who_low_high`.
Compatible completed sources are reused. No fresh or no-cross-depth target arm
is required. Source training does not add the target-only cross-depth readout.
Training seed remains 2026; label seed is 2028 + fold. Layer taps 4/8/10 and rank
64 are unchanged. Patient pools and MRI caches are preserved; protocol identity
changes with WHO definition. Old-label results are not paired against new labels.
Completion receipts validate input and output hashes. Incomplete directories
are refused rather than overwritten; interrupted training is not auto-resumed.
The local dataset has grade II/III/IV, so this mapping yields 56 low and 438 high;
there are no grade-I patients to add. Accuracy alone is insufficient in this mix.

## Question and fixed design

Can direct access to intermediate MRI representations improve overall transfer,
without replacing the final-layer representation or changing source learning?
Existing subtype-stratified errors motivate this hypothesis but do not establish
its cause. This is an unproven architecture ablation, not a promised improvement.

The target keeps endpoint-v2, seed 2026, original voxel tokenizer, spatial adapter,
two private blocks, mean pooling, crops, losses, optimizer and early stopping.
At 1-based layers 4, 8, 10, immediately after each target adapter: exclude CLS,
non-affine LayerNorm per token, mean over spatial tokens, concatenate. A bias-free
1152 -> 64 -> 384 GELU readout adds a residual to the unchanged final pooled vector.
The last projection starts at zero. Total new parameters: 98,304. Its constructor
preserves the existing CPU RNG stream. Source task paths never use this module.
This generic multi-layer readout is not claimed as a new established clinical method.

Disabled by default, with no extra legacy state keys. The same structure applies
to IDH, WHO and survival; no subgroup labels, age or other extra inputs. WHO follows
the existing protocol: II/III versus IV. No threshold search, no source retraining,
no fresh baseline and no mixing with SAM, covariance, Q/K or composition.

## Server commands

Upload outputs/cross_depth_readout_update_20261010.tar.gz into the project outputs/
directory, then from the project root:

```bash
tar -xzf outputs/cross_depth_readout_update_20261010.tar.gz && \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m scripts.run_cross_depth_transfer --device cuda --dry-run

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -u -m scripts.run_cross_depth_transfer --device cuda

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.run_cross_depth_transfer --report-only
```

Default: Fold0 and Fold1, survival/WHO/IDH, six target adaptations, automatic Test.
The launcher derives each configuration from that endpoint-v2 run's saved best.pt,
not from mutable YAML. Defaults to runs/source_initialized_adaptation_endpoint_v2.
Use --baseline-root only to point at a relocated *same experiment*. Its best.pt,
effective_config.json, data_split.json, test_metrics.json and original source best.pt
must remain available. All six runs are preflighted before training. No files are
written by --dry-run. Baseline artifacts, source tensors and patient lists are checked.

New outputs: runs/source_initialized_adaptation_cross_depth. Saves best.pt only,
logs, predictions, effective config, paired input hashes and completion receipts.
Completed verified runs are skipped on relaunch. Incomplete directories are refused,
not silently resumed or overwritten; this command does not implement interrupted
training recovery. Keep at least 2 GiB free (more for six checkpoints).

## Interpretation

Report all six paired outcomes. Look at survival C-index, WHO/IDH AUC together with
ACC/SEN/SPEC, and survival IBS (lower is better). No selecting a different module
per fold; no test-driven epoch/seed choice. Six runs cannot demonstrate five-fold
improvement or low-label superiority. Repeatedly inspected test sets are exploratory;
a locked independent evaluation is needed for confirmatory claims. Remaining folds
are not automatically launched. No clinical efficacy is inferred from synthetic tests.

## Local verification

- Final complete suite: 616 passed, 4 skipped, 2 subtests passed (282.05 seconds).
- Opt-in production CUDA/AMP test: 1 passed; synthetic batch 2 x 4 x 128^3,
  depth 12 / width 384, two optimizer updates, source tensors unchanged.
- All three tasks: tiny-cache training -> automatic Test -> strict test-only reload.
- Initial predictions and RNG preserved; optimizer covers every trainable parameter;
  gradients reach the readout and earlier target adapters; old source predictions unchanged.
- Launcher: read-only preflight, mismatched source refusal, simulated subprocess completion
  with real receipt/report/skip checks, tampered prediction refusal. Clinical six-run
  launcher execution has NOT been performed locally.
- Independent code review found an output-suffix protection issue; reproduced with a
  failing test, then repaired and included in the final passing suite.
