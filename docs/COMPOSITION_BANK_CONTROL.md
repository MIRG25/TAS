# Fold0 control: target component-bank trainability

Question: did freezing the copied source component bank restrict target adaptation?
This is a single-factor exploratory control, not a new architecture or a promise
of improvement. No additional source training and no automatic five-fold sweep.

The optional setting is:

```yaml
adaptation:
  composition_trainable_bank: true
```

It defaults to false. It affects only the target copy of each component bank
(LayerNorm, down projection, up projection). The original source banks, routes,
heads and private source adapters remain unchanged. Target routing, spatial
mixing, voxel stem, two private blocks and head are otherwise trained exactly
as before. Initialization and RNG consumption are unchanged by the switch.

## Execution

Upload this archive to the server project's `outputs/` directory, then run from
the project root:

```bash
tar -xzf outputs/composition_bank_control_update_20261009.tar.gz &&
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u -m scripts.run_composition_bank_control --device cuda
```

Default scope: Fold0, survival / WHO / IDH, 100% target labels, seed 2026.
The launcher reads `runs/source_compositional_adaptation` completed results,
the corresponding best.pt, effective_config.json and completion receipts. It
checks hashes of the original source, target checkpoint, metrics and predictions.
It copies each actual saved config, changing only the new trainability flag,
experiment name and output directory. It does not rely on a recently edited YAML.
Generated configs are under `runs/source_compositional_bank_trainable/_inputs`.

Every run uses the original source initialization (not the selected target
checkpoint), the original target learning rates, optimizer and patient protocol.
The existing training script selects on validation, then automatically evaluates
Test and prints all task metrics. Source hashes and paired patient/seed metadata
are checked after training. New outputs and logs go under
`runs/source_compositional_bank_trainable`; frozen outputs remain untouched.

Optional preflight (no writes, training or test):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.run_composition_bank_control --dry-run
```

Summarize completed controls later:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.run_composition_bank_control --report-only
```

The final summary prints saved `test_metrics` and differences against each
frozen compositional counterpart. No mean log-rank p or five-fold estimate is
produced. A failure stops the batch. To run unstarted tasks only, use for example
`--target-task who idh`; an existing partial target directory is never overwritten.

## Interpretation and boundaries

The primary comparison is trainable versus frozen COMPOSITIONAL target bank,
not versus the original voxel model (which also had different source weights).
If there is no consistent validation evidence of benefit, do not automatically
expand to folds 1–4. Test results are descriptive: these outer tests have been
repeatedly inspected during development and are not an untouched confirmation set.

This code update intentionally changes the implementation hash used by the old
`run_compositional_transfer` launcher. Do not resume that old 15-run launcher:
its existing completion fingerprints will no longer match. This control launcher
verifies immutable artifact hashes directly and never schedules source training.
Default-false behavior remains available for loading old checkpoints.
