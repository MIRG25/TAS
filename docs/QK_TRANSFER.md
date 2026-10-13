# Source-trained Q/K transfer (2026-10-09)

This is an unvalidated architectural experiment built on the fixed seed2026
voxel + spatial adapter baseline. It does not establish improved performance.

## One structural change

Each of the 12 existing MAE blocks gets rank8 deltas to its Q and K projection
weights. Scale is one (LoRA alpha=rank). V, attention output and original MAE
weights remain frozen during source learning. Added source parameters: 147,456.
The deltas are outside encoder_state, so old MAE weights remain byte-identical.
Down matrices use the Linear default Kaiming-uniform initialization; up matrices
are zero. Module construction isolates Torch RNG. At initialization the forward
function matches the original (within floating-point kernel precision).

Source tasks jointly train shared Q/K deltas, shared adapters, private adapters
and their heads. Q/K uses the existing shared-adapter learning rate and joins
PCGrad's shared parameters. No held-out task label is used by this phase.

Target adaptation copies shared Q/K and adapters into independent target-owned
storage; source Q/K/adapters/heads remain frozen. The target also retains the
original voxel stem, spatial adapters, two private MAE blocks and task head.
Target Q/K uses the existing target-adapter learning rate. Old task inference
does not enter target modules. No clinical variables or other-task labels enter
the MRI model. Target validation alone selects the checkpoint; Test is then
evaluated once by the child entry point and saved with patient predictions.

Q/K is default-off. Legacy checkpoints have no new tensor keys and retain the
old execution path. The new config refuses old source checkpoints. Extra
metadata is embedded in source/target checkpoints, results and provenance;
test-only reconstructs it from the checkpoint config.

## Unchanged experimental conditions

- Seed2026; label seed2028+fold; the existing binary-WHO transfer protocol.
- 100% target labels, including inner validation; no full-budget refit.
- Same cache, 128-cubed MRI input, crop probability .5, losses, optimizer rates,
  scheduler, validation selection, patience, pooling and private-block depth.
- No composition, SAM, contrast, retention, bridge, bypass, covariance,
  head-warmstart, seed search or fresh sweep.
- Best-only checkpoint storage. No optimizer resume, last or periodic weights.

## Deploy and run

Upload `outputs/source_qk_transfer_update_20261009.tar.gz` to the server project
`outputs/` folder. This is a patch for the current project, not a dataset or
standalone installation. Keep its existing cache, folds/protocol and MAE best.pt.

```bash
cd /root/autodl-tmp/project/glioma_meta_learning &&
tar -xzf outputs/source_qk_transfer_update_20261009.tar.gz &&
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u -m scripts.run_qk_transfer --device cuda
```

Default: sequentially train 15 NEW source models and 15 target models covering
fold0-4 and all three held-out tasks. Do not pass legacy source checkpoints.
No new MAE pretraining or cache generation is performed.

New roots are `runs/source_qk_learning` and `runs/source_qk_adaptation`.
Each job streams and saves its log; each target automatically prints Test.
The batch runner prints the aggregate report at completion. At any time:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.report_qk_results
```

Optional command preview without writes, MRI training or tests:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m scripts.run_qk_transfer --device cuda --dry-run
```

`--fold 0` or `--target-task survival` restricts jobs explicitly. The launcher
checks all selected patient splits/inputs before training. Completed jobs can
be skipped only when configuration, implementation, source and output hashes
match their completion receipts. Incomplete directories fail closed: no silent
overwrite, deletion or restart. A selected best.pt alone is not a completion
receipt. Logs/results of interrupted jobs must be preserved before deciding
whether to restart in a separately named directory. At least 2 GiB free is
checked per job; allow additional space for all 30 checkpoints and outputs.

The report verifies completion receipts, effective-config/checkpoint/prediction
hashes and a common configuration/code/protocol/manifest/MAE identity across
folds. Run it from the project root and retain the source checkpoints. These
checks do NOT hash the large MRI/cache/mask contents. This experiment assumes
those files are immutable; do not regenerate or replace them between jobs.
Changing data contents requires a separately identified experiment, even if
the file paths remain identical. A receipt is an integrity check, not proof of
dataset identity or an adversarial tamper-proof signature.

## Interpretation

Q/K updates directly expose a previously frozen parameter subspace to source
supervision. Existing adapters can already change attention indirectly through
their token inputs; this is not evidence that the baseline learns no relations.
The hypothesis is not a diagnosed cause of fold gaps. LoRA itself is established,
not a new-method novelty claim. Zero initialization preserves the initial
function but cannot guarantee the selected or Test model does not regress.

All three task directions use the same structural design. Compare complete
five-fold reports without selecting per-fold winners, thresholds or seeds from
Test. Missing results are not zeros; partial means are not complete five-fold
results. IBS is lower-is-better; log-rank p is reported per fold, never averaged.
These repeatedly inspected outer tests remain exploratory. Low-label transfer
claims require later label-budget-matched evidence, not this 100% experiment.

## Verification scope

Local full regression after review fixes: 530 passed, 3 skipped, 2 subtests
passed. The extra report acceptance/mixed-identity regression also passed in
the five-test launcher suite. Existing NumPy/pandas deprecation warnings remain.

Tests cover explicit Q/K attention math, even-head eval/no_grad execution,
default-off compatibility/RNG, source PCGrad and optimizer coverage, independent
target storage and preservation of original source weights/predictions. Tiny
cache source/target/test-only roundtrips pass for all three task directions.
The batch launcher is tested for all 30 jobs, checkpoint provenance, refusal
to overwrite incomplete runs, mixed-config rejection and result integrity.
A synthetic CUDA FP16 check with batch2, four modalities and 128-cubed inputs
completed two backward/optimizer steps on a local RTX3070Ti Laptop GPU. This
does not validate clinical scores, performance improvement or full server runs.
