# Source-trained Q/K low-rank transfer

Approved conversational design: keep voxel + spatial + two private blocks,
add rank-8 Q/K updates to all 12 MAE attention layers. No new loss or fresh sweep.
User requests direct implementation without repeated confirmation gates.

## Contract

- Base MAE, source shared/private adapters and heads retain their existing paths.
- Q/K deltas: Wq'=Wq+Bq Aq, Wk'=Wk+Bk Ak, rank=8, scale=1 (alpha=rank).
  V and output projection are unchanged. Initialize A with Kaiming uniform,
  B with zero; isolate constructor RNG so legacy head initialization is unchanged.
- Store source deltas outside encoder state. Include them in AdamW with the
  shared-adapter LR and in PCGrad's shared parameter set.
- Train new source models from the same MAE and protocol. Never load old sources
  as if they had learned Q/K deltas. Source stage never consumes held-out labels.
- Target owns independent copies of source Q/K and shared adapters. Target Q/K
  uses adapter LR. Original source weights and predictions remain unchanged.
- Default off: no new tensors/random draws/prediction changes in legacy runs.
- Reject composition, legacy residuals, bridge, retention, contrast, SAM,
  covariance, bypass, warmstart and refit combinations for this v1 profile.
- Preserve seed2026, crop .5, batch-weight-mean CE, splits, selection, metrics,
  100% labels and existing learning rates. New roots source_qk_learning and
  source_qk_adaptation; best-only checkpoints and automatic final Test.
- Version/rank metadata in checkpoints/results; strict reload and mismatch guard.
- Runner defaults all five folds and three directions; completed-run reuse only
  with verified config/input/code/artifact hashes. No overwrite of incomplete runs.

## Evidence boundaries

This tests a structural hypothesis, not a proven failure cause or new LoRA method.
No performance guarantee. Repeatedly inspected outer folds remain exploratory.
Scientific adoption requires cross-task results, not per-fold winner selection.

## Verification

Zero-update equivalence; nonzero Q/K math reference; gradients/update of deltas;
source optimizer/PCGrad membership; independent target copies; immutable source;
legacy RNG/state compatibility; config mismatch rejection; all three directions
tiny-cache source training -> target adaptation -> identical test-only reload;
runner guards, report identity, full suite and representative GPU backward.
