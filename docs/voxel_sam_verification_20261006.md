# Local SAM implementation verification (2026-10-06)

Environment: Windows, Python 3.10 (`cmh3.10`), PyTorch 2.6.0+cu118;
OMP_NUM_THREADS=2, MKL_NUM_THREADS=2, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
This verifies execution and safety, **not clinical performance** on UCSF-PDGM.

- Full existing-plus-SAM suite: 355 passed, 2 subtests passed (199.40 s).
- After adding checkpoint-metadata rejection tests and tightening report CLI:
  44 selected unit/launcher/report/integration checks passed (10.70 s).
- Separate opt-in real-size CUDA/FP16 test: 1 passed (7.79 s), synthetic
  2 x 4 x 128 x 128 x 128 input, same architecture and gradient checkpointing.
  Initial second-pass overflow safely skipped; scaler 65536 -> 32768;
  then two finite AdamW updates. Frozen source state remained exactly equal.
- Three-task tiny-cache training automatically evaluated only the saved outer
  test IDs, preserved source weights, and saved best-only checkpoints.
- Independent test-only reload reproduced each task's metrics exactly without
  constructing an optimizer or modifying the checkpoint.
- SAM absent vs explicitly disabled produced identical tensors and training
  histories. Original voxel regression tests also passed.
- Numerical reference checks cover one AdamW descent (not two updates), ascent
  before clipping, exact restoration after exceptions, identical stochastic
  masks between passes, finite large-gradient norms, zero gradients, CPU/CUDA
  AMP overflow at either pass, and frozen-parameter rejection.
- Effective config comparison verifies the candidate changes only SAM,
  experiment name and output root relative to the completed voxel reference.

Warnings were existing pandas/NumPy deprecation warnings. No patient training
or outer-test performance evaluation was run locally in this verification.
