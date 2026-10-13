# Implementation verification: covariance residual readout

Date: 2026-10-07. Local Windows Python 3.10 (`cmh3.10`), PyTorch 2.6.0+cu118.
Only synthetic caches/input were used. No real UCSF patient training/evaluation,
no server job execution, and no changes to existing checkpoints or results.

## Executed checks

```text
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest tests -q --disable-warnings
402 passed, 2 skipped, 2 subtests passed in 192.94s

GLIOMA_COVARIANCE_CUDA_SMOKE=1
python -m pytest tests/test_covariance_cuda_smoke.py -q -s --disable-warnings
CUDA covariance: loss=2.420166 finite=True scale=65536.0
CUDA covariance: loss=2.296753 finite=True scale=65536.0
CUDA peak allocated GiB=0.405
1 passed in 7.19s

python -m scripts.run_covariance_transfer --help
python -m scripts.report_covariance_results --help
Both exited 0.
```

The full-suite skipped tests are opt-in real-size CUDA tests; the new covariance
one was then run explicitly. Deprecation warnings from existing dependencies
remain. The GPU measurement is PyTorch peak allocated memory during this
synthetic smoke only, not a server resource estimate or training benchmark.

## Coverage

- Exact covariance formula, trace scale, symmetric vectorization, no CLS.
- Equal-mean/different-covariance examples give different second-order features.
- Token permutation invariance; patient independence; constant/N=1 stability.
- Invalid ranks/flag combinations rejected, including mixed failed architectures.
- Zero residual initially matches voxel reference outputs, existing parameter
  values, and RNG stream. Disabled module adds no state keys to old checkpoints.
- All three target tasks: gradient flow, exact optimizer parameter coverage,
  copied-not-tied weights, unchanged original source parameters/predictions.
- CPU bfloat16 and CUDA float16 backward with gradient checkpointing.
- Production shape: batch 2, 4 channels, 128-cube, D384, 12 blocks, last two
  target-private blocks, spatial adapters, voxel tokenizer, rank32 readout.
- Real tiny-cache train -> validation-selected best -> automatic Test for all
  three tasks; metrics and predictions saved with architecture metadata.
- Test-only reload requires no optimizer/external source checkpoint and leaves
  the checkpoint file unchanged. Predictions agree within FP32 tolerance 1e-6;
  the synthetic survival IBS differed by ~2.4e-9 across attention kernel paths.
- Rank/version/enable metadata tampering rejected; strict model-state loading.
- Runner plans 15 target-only jobs, no source training, no fresh, no
  validation-only flag, no writes in dry planning; existing paths rejected.
- Config differs from voxel reference only in readout flag/rank and name/root.
- Partial reports do not fill missing folds, mix variants or average p values;
  paired deltas require matching provenance, patient IDs and effective settings.
- Existing voxel, SAM, threshold-audit, source/target protocol and other tests
  remain passing. Failed experimental configurations were not enabled.

These tests verify implementation behavior, NOT improved clinical performance
or stability. The server experiment is a developmental architecture ablation.
