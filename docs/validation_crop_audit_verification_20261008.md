# Verification: frozen validation crop audit

Date: 2026-10-08. Only synthetic data were used locally; no UCSF MRI cohort
inference, source training or target training was run for this change.

Implementation is isolated in two new scripts and one evaluation helper.
No edits to training, network, production crop or production evaluation code.
README now marks the covariance experiment as a completed mixed-result ablation.

## Tests

Environment: Windows, Python 3.10 environment `cmh3.10`, PyTorch CUDA runtime;
OMP/MKL threads 2, external pytest plugin autoload disabled.

```text
python -m pytest tests/test_crop_sensitivity.py tests/test_crop_audit.py \
  tests/test_source_initialized_transfer.py tests/test_voxel_tokenizer_integration.py -q
75 passed in 30.41s
```

`GLIOMA_CROP_AUDIT_CUDA_SMOKE=1` enabled the production-size synthetic check:
frozen FP32 forward on 2 x 4 x 128 x 128 x 128, repeatable finite outputs and
unchanged complete model state. This is not a speed estimate for the real cohort.
Existing pandas/NumPy deprecation warnings were present; no test failures.

New tests cover:

- Exact center-crop parity with `_online_crop`, including odd shapes, boundary
  tumors, full masks and empty masks. No RNG advancement or intensity changes.
- Aligned four-channel offsets, clipping, no padding/resizing, and contiguous FP32.
- Deduplication of repeated actual crop starts, class flips, rank/SD edge cases.
- All three target tasks with real tiny synthetic cache/model forward passes.
- Guarded file reads: only saved validation MRI IDs may be loaded; no fit/test
  MRI. Deliberately corrupted test labels do not affect audit execution.
- Optimizer/backward forbidden; all model tensors and checkpoint bytes unchanged.
- Original validation metric mismatch stops before the six shifted views.
- Strict baseline/protocol validation, duplicate checkpoint rejection, read-only
  preflight, completed-run hash verification and no overwrite of partial runs.

Actual saved model performance/shift sensitivity remains to be measured on the
server. The audit does not imply that cropping causes the observed fold gaps.

## v2 checkpoint-diagnostic fix

The server reported the generic `validate_baseline` error. That traceback does
not reveal the failed field; the server weights are not available locally.
No specific metadata mismatch is assumed or silently rewritten.

- Every failed check now includes field, actual/expected value, checkpoint
  path and task/fold/epoch. New `--inspect-checkpoints` scans all files without
  MRI, protocol, GPU or output-file writes.
- Exclude consistently declared matched-fresh controls during discovery;
  still reject ambiguous identities, duplicated source-shared runs and any
  genuinely incompatible source-shared baseline.
- Preserve architecture/seed/label-budget rules and all later split/weight/
  center-reproduction checks. No relaxed validation or checkpoint modification.
- Additional targeted suite: `24 passed, 1 skipped in 11.57s`. The skipped test
  was the opt-in full-size GPU smoke, already passed for v1; no GPU rerun needed
  for metadata-only changes. Tests cover multiple mismatches at once, inventory
  without protocol/cache/GPU, fresh exclusion and unchanged rejection rules.
