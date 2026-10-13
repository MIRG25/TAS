# Local verification: target training-seed repeat

Date: 2026-10-08. Windows, existing `cmh3.10` environment, CPU tiny-cache tests.
No production MRI training, server jobs, source checkpoint edits or deletions.

## Scope

New `--training-seed` / `--repeat-baseline-checkpoint` arguments in
`scripts.adapt_source_guided`; independent repeat launcher and paired report.
No network, losses, scheduler, selection rule, clinical label definitions,
patient protocol, or existing YAML is changed. Omission of the new arguments
retains previous seeds and directory names.

## Executed checks

```text
python -m pytest tests/test_voxel_seed_repeat.py
    tests/test_voxel_tokenizer_launcher.py tests/test_voxel_tokenizer_integration.py
    tests/test_source_initialized_transfer.py tests/test_crop_sensitivity.py
    tests/test_sam_transfer.py tests/test_covariance_transfer.py -q
120 passed, 1 skipped (46.62 s)
```

The skip is the existing opt-in full-resolution crop-audit CUDA check; it was
not requested for this seed/metadata-only change. Existing pandas/NumPy
deprecation warnings remain. After tightening completion/provenance checks,
the new test file was rerun separately (18 tests).

- Real tiny-cache source/target execution for IDH, WHO and survival.
- Each task: baseline seed 2026, repeat seed 2027, deterministic replay 2027.
  Exact replay model tensors/predictions agree; new seed changes target weights.
- Source checkpoint bytes, original target checkpoint bytes and frozen source
  tensors remain unchanged; fit/validation/test/budget IDs and label seed agree.
- Automatic final Test, best-only checkpoints, prediction seed metadata and
  read-only paired report exercised, including unavailable survival AUC values.
- Actual seed-repeat preflight, completed-run fingerprints/skip, partial-run
  rejection and configuration/source/patient mismatch rejection exercised.
- Fifteen-job planning fixture: all three targets and folds 0-4, no fresh/source
  jobs. A source-weight mismatch in the last job prevents ALL training.
- Invalid seeds and same-seed repeats fail; no inference during dry planning.
- CLI help for both new commands checked.

These tests validate implementation, not MRI cohort performance. Exact GPU
reproducibility and equivalence to a historical software/hardware environment
are not asserted. The current runtime is recorded; old runtime metadata may
be absent. Two seeds provide a diagnostic comparison, not proof of robustness.
