# Local verification: frozen threshold audit

Date: 2026-10-07. No real patient training or new cohort evaluation was run locally.
The user's original trained checkpoints remain on the server.

Environment: Windows, Python 3.10 (`cmh3.10`), PyTorch 2.6.0+cu118.
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

Executed:

```text
python -m pytest tests/test_classification_thresholds.py tests/test_sam.py tests/test_sam_transfer.py tests/test_voxel_tokenizer.py tests/test_voxel_tokenizer_launcher.py -q
79 passed in 15.27s
python -m scripts.audit_classification_thresholds --help
exit 0
```

Existing pandas/NumPy deprecation warnings were emitted; no test failures.
Sixteen threshold-specific tests include:

- Binary argmax/0.5 equivalence and explicit equal-score behavior.
- AUC invariance under decision-threshold changes.
- Exact objective ties and deterministic tie-breaking, row-order invariance.
- Rejection of missing labels, invalid probabilities and single-class validation.
- Threshold selection accepts validation arrays only, unaffected by test labels.
- Actual tiny synthetic MRI-cache pipelines for both IDH and WHO, voxel and SAM.
- Metadata-only dry run performs no inference or writes.
- No optimizer/backward calls during the audit; all parameters frozen.
- Threshold lock exists before Test inference; both default and threshold metrics saved.
- Original validation/Test metrics replay, complete patient sets, unchanged checkpoint bytes.
- Hash-verified completed-job skipping; report-only makes no new MRI passes.
- Rejection of duplicate/missing results, changed audit files and replay mismatches.
- Verified same-condition SAM-minus-voxel reports, without per-fold method selection.

Only new evaluation helpers/entry point/tests and documentation were changed.
Training scripts, network architecture, metric implementation, experiment YAMLs,
original checkpoints and original result files were not modified.

These checks establish implementation behavior on synthetic data, NOT improved
clinical accuracy. The server command must evaluate the actual saved models.
