# Stopped compositional experiment: checkpoint-only audit

This diagnostic does not propose another training variant. Keep seed 2026 and
the completed voxel + spatial baseline unchanged. Do not restart the stopped
15-run sweep or choose a different method/seed for individual test folds.

Run from the project root after uploading the update archive:

```bash
tar -xzf outputs/compositional_route_audit_update_20261009.tar.gz &&
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u -m scripts.audit_compositional_routes
```

The default input is `runs/source_compositional_adaptation`. Only direct
`fold*/test_metrics.json` final results and their corresponding `best.pt` are
read. Partial experiments and nested test-only runs are not audited. Each
checkpoint is loaded sequentially on CPU; no MRI, GPU, model construction,
optimizer or evaluator is used. Use only trusted project-created checkpoints.

The script prints a compact table and creates one timestamped JSON under
`outputs/`. It refuses to overwrite an existing report or write inside runs.
Checkpoint and result hashes are checked before/after inspection. A mismatch
between the checkpoint epoch, patient protocol, task or result raises an error.

## Reading the table

- `Source-route-TV`: mean layerwise total variation distance between source-task
  component probabilities. Zero means identical coefficients, not necessarily
  identical features or failed learning.
- `Target-change-TV`: mean distance from the mean source probabilities used to
  initialize the target to its selected-checkpoint probabilities. The JSON also
  contains per-layer values and source routing drift from deterministic initialization.
- `Banks-equal`: exact equality of the saved target component-bank tensors to
  the embedded source-bank tensors. Expected to be true in v1. This confirms
  equality at this checkpoint only, not the complete optimization history or the
  integrity of the external original source file.

Total variation lies in [0,1]. There is no prespecified threshold proving failure.
Component weight scales can compensate for routing coefficients; this is not a
functional ablation. The baseline updated the target adapter bank whereas v1
freezes the copied bank. This architectural difference is a candidate limitation,
not a causal conclusion from the current partial test results.

The report includes previously recorded test metrics explicitly labelled as
existing results. It performs no new test evaluation and no model selection.
These repeatedly inspected outer tests remain exploratory. Do not interpret
partial results as a complete five-fold estimate or use this audit to pick
per-fold winners. The immediate next decision is whether a concrete routing or
copying defect is evidenced; do not automatically launch another full sweep.
