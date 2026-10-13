# Q/K stage audit — existing records only

Purpose: compare the stopped Q/K experiment with the fixed voxel + spatial
baseline without training, checkpoint loading, MRI inference, or new Test
evaluation. This is a bounded diagnostic addition, not another model variant.
No existing model, trainer, configuration, result or checkpoint is modified.

The local workspace does not contain the server's completed Q/K runs. The
diagnostic therefore runs on the server, using only Python's standard library.

## Run

Upload `outputs/qk_stage_audit_update_20261009.tar.gz` into the project `outputs/`
directory, then run from the server project root:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning &&
tar -xzf outputs/qk_stage_audit_update_20261009.tar.gz &&
python -u -m scripts.audit_qk_stages
```

Default: Fold0, all completed target directions, comparing
`runs/source_initialized_adaptation_voxel_tokenizer` with
`runs/source_qk_adaptation`. Matching uses task, fold and label seed; only
source-shared, 100% results are included. Nested retests are not selected.
Use `--fold 0 1` only if those folds have already completed; it does not train
anything. `--baseline-root` and `--qk-root` can point to relocated result roots.
Source directories are read from each result's `source_checkpoint.path`.

Required target files: `test_metrics.json`, `training_history.json`,
`data_split.json`, `effective_config.json`. The source's corresponding history,
split and config are used when available; missing evidence is reported, never
fabricated. No `.pt` file needs to be loaded. Duplicate matching results fail
rather than choosing an arbitrary run.

## What is measured

- Exact target fit/validation/test/budget patient sets and source patient sets.
- Protocol agreement, duplicate IDs and fit/validation/test overlap checks.
- Source primary metrics at the **recorded source checkpoint epoch**, not an
  epoch re-selected by target results; source and target curves are retained.
- Target validation at its recorded selected epoch and last validated epoch,
  paired with training loss. Test is reprinted from the saved result only.
- Source and target configuration differences; Q/K flags, experiment/output
  names and source checkpoint storage policy are expected differences.
  Learning rates, selection metrics, training schedules and other differences
  remain visible. Missing defaults are not silently presumed equivalent.
- Runtime metadata, when available, and hashes of JSON inputs read.

Patient-set mismatch suppresses paired deltas. Config mismatches remain visibly
flagged as confounders. The output contains de-identified patient IDs from the
existing split records; keep it within the research workspace.

## Interpretation limits

Source validation evaluates known source tasks, **not** transferability to an
unseen endpoint. Comparable source AUC does not establish equivalent target
initializations. Declining validation after the selected epoch is compatible
with overfitting/selection noise, but does not explain why the selected Q/K
model differs from the baseline. Q/K was trainable in both stages, so existing
histories cannot causally isolate the stage responsible for the Test decline.
The report deliberately keeps `causal_stage=undetermined` and does not
automatically recommend a new architecture, threshold, epoch or seed.

Read the three printed comparison blocks first; there is no need to upload full
logs or large checkpoints. A new timestamped JSON report is written inside
project `outputs/`; existing files are never overwritten. NaN/Infinity constants
in old metric JSON become null/NA, not zero. Missing results are not five-fold
estimates. No model changes or training should be launched solely from the
direction of a post-hoc Test difference.

## Local verification

Tests exercise actual JSON/CLI behavior with hand-checked fixtures: correct
recorded epoch, exact patient pairing, config drift, missing source records,
duplicate runs, no modifications of input files, output overwrite protection,
and unestimable metrics remaining missing. No clinical result is inferred from
these synthetic fixtures.

Verification on 2026-10-09: full suite 536 passed, 3 skipped, 2 subtests passed
(283.95s); focused stage-audit plus existing Q/K launcher suite 11 passed,
including the additional unestimable-metric regression. Existing NumPy/pandas
deprecation warnings remain. Actual server histories were not available locally;
no clinical stage attribution or improved predictive performance is claimed.
