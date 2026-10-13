# Seed-repeat v2: preserve JSON configuration types

## Reproduced cause

The server's 15 preflight checks passed, followed by a first-child error in
`check_repeat()` before loader construction, optimization, or Test. The parent
used `load_json()` to compare the saved effective config, but the child used
`load_config()`, which previously always called `yaml.safe_load()`.

On the production voxel config, serializing to JSON and rereading as YAML
changes `training.weight_decay` and `adaptation.target_block_learning_rate`
from float `1e-05` to string `'1e-05'`. Exact comparison correctly rejects
these types. This was a configuration parser bug, not an MRI, model, label,
or hyperparameter problem. Earlier tiny fixtures used 0.001 and lacked the
small scientific-notation values, so they missed the production failure.

## Fix and boundaries

- `glioma_meta/utils.py`: parse `.json` (case-insensitive suffix) with
  `json.load`; retain the existing YAML parser and inheritance behavior.
- Repeat launcher preflight uses the same `load_config` as child training.
- Config mismatches list differing paths, values and types; no coercion,
  ignored keys, relaxed pairing, or bypass flag is introduced.
- No network, optimizer, loss, seed, learning-rate, validation/selection,
  protocol, label budget, or original output is changed.
- v2 contains the previous seed-repeat implementation as well as this fix;
  deploy it over v1. No historical result or checkpoint is deleted/rewritten.

## Verification

Before the fix, the production-config JSON roundtrip failed and a real tiny
survival repeat reproduced the exact server exception at `adapt_source_guided.py`
line 694. After the fix, 25 targeted tests passed, including:

- scientific numbers, quoted numeric strings, nested lists, booleans/null;
- production voxel-config JSON roundtrip;
- YAML/JSON inheritance and cycle rejection;
- IDH, WHO and survival tiny-cache repeats containing learning rate 1e-5;
- deterministic replay, identical patient sets and frozen source tensors;
- actual launcher -> separate Python process -> saved JSON -> automatic Test
  -> paired completion verification (CPU tiny survival case);
- mismatch diagnostics that still reject strings substituted for numbers.

Full regression after the fix: **127 passed, 1 skipped (58.48 s)**. It covered
config loading, seed repeats, voxel launcher/integration, source-initialized
transfer, crop sensitivity, SAM and covariance compatibility. The skip is the
pre-existing opt-in full-resolution crop-audit CUDA test. Existing pandas/NumPy
deprecation warnings remain; no dependency installation/upgrade was needed.

No production MRI cohort or remote GPU training was run locally. This patch
fixes execution only and does not claim an improvement in clinical metrics.

## Retry on server

Upload the local `outputs/voxel_training_seed_repeat_update_20261008_v2.tar.gz`
to the server project's `outputs/`, then:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning &&
tar -xzf outputs/voxel_training_seed_repeat_update_20261008_v2.tar.gz &&
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.run_voxel_seed_repeat --device cuda
```

The reported failure only produced a launcher log, before target artifacts;
retrying does not require deleting runs or starting another seed. All 15 jobs
remain planned, with automatic final Test and the existing paired report:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.report_voxel_seed_repeat
```
