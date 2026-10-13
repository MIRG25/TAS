# Completed seed repeat: next check uses existing predictions only

The user supplied all 15 repeat results. Survival C-index was
0.6779 +/- 0.0274 at seed 2026 and 0.6593 +/- 0.0495 at seed 2027.
WHO AUC was 0.9417 vs 0.9306; IDH AUC was 0.8763 vs 0.8814, while IDH ACC
fell from 0.8847 to 0.8523. These are fold summaries, not confidence intervals.
Two seeds do not establish general stability or a unique cause.

Survival folds 3/4 remain weak in both runs. IDH fold 1 changes substantially
but fold 3 remains near 0.81 despite validation AUC near 0.98. Do not start
another seed search, choose seed winners by outer Test, or revive failed
modules based on these observations. The source checkpoints were fixed,
so source-training randomness is not tested by this experiment.

## Bounded next question

Do the two runs make errors on the same patients and ordering pairs, or is
the aggregate low score hiding different error sets? Similar C-index/AUC
values alone cannot answer that question. The new script reads JSON/CSV
only: no MRI loading, checkpoint loading, GPU, optimization, inference,
new Test, refit, threshold calibration, ensemble, or deletion.

It checks all 15 paired identities, configs (apart from training seed),
patient splits, unchanged labels, and consistency of existing CSV scores
with saved Test C-index/AUC/ACC. It uses already recorded source-weight
provenance, not a new tensor audit; the existing report_voxel_seed_repeat
performs the stronger checkpoint check.

For survival, a comparable pair is an observed death versus a later
observed time. Equal observed times are excluded, matching current metrics.
For classification, pairs are all positive vs negative patients. Positive
means IDH mutant or WHO grade IV. Score ties receive half-credit in primary
metrics and are reported separately, not counted as strict reversals.

- `Score-rank-r`: Spearman correlation of the two existing patient score vectors.
- `Both-wrong-pair-rate`: strictly wrong in both runs / all comparable pairs.
- `Wrong-pair-Jaccard`: shared strictly wrong pairs / union of wrong pairs.
- `Old-wrong-persist-rate`: shared wrong pairs / wrong pairs in seed 2026.
- Classification table: shared wrong patients, shared FN/FP, and decisions
  that changed in either direction under the original argmax rule.

High overlap is evidence of repeated errors in these runs, not proof of source
bias, label errors, cohort shift, or a particular architectural defect. Pairs
share patients and cannot be treated as independent observations for significance
tests. Low overlap likewise does not isolate initialization from augmentation,
optimization or environment changes. No difficulty thresholds or cohort exclusions
are introduced. All analyses of these repeatedly inspected tests are exploratory.

## Server command

Upload `outputs/seed_prediction_consistency_audit_20261008.tar.gz` into the
server project's `outputs/`, then:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning &&
tar -xzf outputs/seed_prediction_consistency_audit_20261008.tar.gz &&
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.audit_seed_prediction_consistency
```

Default inputs are `runs/source_initialized_adaptation_voxel_tokenizer` and
`runs/source_initialized_adaptation_voxel_seed2027`. The script prints to the
terminal and writes no files. It refuses missing pairs, duplicate results,
changed thresholds, mismatched patient IDs, truth, configs, or saved metrics.

This is diagnostic code only; no training module, config or result is altered.

## Local verification

`python -m pytest tests/test_seed_prediction_consistency.py tests/test_metrics.py -q`:
**21 passed (17.44 s)**. Tests cover tied scores, tied survival times, censoring,
constant scores, shuffled patient order, decision-vs-ranking distinctions,
wrong labels/IDs/configs/seeds/metrics, missing comparisons, and an all-15-pair
synthetic CLI invocation with byte hashes proving input files remain unchanged.
The CLI help also executed successfully. Real server prediction files were not
available locally, so no patient-level clinical conclusion is claimed yet.
