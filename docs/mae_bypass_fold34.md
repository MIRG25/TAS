# Fold3/4: same-crop frozen MAE feature bypass

## Scope and hypothesis

Keep the current source-shared initialization, target-private two blocks and
attention pooling. Give the target head independent access to the SAME MRI
crop's raw MAE spatial-mean features. This tests possible complementarity;
it does not establish that source supervision destroyed information or promise
a C-index above 0.70. This is a task-general engineering experiment, not a
standalone novelty claim.

The output logits are `existing_head(adapted_features) + W_mae * raw_mae_features`.
This equals a linear head on concatenated features. `W_mae` starts at zero,
has no additional bias, and consumes no random initialization draws. Existing
target-head initialization and initial predictions are preserved. Softmax or
survival-hazard sigmoid is still applied downstream as before, not to each
branch separately. IDH, WHO and survival retain their original losses/metrics.

The already embedded immutable encoder is reused, not duplicated. The raw
branch uses no adapter, dropout or gradient; its spatial mean excludes CLS.
The new weight uses the existing target-head learning rate and weight decay.
For feature dimension 384 this adds 768 parameters for either binary task,
or 4,608 for the 12-bin survival head. It does add one no-grad encoder forward
per crop during training and inference; GPU wall time has not been benchmarked.

Source tasks, their heads and encoder remain unchanged. No source truth labels
or clinical variables are given to the target. This is NOT a fresh target:
its main branch still copies the source-shared adapters. No MAE/source retrain,
new crop, MRI cache change, threshold tuning or full-budget refit is required.

## Configuration and compatibility

`configs/foundation/source_initialized_mae_bypass.yaml` inherits the completed
attention experiment, including tumor-guided crop probability 0.5. Only
`adaptation.target_mae_bypass`, the experiment name and output root change.

Legacy configurations default to false and create no bypass state-dict keys.
The CLI supports `--target-mae-bypass` and `--no-target-mae-bypass` for explicit
overrides. Checkpoints/results record the actual flag, and enabled run names
end in `_pool-attention_mae-bypass`. Independent test-only reconstruction checks
flag consistency and strictly loads weights. Existing results are not overwritten.

Upload the update archive into the SERVER project root before extracting it.
It contains code/config/tests/docs only, never MRI caches or training outputs.

## Run only Fold3 and Fold4; automatic outer test

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf mae_bypass_update_20260930.tar.gz || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p runs/source_initialized_adaptation_mae_bypass/logs
set -o pipefail

for fold in 3 4; do
  python -u -m scripts.adapt_source_guided \
    --config configs/foundation/source_initialized_mae_bypass.yaml \
    --folds artifacts/dataset_who_binary/folds.json \
    --fold "$fold" \
    --source-checkpoint "runs/source_initialized_learning/fold${fold}_idh-who_holdout-survival/best.pt" \
    --target-task survival \
    --target-initialization source-shared \
    --label-fraction 1.0 \
    --label-seed 2028 \
    --device cuda 2>&1 | tee \
    "runs/source_initialized_adaptation_mae_bypass/logs/fold${fold}_survival_mae_bypass_100.log" || break
done
```

Do not add `--validation-only`. The startup line must contain
`target_pooling=attention target_mae_bypass=True`. For survival the expected
trainable parameter count is 4,458,252 (4,453,644 + 4,608).

Training uses fit patients; validation alone selects the epoch. Training then
loads best.pt and automatically prints/saves test_metrics and patient-level
predictions. Label seed 2028 becomes 2031/2032 after the fold offset. The 100%
label budget includes reserved validation patients: fit/validation/test are
316/79/99 for Fold3 and 316/80/98 for Fold4. Old tasks remain in the checkpoint.

Compare against the already completed attention runs (Test C-index 0.6594066842
and 0.6748195670), not a newly selected best historical configuration. Report
both folds and all metrics. These repeatedly inspected outer folds are
exploratory development evidence, not untouched confirmatory tests; do not
splice them with old Fold0-2 scores to claim a new five-fold mean. This 100%
experiment cannot establish few-label transfer or all-three-direction benefit.
