# Fold3/4: source-stage MAE spatial-token retention

## Scope

This is a single source-objective ablation, not another target readout or an
episodic meta-learning method. It tests whether source-supervised shared adapters
can retain useful generic MRI responses while learning clinical tasks. It does
not establish that the original adapters destroyed information and cannot
guarantee a C-index above 0.70. Frozen MAE features may themselves lack useful
target information; excessive retention could limit beneficial source learning.

Keep the source heads/private adapters, supervision, orthogonality and PCGrad.
With the same four-channel MRI crop x:

```
student = final normalized spatial tokens of MAE + shared source adapters
teacher = final normalized spatial tokens of frozen MAE alone (eval, no_grad)
L_retain = mean_image,spatial_token[1 - cosine(student, stopgrad(teacher))]
L_total = L_source + orthogonality_weight * L_orth + 0.1 * L_retain
```

Corresponding tokens are compared before pooling and before private adapters.
CLS is excluded. FP32 cosine/reduction is used under AMP. The task predictions
and regularizer reuse one student forward; a second frozen encoder forward uses
the SAME crop, no new augmentation or image loader. No new encoder parameters
are allocated. Teacher evaluation restores all encoder submodule modes before
checkpointed student backward. PCGrad replaces only source-task gradients;
orthogonality and retention gradients are kept. The loss works at batch size
one/two, without batch covariance estimates or negative examples.

Source training consumes only source labels; held-out labels are not used in
this term or source checkpoint selection. Existing source task parameters are
frozen when the target task is added. MRI inputs, labels, patient protocol,
metrics, class weights, optimizers and epoch selection rules are unchanged.
The target still copies source-shared adapters, trains two private blocks and
attention pooling, with no explicit source residual gates and no MAE bypass.
Source training requires an additional frozen forward; target inference cost
and trainable parameter count are unchanged from attention-only.

## Configuration and provenance

`configs/foundation/source_initialized_token_retention.yaml` inherits
`source_initialized_attention_pool.yaml`, including crop guidance probability
0.5, 100% target label budget, and disabled source/target full-budget refit.

- `source_training.mae_token_retention_weight: 0.1` is fixed for this experiment.
- Missing/zero weight preserves the original source training path/RNG/cost.
- Negative, non-finite and boolean weights fail before training.
- A nonzero weight requires RETRAINING source adapters from the existing MAE.
  Do not pass an old source checkpoint while merely changing this YAML.
- Checkpoints/effective configs store the weight; adaptation checks explicit
  requested weight against the source checkpoint, and records it in provenance.
- Source loss logs/history include `train_mae_token_retention_loss` and
  `train_weighted_mae_token_retention_loss`; these are separate from task losses.
- At initial identity adapters the retention loss can be approximately zero.
  It need not decrease monotonically because source supervision also acts.

No MRI recaching or MAE pretraining is needed. Old runs/checkpoints are retained.
New output roots:

```
runs/source_initialized_learning_token_retention/
runs/source_initialized_adaptation_token_retention/
```

## Upload and run only Fold3/4

Upload `outputs/source_token_retention_update_20261001.tar.gz` from the local
project into the SERVER project root. The archive contains code/config/tests/docs
only. It does not contain MRI data or checkpoints. Then run from the server:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf source_token_retention_update_20261001.tar.gz || exit 1

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -o pipefail
mkdir -p runs/source_initialized_learning_token_retention/logs
mkdir -p runs/source_initialized_adaptation_token_retention/logs

python -c "from glioma_meta.utils import load_config; from glioma_meta.models import SourceGuidedFoundationModel as M; c=load_config('configs/foundation/source_initialized_token_retention.yaml'); assert hasattr(M,'source_components_with_retention'); assert c['source_training']['mae_token_retention_weight']==0.1; assert c['adaptation']['target_pooling']=='attention'; assert c['adaptation']['target_mae_bypass'] is False; print('retention=0.1, attention=True, bypass=False')" || exit 1

for fold in 3 4; do
  python -u -m scripts.train_source_guided \
    --config configs/foundation/source_initialized_token_retention.yaml \
    --folds artifacts/dataset_who_binary/folds.json \
    --fold "$fold" \
    --source-tasks idh who \
    --held-out-task survival \
    --pretrained-checkpoint runs/foundation_mae_base/best.pt \
    --device cuda 2>&1 | tee \
    "runs/source_initialized_learning_token_retention/logs/fold${fold}_idh-who.log" || break

  python -u -m scripts.adapt_source_guided \
    --config configs/foundation/source_initialized_token_retention.yaml \
    --folds artifacts/dataset_who_binary/folds.json \
    --fold "$fold" \
    --source-checkpoint "runs/source_initialized_learning_token_retention/fold${fold}_idh-who_holdout-survival/best.pt" \
    --target-task survival \
    --target-initialization source-shared \
    --label-fraction 1.0 \
    --label-seed 2028 \
    --device cuda 2>&1 | tee \
    "runs/source_initialized_adaptation_token_retention/logs/fold${fold}_survival_100.log" || break
done
```

Do not add `--validation-only`: after target training, best.pt selected by inner
validation is automatically evaluated, printing and saving `test_metrics.json`
and `test/survival/predictions.csv`, including IBS and log-rank outputs. Source
training prints source validation metrics, NOT an unseen-target test. Source
startup must print `mae_token_retention_weight=0.1`. Target startup must print
`target_pooling=attention target_mae_bypass=False`; source provenance must record
`mae_token_retention_weight: 0.1`. Gate values remain zero by design, because
transfer occurs by copying the shared adapters, not by source residual gates.

The effective label seed is base seed + fold (2031/2032). The 100% label budget
includes validation: fit/validation/test are 316/79/99 and 316/80/98. No refit is
performed. The source and target checkpoint files remain separate.

If a target run fails after source training finishes, run just the adaptation
command for that fold using its newly trained source checkpoint; do not retrain
or overwrite completed source results. Run directories refuse overwriting an
existing best.pt. Inspect partial outputs before selecting another output root.

## Interpretation

This adds a source feature-response constraint, unlike the earlier target
parameter L2 anchor or a frozen-feature bypass appended to the target head.
The comparison is to the fixed attention-only protocol, not whichever historical
version won a given outer fold. The same loss is available for all three source
task pairs. One survival experiment does not establish all-task/low-label gains.

Do not sweep the retention weight or select epochs based on these outer-test
metrics. Fold3/4 have repeatedly been inspected and serve as exploratory
development evidence. Report both folds and all metrics; do not splice improved
folds with older Fold0-2 results and call it a new five-fold mean. A confirmed
claim needs a locked protocol evaluated consistently across folds and, ideally,
an untouched external cohort. Log-rank p measures risk-group separation, not
paired superiority over another model.
