# Source patient contrast: one source-objective change, unchanged spatial target

Date: 2026-10-05. Status: implementation verified on tiny synthetic MRI caches;
clinical performance has NOT been established. This is not a new MAE pretraining
run, not episodic meta-learning, and not a replacement for the existing source tasks.

## Question and motivation

Does a label-free patient-discrimination constraint during source-task learning
preserve MRI variation useful to unseen tasks better than source supervision alone?
The spatial baseline remains the comparator. The prior SKB experiment reduced
five-fold survival C-index from approximately 0.6851 to 0.6665. Its subgroup
analysis suggests that cross-IDH separation is not the principal bottleneck;
within-group ordering is weak. This is a hypothesis, NOT proof that representations
collapsed or that all clinically similar patients should be pushed apart.

## Implementation

The existing source supervised loss, PCGrad between supervised tasks, private
adapters, and shared/private orthogonality remain unchanged. Add

`L = L_source + 0.01 * L_orthogonality + 0.05 * L_patient_contrast`.

- The student uses the existing online MRI crop/augmentation and its shared
  feature before any task-private adapter. The second view keeps exactly the
  same geometry and all four modalities, with independent per-modality gain
  (0.9--1.1), Gaussian intensity offset (SD 0.05), and noise (SD 0.02) on
  normalized nonzero voxels. Exact-zero background stays zero. No extra crop,
  tumor annotation, clinical label, or task-head prediction enters this loss.
- A training-only LayerNorm/MLP projector produces 128-D normalized vectors.
  Its momentum counterpart reads a frozen-gradient copy of the shared source
  encoder/adapters. Only shared adapters and the projector are updated by EMA,
  momentum 0.99. MAE weights stay frozen. This teacher is NOT used for clinical
  validation/checkpoint selection and is NOT the previous EMA-selection scheme.
- InfoNCE temperature 0.2; a maximum 256-patient queue plus the current batch.
  Each candidate patient appears once. Current keys replace stale same-patient
  keys, and all same-patient negatives are masked. No validation/test images
  enter the teacher or queue. Patient IDs only prevent duplicate negatives;
  they are not learned embeddings or predictive model inputs.
- The auxiliary gradient goes to the shared adapters and projector, not private
  clinical adapters or heads. Supervised PCGrad retains the auxiliary gradient.
  AMP-overflow-skipped optimizer steps do not advance teacher or queue.
- The teacher, projector and queue are discarded at transfer/inference. Target
  adaptation is the original **source-shared + spatial adapter + 2 private
  blocks + mean pooling**, without changed target losses or training schedule.
  Existing source parameters remain frozen during target adaptation.

This is a MoCo-style application, not a claim of inventing contrastive learning:
[He et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html).
Patient negatives can be biologically similar, and an identity objective can
capture scanner/acquisition nuisance. Moderate intensity perturbations reduce
but do not eliminate that risk. Better clinical generalization is an experimental
hypothesis, not a guaranteed consequence of decreasing contrastive loss.

## Fixed protocol / exclusions

Use the existing `artifacts/dataset_who_binary/transfer_protocol.json`. Three
directions: IDH+WHO -> survival; IDH+survival -> WHO; WHO+survival -> IDH. WHO uses
the repository's existing binary endpoint (grades 2/3 vs 4), not new relabeling.
Source training accesses only source labels on fit patients; source validation
accesses source labels on reserved validation patients. The held-out task never
enters source losses, discretization fitting or the contrastive branch.

Target label fraction 1.0 includes fit AND validation labels in the budget.
There is no full-budget refit: target fit/validation/test are unchanged, source
and target early stopping remain based on their respective validation metrics.
The selected target is automatically tested once after training; predictions,
metrics, selected .pt and provenance are saved. No test-based branch/epoch choice.

All of the following remain OFF: SKB, head warmstart, attention pooling, MAE
bypass, source token retention, all-block adaptation, gated residual experts,
EMA clinical selection, and refit. No legacy failed experiment is auto-launched.
The crop probability remains 0.5, NOT the later tumor-crop ablation.

These five outer folds have already been repeatedly inspected. New results
are exploratory development evidence, not an untouched confirmatory estimate.
Do not select different modules for different outer test folds or promise that
Fold0--2 cannot decline. Low-label superiority and preserved existing-task
performance still need separate, matched evaluation after this phase.

## Server execution

Upload the update archive to the project root, then:

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf source_patient_contrast_update_20261005.tar.gz || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m scripts.run_patient_contrast_transfer --dry-run && \
python -u -m scripts.run_patient_contrast_transfer --device cuda
```

Default: 15 NEW source trainings + 15 target adaptations, sequentially, using
the existing MAE checkpoint. One extra no-gradient encoder forward per source
batch; target compute is unchanged. Source/target use **best-only** predictive
checkpoints, not periodic/last/optimizer snapshots. These are not resume files.
No cache/checkpoint/result is deleted. New roots:

- `runs/source_patient_contrast_learning`
- `runs/source_initialized_adaptation_patient_contrast`

`--fold 3 4 --target-task survival` restricts jobs if explicitly desired. No
restriction is used by default. Existing target directories are refused, even
if incomplete; use a new output directory or explicitly select remaining jobs.
Only completed, matching new source runs may be reused; legacy source checkpoints
are rejected. A source crash requires a clean separate source output root, not
silently treating its best-so-far checkpoint as a finished source experiment.

After (or during) training, read completed Test results without retraining:

```bash
python -m scripts.report_patient_contrast_results
```

This reports all three tasks, completion counts and per-fold Test metrics;
missing results are not zero, incomplete means are marked, p-values not averaged.
No CPU/GPU unit test proves clinical efficacy. Compare complete matched five-fold
results to the spatial baseline; do not cherry-pick the better method per fold.

## Implementation validation

2026-10-05: full local regression, 284 passed and 2 subtests passed (185.36 s).
Existing NumPy/pandas deprecation warnings remain. The new 20-test subset also
passed after adding selected-checkpoint reload checks. CPU bfloat16 autocast
forward/backward returned finite losses and gradients. These checks cover queue
deduplication/self-negative masking, fit-only access, auxiliary gradient routing,
held-out target-label removal, three-task automatic Test, source immutability,
legacy default-off compatibility, and full-grid planning. No full-size CUDA
throughput, 3090 peak memory, or clinical-performance claim follows from them.
