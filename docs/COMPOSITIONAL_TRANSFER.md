# Compositional source transfer v1

This is an experimental replacement for the source shared-adapter representation, not a seed change, attention pooling, SKB, SAM, MAE bypass or token-retention retry. Performance improvements are not established yet. The seed-2026 voxel reference is unchanged.

## What changes

At each frozen MAE block, split the width-96 adapter into four rank-24 components. Each source task learns four softmax coefficients per layer. All source tasks update the same component bank; PCGrad applies to shared-bank parameters, while task-specific coefficients receive their own task gradients. The existing source-private feature adapters and heads remain.

For hidden components h_k and up-projection partitions U_k:

`adapter(x, task) = x + sum_k 4 * softmax(a_task)_k * U_k GELU(h_k(x)) + bias`

Uniform coefficients reproduce an ordinary width-96 residual adapter; the initial up projection is zero. Source coefficients receive a deterministic 0.01 offset for different task indices to avoid perfectly identical initial routes. The components are not assigned clinical meanings.

For an unseen target, copy and freeze the bank. Initialize target coefficients from the arithmetic mean of source probabilities, then take their logarithm (not mean logits). Optimize target coefficients, zero-initialized spatial mixing, the existing voxel stem, two private MAE blocks, and a new head. The source coefficients, source-private adapters, source heads and original MAE stay frozen. The target's copied bank also stays frozen. Existing source predictions use their original task route; no clinical covariates or other-task ground-truth labels enter target inference.

This differs from the original voxel baseline in both source representation and target trainable adapter capacity. A positive result would support this combined hypothesis, not isolate a single causal mechanism. It is inspired by learning reusable task-adaptation structure, not an implementation or claimed reproduction of TAFL architecture search. It is not yet a proven novel contribution.

## Fixed protocol

- MAE checkpoint reused; NEW compositional source training is required. Legacy source checkpoints are rejected.
- Three directions, five outer folds, training seed 2026, label seeds 2028 + fold, 100% label budget including reserved validation.
- Same MRI crop, losses, source selection and target validation selection as the voxel baseline. WHO uses the existing manifest definition.
- No fresh comparison now; no refit; automatic final Test after each target, all metrics and patient predictions saved.
- Classification: ACC, AUC, sensitivity, specificity, F1. Survival: C-index, iAUC, time AUCs, IBS, log-rank metrics (existing implementation).
- New directories: `runs/source_compositional_learning` and `runs/source_compositional_adaptation`.
- Only best.pt retained in each source/target run; logs, configs, histories and predictions retained. Thirty best checkpoints still require multiple GiB. Keep at least 10 GiB free for a full run; launcher also checks 2 GiB before each job. Actual requirements depend on architecture/checkpoint serialization.
- Source tasks now have separate routed trajectories through the frozen MAE: source training is more expensive than the single shared trajectory. Target inference has one trajectory, not four separate encoders.
- A completed job gets a hash receipt. Rerunning the launcher skips only verified completed jobs. Partial directories are NOT overwritten or silently treated as complete. This is not optimizer-state resume; best-only checkpoints have no optimizer state.

## Server commands (project root)

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u -m scripts.run_compositional_transfer --device cuda
python -m scripts.report_compositional_results
```

Default: 15 source trainings and 15 target adaptations, each with final Test; nothing is run locally by preparing this update. Optional `--dry-run` prints and audits without training/writing. `--fold 3 4 --target-task survival` explicitly selects remaining jobs, but do not combine different architectures' per-fold winners.

Metrics live under each target directory in `test_metrics.json`; predictions under `test/<task>/predictions.csv`. The JSON's `composition.target_probabilities` are the new coefficients. Legacy `gate_mean=0` remains expected: the old source residual gates are disabled and are not the compositional coefficients.

These outer tests have been inspected repeatedly during development; treat comparisons as exploratory. Final claims need a locked protocol, matched budgets/baselines and genuinely independent evaluation. No module can guarantee every fold exceeds a fixed score.

## Implementation verification (2026-10-09)

Full local regression: 491 passed, 3 skipped, 2 subtests passed. The 15 new compositional tests also pass from a clean extraction of the deployment archive. Coverage includes all three tiny-cache source/target/Test pipelines, target-label absence during source training, exact source-state preservation, frozen target banks, test-only reconstruction, completed-run integrity and behavior-code fingerprint changes. This is CPU/software verification, not evidence of improved clinical metrics or full-size GPU memory verification.
