# Low-rank second-order residual readout: one controlled candidate

Status: implementation candidate, 2026-10-07. Clinical benefit is NOT established.
Reference: the completed **voxel tokenizer + spatial adapter**, without SAM.
No MAE/source retraining, fresh sweep, threshold search, full-budget refit,
attention pooling, source bridge, patient contrast or token-retention loss.

## What the diagnostic does and does not establish

The completed threshold audit changed sensitivity/specificity but did not
improve overall ACC or its fold-to-fold stability. It cannot improve AUC or
C-index. Close that diagnostic; do not rerun it or tune another threshold
objective against observed Test results.

The code feeds the target head a mean of normalized spatial tokens. The mean
does not explicitly encode their dispersion or feature co-occurrence. This is
a concrete representational limitation, **not proof** that it caused the poor
folds: sampling variation, source bias and target-label noise remain possible.
Covariance can expose additional information, but can also overfit.

## Definition (same in all three tasks)

Let X be the N x D normalized spatial token matrix for one MRI crop, excluding
CLS. The original output is m = mean(X). Introduce a bias-free D-to-r projection
W, with r=32 fixed for this experiment:

```
Z = X W;                  Zc = Z - mean(Z)
C = Zc.T Zc / N;          t = trace(C)
v = [svec(C / (t + 1e-6)), log(1+t)]
output = m + U v
```

`svec` uses the upper triangle and multiplies off-diagonal elements by sqrt(2).
Covariance captures within-patient feature co-occurrence; trace normalization
controls scale and the extra log-trace retains dispersion magnitude. This is
not a matrix-square-root implementation. Reductions/matrix products run in
FP32 under AMP. No batch covariance or running statistics are maintained.
N=1/constant tokens give a zero descriptor without division by zero.

U is initialized to zero. The initial function, old weights and random stream
therefore match the baseline. W receives task gradients once U starts moving.
This does **not** guarantee preservation of baseline performance after training.
The module reads the target-adapted tokens initialized from source-shared
adapters, not source ground-truth labels, private expert outputs or raw-MAE
bypass features. Frozen original source paths are not modified.

With D=384 and r=32: 529 descriptor coordinates; **215,424 additional trainable
parameters**. All source/target losses and heads remain unchanged. The module
is permutation invariant over spatial tokens; it does not recover information
outside the input crop or explicitly encode spatial distances.

Second-order pooling is established technology, not a claimed invention here.
Primary background checked on 2026-10-07 (CVF proceedings; the academic-search
MCP was unavailable, so primary publisher pages were consulted):

- Lin, RoyChowdhury and Maji, ICCV 2015,
  [Bilinear CNN Models for Fine-Grained Visual Recognition](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html).
- Li et al., CVPR 2018,
  [Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html).

Those papers motivate feature interactions; they do not demonstrate benefit
for UCSF-PDGM or our cross-clinical-task transfer setting. This implementation
is a residual low-rank covariance candidate, not a reproduction of either paper.

## Fixed experiment

- IDH + binary WHO -> survival; IDH + survival -> binary WHO;
  WHO + survival -> IDH. Original source `best.pt` from the same fold only.
- All Fold0-4, 100% target-label budget; no fresh experiment for this round.
- Same reserved fit/validation/Test patient protocol and label seeds 2028+fold.
- Same 128-cube crop, tumor-guidance probability 0.5, two private blocks,
  learning rates, weight decay, losses, class weights, selection and stopping.
- IDH/WHO retain argmax / fixed 0.5. Validation AUC selects classification
  checkpoints; validation C-index selects survival checkpoints. No new
  best-epoch floor, threshold calibration or refit.
- Train and select using fit/validation only; then automatically print/save
  `test_metrics` and patient predictions for the selected checkpoint.
- New directories end `_spatial3d_voxel_covr32`; save only `best.pt`, no
  optimizer-state or periodic weight copies. Existing runs are never overwritten.

The YAML inherits the completed voxel profile and changes only the readout
flag/rank and experiment name/output root. Launcher preflights all requested
source files, refuses incompatible/missing sources, and never starts source
training. Existing output directories fail closed; choose remaining jobs using
`--fold` / `--target-task`, not by overwriting previous runs.

## Server commands

Upload `source_covariance_readout_update_20261007.tar.gz` to the **server project
root** first. A local file/link is not a server upload. Stop other training
before replacing shared code. The archive contains code/docs, not MRI/weights.

```bash
cd /root/autodl-tmp/project/glioma_meta_learning
test -f source_covariance_readout_update_20261007.tar.gz && \
tar -xzf source_covariance_readout_update_20261007.tar.gz && \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.run_covariance_transfer --device cuda
```

Default: 15 target adaptations, source-shared only, each followed by Test.
No extra `tee` is necessary: the runner records each job under the new root's
`logs/`. It checks at least 2 GiB free before every job; budget more than that
for the full sweep (roughly 2-3 GiB for 15 best-only model files plus outputs).
It cannot resume an interrupted target optimizer; preserve any partial run
and use a separate output root if that target must be restarted.

Read completed results at any time, without new evaluation:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m scripts.report_covariance_results
```

Output: per-fold Test metrics, mean/sample SD, valid fold count, primary-metric
min/max/range, and paired differences against the fixed voxel reference when
source provenance, patient sets and effective configurations match. Missing
folds are not zero and partial summaries are not five-fold results. Log-rank p
values are reported per fold, never averaged.

## Interpretation fixed before running

Use the full per-task comparisons, not per-fold method winners. Improvement in
one endpoint is not a universal improvement. Compare classification AUC **and**
ACC/SEN/SPEC, survival C-index **and** iAUC/IBS, with fold SD and the weakest
folds shown. A smaller SD caused by worse strong folds is not sufficient.
The existing voxel reference is survival C-index 0.6779 +/- 0.0274, WHO AUC
0.9417 +/- 0.0306 / ACC 0.9069 +/- 0.0231, IDH AUC 0.8763 +/- 0.0640 /
ACC 0.8847 +/- 0.0431. These are user-reported developmental results.

Repeated inspection of these outer folds makes subsequent comparisons
exploratory. No claim of untouched-test confirmation or reliable low-label
superiority follows from this 100%-label ablation. Any retained method still
needs matched controls/label budgets and independent confirmation.
