# Compositional source-task transfer

Goal: test whether a reusable source-trained adapter basis improves unseen-task transfer, without modifying the seed-2026 voxel baseline.

Architecture: four width-24 components replace each width-96 shared bottleneck. Source tasks have layer-wise softmax coefficients. Target coefficients start at the mean source probabilities; copied component banks are frozen. Target voxel stem, spatial mixing, two private blocks and head remain trainable. No source truth enters target inference.

Tech: PyTorch, existing source/target training and patient-disjoint protocol. Design basis: outputs/tafl_reassessment_20261008/paper-card.md section 16. This is a hypothesis, not a validated improvement or a reproduction of TAFL NAS.

Constraints: seed 2026; unchanged labels, losses, splits and validation selection; no fresh/SAM/retention/contrast/bridge; automatic final test; best-only checkpoints; independent outputs. Never select per-fold winners. Previously inspected tests are exploratory.

Review focus: source routing is optimized; target banks and old predictions remain unchanged; strict checkpoint reconstruction; missing results are not zero; no target leakage during source training.

## Execution ledger

Repository is not a Git worktree. User explicitly requests in-place implementation; no repository initialization or large data copy. Changes are isolated behind an opt-in configuration, with a deployment archive and tests instead of commits.

1. RED/GREEN: component model, routing, freeze and legacy compatibility.
2. RED/GREEN: real tiny-cache source training, target adaptation and test-only reload for all three tasks.
3. RED/GREEN: fixed-profile runner and report; completed-run skipping with provenance checks, partial-run refusal.
4. Full regression, fresh independent review, deployment archive, server commands.

Task 1: component unit tests RED (module absent) -> GREEN 5/5. Source coefficient gradients, arithmetic-mean initialization, frozen target banks and exact old-source parameter/prediction preservation covered.

Task 2: integration RED (missing composition metadata) -> GREEN 4/4. All three targets trained from tiny real caches with their target labels absent during source training; automatic Test and test-only reload match. Legacy factories preserve state keys and RNG stream.

Task 3: launcher/report RED (module absent) -> GREEN 4/4. Thirty jobs by default, dry-run read-only, partial directory refusal, completed checkpoint integrity checks and legacy-result rejection covered.

Full regression before review fixes: 489 passed, 3 skipped, 2 subtests passed.

Final independent review: core model passes; two Important completion/provenance gaps, no Critical/Minor findings.

Final: fixed missing-prediction skip — test_missing_predictions_invalidates_completion RED (did not raise) -> GREEN; full suite 491 passed, 3 skipped, 2 subtests passed.
Final: fixed incomplete implementation fingerprint — test_runtime_change_invalidates_fingerprint RED (unchanged hash) -> GREEN; full suite 491 passed, 3 skipped, 2 subtests passed.
Final: Ruling: empirical performance improvement remains unjudged — implementation tests cannot establish clinical-task accuracy; the experiment is explicitly unvalidated, and the cost if the hypothesis is wrong is additional training compute without an improvement. No promised scores.

Deployment validation: archive extracted into a clean temporary directory; launcher --help works and all 15 compositional tests pass there. Python syntax and every archived file SHA256 verified. New YAML ships resolved, without overwriting old YAML files. No MRI, cache, checkpoint or historical run bundled.

Status: implementation and verification complete. No GPU/MRI-scale performance experiment performed locally. No deferred minor findings. No Git merge/cleanup applies; preserve in-place code and this ledger.
