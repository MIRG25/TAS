# Q/K Transfer Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task.

**Goal:** Add an isolated source-trained Q/K low-rank transfer experiment.
**Architecture:** Optional Q/K modules outside the frozen MAE, passed explicitly
to block execution. Source deltas enter shared optimization; target owns copies.
**Tech Stack:** PyTorch, existing YAML/CLI/protocol/pytest stack.
**Spec:** ../specs/2026-10-09-qk-transfer-design.md

## Global Constraints

Rank8, scale1, seed2026, 100% labels; default off; no altered loss/split/selection;
no refit or fresh sweep; separate source/target roots; automatic Test; best-only.

## Review Focus

- Attention eval fastpath must not silently bypass Q/K deltas.
- PCGrad and optimizer must both include source deltas.
- Copies must not alias or mutate the original source at target adaptation.
- Old sources must fail strict architecture validation rather than silently load.
- Completed runs must be bound to source/config/patient/artifact provenance.

## Tasks

1. [x] Tests then core: `tests/test_qk_transfer.py`, `models/qk_adapter.py`,
   `models/source_guided.py`. `QKLowRankAdapter.forward(block,tokens)` preserves
   residual/normalization/dropout; explicit optional per-block source/target paths.
   Verify math, gradients, source preservation, default RNG and strict config.
2. [x] Integrate factory/metadata/checkpoint guards and source AdamW/PCGrad plus
   target optimizer. Tests in `tests/test_qk_integration.py` run real tiny-cache
   source, target and test-only CLI for all three tasks with target labels absent
   during source training. Check saved source/target QK tensors actually updated.
3. [x] New fixed YAML, runner and report. Tests in `tests/test_qk_launcher.py`
   exercise plan counts, mismatches, receipt tampering and partial reporting.
4. [x] Run focused and whole pytest suite, GPU backward if available, independent
   code review, documentation, versioned deployment archive under outputs/.

## Execution ledger

- No Git repository (verified); work in user-requested project, preserve an
  archive of pre-edit files instead of inventing commits/worktrees.
- Approval gates waived by user's standing direct-implementation instruction.
- Interfaces: model extra kwargs come from the existing factory; both trainers
  and test-only use it. Existing composition metadata helper will forward Q/K
  metadata for non-composition models without changing legacy empty metadata.
- Task1 RED: five failures (missing rank constructor/module/factory support).
  GREEN: math/reference, RNG, independent source/target tests pass. Source
  prediction comparison uses no_grad on both sides to match inference kernels.
- Task2 RED: real PCGrad parameter-count checks and target optimizer membership
  failed. GREEN: 9 focused tests including 3 real tiny-cache CLI roundtrips pass.
  Source Q/K updates nonzero; target copies change; all original source tensors
  unchanged; automatic-Test and test-only metrics match. Existing NumPy/pandas
  deprecation warnings remain, not introduced by this module.
- Task3 RED: three missing launcher/report failures. GREEN: 9 new plus existing
  compositional launcher tests pass. Shared receipt logic is parameterized with
  metadata_key/version; legacy defaults unchanged, Q/K receipts kept separate.
- Initial full suite: 528 passed, 3 skipped, 2 subtests passed (246.53s).
- Independent review found partial profile validation and unauthenticated report
  aggregation. RED: three failures reproduced missing config hash, unlocked
  LR/selection and missing receipt verification. GREEN: 17 focused/legacy tests
  passed after enforcing fixed schedules and verified common run identities.
- Added positive partial-report acceptance and mixed-identity rejection test:
  launcher suite 5 passed. Added even-head eval/no_grad Q/K fastpath regression.
- Reviewer follow-up: no remaining Critical/Important findings. MRI/cache/mask
  content hashes remain out of scope; immutable-data assumption is now explicit
  in docs and launcher (manifest/protocol hashing is not data-content hashing).
- Synthetic CUDA check: RTX3070Ti Laptop, input [2,4,128,128,128], FP16, two
  optimizer steps; finite gradients, target Q/K updated, source Q/K unchanged.
  This is not a clinical performance or full-run resource benchmark.
- Final full suite after review fixes: 530 passed, 3 skipped, 2 subtests passed
  (245.37s). An additional positive report/mixed-identity case added after full
  collection passed in the five-test launcher suite. CLI --help imports pass.
- Deployment archive: outputs/source_qk_transfer_update_20261009.tar.gz; explicit
  code/config/test/documentation patch, no checkpoints, caches or patient data.
