# Episodic Source Meta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 五折源端一阶 MAML + PCGrad，复用原目标适配及经核验的 fresh。
**Architecture:** 独立 episodic helper 与源训练脚本；新 orchestrator 复用原对照校验。原目标入口只增加源 objective 校验/记录。
**Tech Stack:** Python 3.10、PyTorch、pandas、现有 MAE/adapters/runtime。
**Spec:** `docs/superpowers/specs/2026-10-12-episodic-source-meta-design.md`

## Global Constraints

- Fold0–4、三个方向、15 source + 15 target、0 MAE；fresh 不自动重训。
- WHO I/II vs III/IV；原 stratified 100% 协议；source fit-patient-mean、target 旧 CE、endpoint-v2。
- 3 inner SGD steps、LR .001、Support/Query 各16、微批2；MAE immutable。
- 原 joint 与 fixed-basis 结果只读；当前目录无 Git，不创建 Git/worktree、不虚构 commits。
- 用户已批准完整设计，并授权当前会话完成规划与实现；连续执行，不重复请求阶段确认。

## Review Focus

- 尾块、缺失标签与微批均值不能改变患者权重。
- fast reset 必须发生在 backward 完成后；不能污染 outer/冻结 encoder。
- 半完成/非有限 episode 不得更新 outer；PCGrad 要保留正则。
- fresh 复用必须证明相同目标前向状态；元源字段不能静默丢失。
- 所有请求项预检后才写入；输入输出篡改与未完成目录硬拒绝。

## Task 1: Episode engine

**Files:** create `glioma_meta/training/episodic_source.py`, `tests/test_episodic_source.py`.
**Interfaces:** `settings(config)`, `episode_plan(ids,seed,fold,epoch,size=32)`, `plan_digest(plan)`, `validate_labels(plan,frame,tasks)`, `FastLearner(model)`, `meta_step(model,fast,support,query,tasks,optimizer,config,discretizer,weights,amp=False)`, `validate_checkpoint(saved,config)`, `provenance(saved)`.

- [x] RED: deterministic patient coverage/tails; strict hyperparameter guards; actual tiny MRI FO gradient vs manual adapted Query reference; storage isolation/reset; frozen MAE and inactive task invariants; masked microbatch equivalence; NaN no outer updates; AMP CPU/CUDA cases.
- [x] GREEN: implement plan/FP32 accumulation, cloned light modules sharing frozen encoder, 3-step inner SGD, Query gradients, task-only PCGrad and independent regularizer accumulation.
- [x] Verify helper tests and ledger evidence.

## Task 2: Source entry and provenance

**Files:** create `scripts/train_episodic_source.py`, `tests/test_episodic_source_training.py`; modify `scripts/adapt_source_guided.py`, `scripts/test_source_guided.py`.
**Consumes:** Task1 interfaces; existing source model/payload/runtime/protocol helpers.
**Produces:** compatible source best.pt + source_training_complete.json + episodic_audit.json; target/test source objective metadata.

- [x] RED: real tiny MRI source training for three directions, patient audit/held-out exclusion, MAE bytes immutable, source validation selection; target adaptation and standalone Test preserve episodic provenance; legacy unchanged.
- [x] GREEN: new CLI; all epoch plans/label availability preflight; independent episode loaders; source best-only checkpoint/audit; validate and propagate objective in target/Test.
- [x] Verify affected source/target tests and ledger evidence.

## Task 3: Matched five-fold launcher

**Files:** create `scripts/run_episodic_cross_depth.py`, `tests/test_episodic_cross_depth.py`.
**Consumes:** Task2 source CLI/checkpoint and Task1 settings/provenance; existing fresh.prepare/verify/run_job and integrity checks.
**Produces:** dry-run plan, protected source/target completion receipts, three-arm paired reports, Validation-only gate.

- [x] RED: requested plan counts, read-only dry/report, existing matching fresh required, roots/incomplete/tampered rejected before jobs, real source→target runs and verified skip, fresh initial-state proof.
- [x] GREEN: immutable old inputs; independent roots; post-source MAE/checkpoint validation and fresh-path tensor equivalence; original target config except source provenance/output/name; target/Test receipts.
- [x] Verify CPU subprocess/integration and ledger evidence.

## Task 4: Review and delivery

**Files:** actual changed-file tar.gz and human instructions under outputs.

- [x] Appropriate related regression suite and CUDA AMP; help/AST/package member byte checks.
- [x] Fresh-context whole-change reviewer required by executing-plans/requesting-code-review; no implementer subagents.
- [x] Important/Critical fixes RED→GREEN; record rulings/minors and final commands.
- [x] Package only actual code/test changes; final Chinese upload/preflight/train/report commands.
