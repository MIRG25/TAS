# 源任务初始化迁移：当前实验入口

配置：`configs/foundation/source_initialized_transfer.yaml`。

本轮实现“共同患者划分 + 源共享Adapter初始化”的最小对照实验。不重新训练MAE，
不增加EMA、L2-SP、共享辅助监督或新的生存专用损失。旧门控代码与历史结果保留。

## 协议和初始化

- 先生成 `artifacts/dataset_who_binary/transfer_protocol.json`，再进行源任务训练。
  内层划分只使用患者ID和固定随机种子，不使用未见目标任务标签。
- 同一外层fold的所有任务方向使用同一内层患者池。fold0–3为316例源拟合、79例共同验证、
  99例外层测试；fold4为316例拟合、80例共同验证、98例外层测试。
- 源任务只能在拟合池更新参数；共同验证池可用源任务指标选源checkpoint，但不参与参数更新。
- 目标拟合与验证分别从这两个患者池抽样；标签预算包含二者。20%预算为79例：63例拟合、
  16例验证。其余验证池患者的目标标签不参与目标模型选择。
- 目标预算是任务分层抽样的模拟实验；不同标签比例的集合目前不保证嵌套。
- 新协议禁用源/目标全预算refit，最终测试验证选中的原checkpoint。患者清单、协议SHA-256、
  标签定义与随机种子会保存；旧refit checkpoint和缺少协议记录的旧best.pt会被拒绝。
- WHO明确为0=2/3级、1=4级，旧三分类manifest不能用于本实验。

`--target-initialization source-shared` 复制源共享Adapter权重到独立目标Adapter，
只更新副本及新任务模块，冻结旧任务。`fresh`是同容量MAE直接适配对照，不复制源Adapter。
两组均关闭源残差注入、源专家门控和源强度搜索，避免重复叠加共享信息；保留两个从MAE复制的
目标私有Transformer block。预测头初始化、训练随机序列、优化器、标签预算与划分保持一致。

fresh仍传同一源checkpoint，以固定冻结MAE权重和模块构造顺序，但目标前向不使用任何源Adapter、
私有专家或源任务头。源训练始终冻结MAE，因此这组不通过目标路径使用临床源监督。
这是初始化收益对照，并不等价于完整的多任务联合训练对照，也不是已验证的性能提升。
新方案日志中的门控值和source scale为0是正常的：门控已禁用，源知识通过Adapter初始化传递，
不能把这些零值解释为“没有使用源知识”。

## 上传

同步更新的 `glioma_meta/`、`scripts/`、`configs/`、`tests/` 和协议JSON到服务器。
现有MRI缓存、MAE checkpoint无需重做。旧源模型必须按新协议重新训练；只关闭旧refit不够，
因为旧源/目标阶段的患者划分彼此独立。以下命令在服务器项目根目录执行。

## 1. 生成共同患者协议（一次）

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m scripts.prepare_transfer_protocol \
  --folds artifacts/dataset_who_binary/folds.json
```

相同输入重复运行安全；遇到不同的已有协议会拒绝覆盖，必须明确选择新的`--output`。

## 2. 重新训练源任务：先fold0，IDH+WHO → survival

```bash
mkdir -p runs/source_initialized_learning/logs
set -o pipefail
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.train_source_guided \
  --config configs/foundation/source_initialized_transfer.yaml \
  --folds artifacts/dataset_who_binary/folds.json \
  --fold 0 --source-tasks idh who --held-out-task survival \
  --pretrained-checkpoint runs/foundation_mae_base/best.pt \
  --device cuda 2>&1 | tee runs/source_initialized_learning/logs/fold0_idh-who.log
```

新源checkpoint为 `runs/source_initialized_learning/fold0_idh-who_holdout-survival/best.pt`。
这个阶段不优化目标任务，不评估外层测试集。

## 3. 顺序运行20%目标标签下的两组适配

```bash
mkdir -p runs/source_initialized_adaptation/logs
set -o pipefail
for initialization in source-shared fresh; do
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python -u -m scripts.adapt_source_guided \
    --config configs/foundation/source_initialized_transfer.yaml \
    --folds artifacts/dataset_who_binary/folds.json \
    --fold 0 \
    --source-checkpoint runs/source_initialized_learning/fold0_idh-who_holdout-survival/best.pt \
    --target-task survival --label-fraction 0.20 --label-seed 2028 \
    --target-initialization "$initialization" \
    --validation-only --device cuda 2>&1 | tee \
    "runs/source_initialized_adaptation/logs/fold0_survival_${initialization}_20_seed2028.log" || break
done
```

每组保存：`best.pt`、`last.pt`、带epoch/fold的阶段checkpoint、训练历史、有效配置、患者划分、
`validation_metrics.json`及`validation/<task>/predictions.csv`，生存任务还保存KM图。
`--validation-only`不创建测试DataLoader、不评估外层测试、不refit，结果没有`test_metrics`。
输出目录包括fold、任务、初始化方式、标签比例、实际label seed和validation-only标记。
新协议实验已有best.pt时拒绝覆盖；重跑请使用明确的新`--output-dir`。

## 4. 另外两个任务方向

复用同一配置和共同协议，分别更换：

| 源训练参数 | 适配目标 | 源checkpoint目录名 |
|---|---|---|
| `--source-tasks idh survival --held-out-task who` | `--target-task who` | `fold0_idh-survival_holdout-who` |
| `--source-tasks who survival --held-out-task idh` | `--target-task idh` | `fold0_who-survival_holdout-idh` |

每个方向先训练自己的源模型，再运行source-shared/fresh配对。更换fold时，源与目标的fold、
checkpoint目录必须一致；不能跨fold复用源checkpoint。

## 同患者协议、100%标签的MAE全量微调对照

配置：`configs/foundation/full_finetune_control.yaml`，继承当前source-initialized
配置中的数据、编码器结构、患者协议、分类权重、验证与评价设置。调用已有的
`scripts.train_foundation_transfer`入口，MAE编码器全部参与训练，新头只学习目标任务。
不使用源临床任务监督，不改变或覆盖已有源模型。参数量不匹配，目标结构也不同，
因此这是全量微调性能参照，而不是仅改变“解冻深度”的严格单变量消融。

固定编码器学习率1e-5、预测头1e-4，batch=2、200epoch上限、5epoch学习率warmup、
相同余弦调度与weight decay；按原验证规则早停。关闭refit、EMA、L2-SP、梯度累积、
额外head warmup和分层学习率衰减。100%标签中仍保留验证患者：fold0–3为316/79/99，
fold4为316/80/98（训练/验证/测试）。生存删失分布只由目标拟合患者估计。

上传新配置、`scripts/run_full_finetune_control.py`及对应测试文件。
在服务器项目根目录先做只读核对：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.run_full_finetune_control --task survival --dry-run
```

该入口对照已有source-shared各折的`effective_config.json`与`data_split.json`，
核对拟合/验证/测试患者、数据配置、编码器设置、验证规则、评价范围和优化超参数。
任一折不匹配会在启动GPU训练前退出，不会为通过检查而自动改动原实验。
通过后运行全部五折（每折训练结束自动测试）：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.run_full_finetune_control --task survival --device cuda
```

可加`--fold 0`只跑一折。可用`--task idh`或`--task who`，但需要先存在该目标方向、
相同协议的source-shared 100%实验记录，供配对审计。

输出目录：`runs/foundation_full_finetune_control/`。每折子目录为
`survival_fold0_full-finetune_labels_100pct_protocol_seed2028`，含`best.pt`、
`test_metrics.json`、患者预测和KM图；控制台日志自动保存至`logs/survival_fold0_full_100.log`。
保留原阶段checkpoint规则。已完成结果通过配置/患者核对后跳过；有未完成的非空输出目录
则停止，不自动覆盖或从某个任意checkpoint继续。

最终报告所有折，不按测试结果选择学习率、epoch或删除不利折。

## 判断规则与限制

先比较内部验证，不根据历史测试最高分选择方法。20%的目标验证集仅16例，分类少数类可能只有
2–4例。一次AUC差异不能证明收益；预先固定多个label seed和各fold，报告配对变化及不确定性，
不要挑选有利seed。不同label seed也不等于独立外部队列。

目前仍是固定内层holdout，尚未实现预算内重复交叉验证。5%验证人数更少，不适合广泛搜索超参数；
该阶段的交叉验证方案另行锁定后实施。上述规则不能保证达到性能目标。

## 5. 直接测试已选好的模型（不重训）

已完成`--validation-only`适配后，使用独立入口直接读取该实验的`best.pt`：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -u -m scripts.test_source_guided \
  --checkpoint runs/source_initialized_adaptation/fold0_idh-who_to_survival_private2_source-shared_labels_020pct_seed2028_validation_only/best.pt \
  --folds artifacts/dataset_who_binary/folds.json \
  --device cuda
```

模型结构、预处理、评价时间范围、目标任务、fold和标签种子均从checkpoint恢复，
不接受新的训练配置。直接使用保存的患者清单，不重抽标签、不重新选择epoch、不refit，
不需要另行读取MAE或源checkpoint。仅支持新患者隔离协议下的source-shared/fresh目标`best.pt`。
若文件搬家可指定`--protocol`或`--cache-dir`；协议内容哈希必须与checkpoint一致。

输出到原实验目录下的`test_only/`：

- `test_metrics.json`：验证选择记录、测试指标、种子、epoch和原checkpoint的SHA-256。
- `test/<task>/predictions.csv`：患者级概率/风险、生存时间和删失状态等。
- `test/survival/kaplan_meier.png`：生存任务的KM曲线。
- `data_split.json`、`effective_config.json`：患者划分和有效配置。

不另复制大体积`.pt`，保留原`best.pt`并核验其哈希和推理前后所有权重不变。
已有非空结果目录会拒绝覆盖；确需重测时显式指定新的`--output-dir`（精确目录，不自动加后缀）。
删失分布只使用目标拟合患者标签估计，不使用测试标签估计删失分布。
风险分组沿用现有评价实现，不在本轮改变评价定义。

一次性测试fold0的两种初始化、三个固定标签种子：

```bash
mkdir -p runs/source_initialized_adaptation/logs
set -o pipefail
for seed in 2028 2029 2030; do
  for initialization in source-shared fresh; do
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    python -u -m scripts.test_source_guided \
      --checkpoint "runs/source_initialized_adaptation/fold0_idh-who_to_survival_private2_${initialization}_labels_020pct_seed${seed}_validation_only/best.pt" \
      --folds artifacts/dataset_who_binary/folds.json \
      --device cuda 2>&1 | tee \
      "runs/source_initialized_adaptation/logs/test_fold0_survival_${initialization}_20_seed${seed}.log" || break 2
  done
done
```

方案固定后可按以上入口测试所有预先指定的模型，不挑最高分种子。
测试不得参与epoch、初始化方式或seed选择。被多次用于调参的历史外层结果应标记为探索性。

输入仍为四通道MRI；肿瘤mask只用于定位裁剪，不是输入通道。本轮裁剪策略没有变动，各组一致；
A3仍是独立消融，不与本轮同时改变。已有任务通过冻结源参数保留，不把任务真值作为网络输入。

验证命令：`python -m pytest tests -q`。测试覆盖实现和隔离约束，不代表已完成临床性能验证。
