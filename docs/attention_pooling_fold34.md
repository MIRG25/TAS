# Fold3/4：单项空间池化对照

## 修改范围

新增目标分支 `adaptation.target_pooling: attention`。将归一化后的空间 token
（不含 CLS）通过一个线性打分和 softmax 加权汇总，而不是固定平均。384 维特征仅增加
384 个参数；没有冗余偏置，权重全零初始化，不消耗额外随机数，因此开始训练前与原
平均池化在浮点容差内一致。打分、softmax 和求和在 FP32 下计算。

只使用目标拟合集的标签更新，IDH、WHO 和生存沿用各自原有损失与评价指标。
源分支仍使用原池化；源模型、源头部、MAE、患者划分均不变。目标池化使用现有头部的
学习率与权重衰减，不新增调参项。`mean` 仍是旧配置默认，旧检查点可以严格加载。

这是检验“固定空间平均限制目标读出”的单因素实验，不是已经证明的病因或性能提升。
不重训 MAE 或源任务，也不运行五折搜索。本轮只跑两个已有困难折，100% 标签预算包含
内层验证标签；不做全预算 refit，以目标验证选出的 best.pt 进行外层测试。

## 对照锁定

新配置：`configs/foundation/source_initialized_attention_pool.yaml`。
比较的是原始 `source_initialized_adaptation` source-shared 实验：裁剪引导概率 **0.5**。
该文件明确覆盖父配置后来改成的 1.0，避免混入 tumor-crop 或 adapter-LR 试验。
保持原学习率、batch、200 epoch 上限、30 次无改善早停，以及相同源 best.pt。
基础 label seed=2028，代码加 fold 后记录 fold3=2031、fold4=2032。

这些折已多次用于诊断，因此本轮结果属于开发实验；不将它们描述为未接触的确认性测试，
也不拼接不同模型各自最好的折来宣称新的五折均值。

## 上传

从本地项目同步：

- `glioma_meta/models/source_guided.py`
- `scripts/adapt_source_guided.py`
- `scripts/test_source_guided.py`
- `configs/foundation/source_initialized_attention_pool.yaml`

建议同时同步 `tests/` 和本说明。已有缓存、MAE 和源任务检查点无需重做。

## 服务器执行：只运行 fold3 和 fold4，自动输出 test_metrics

```bash
cd /root/autodl-tmp/project/glioma_meta_learning
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p runs/source_initialized_adaptation_attention_pool/logs
set -o pipefail

for fold in 3 4; do
  python -u -m scripts.adapt_source_guided \
    --config configs/foundation/source_initialized_attention_pool.yaml \
    --folds artifacts/dataset_who_binary/folds.json \
    --fold "$fold" \
    --source-checkpoint "runs/source_initialized_learning/fold${fold}_idh-who_holdout-survival/best.pt" \
    --target-task survival \
    --target-initialization source-shared \
    --label-fraction 1.0 \
    --label-seed 2028 \
    --device cuda 2>&1 | tee \
    "runs/source_initialized_adaptation_attention_pool/logs/fold${fold}_survival_attention_100.log" || break
done
```

不要加 `--validation-only`。训练结束自动加载最佳目标验证检查点，仅测试一次，打印
C-index、iAUC、1/3/5 年 AUC、IBS 和 log-rank 指标。每个运行子目录保存
`best.pt`、`last.pt`、`test_metrics.json`、训练历史、实际配置、患者清单，以及
`test/survival/predictions.csv` 和 KM 图。结果和检查点含 `target_pooling: attention`。
新输出根目录独立，子目录加 `_pool-attention`，不会覆盖原平均池化实验。

若需要同配置重跑 mean 对照，只加 `--target-pooling mean`，并指定另外的
`--output-dir`；不是本轮要求额外运行的实验。`--target-initialization fresh` 仍可用于
同容量的无源初始化对照，但本轮命令明确运行任务迁移。

## 本地验证

单元测试覆盖：均匀初始化/不改变头部或随机序列、真实优化器更新、源参数及预测不变、
批次和 token 排列不影响池化、AMP 稳定、旧检查点兼容与新权重严格往返。
三项小规模端到端测试覆盖自动测试、实际 CLI 配置落盘、独立 test-only 重载后预测一致、
错误池化元数据被拒绝。测试不代表 UCSF 的性能结果。
