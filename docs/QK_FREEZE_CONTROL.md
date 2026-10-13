# Fold0：保留源 Q/K，仅冻结目标 Q/K 增量

本实验不增加网络模块，不重训源任务，也不恢复已停止的五折 Q/K 扫描。
问题是：在相同的已训练源 Q/K 表征下，继续更新目标 Q/K 是否有利于迁移？

## 唯一训练变量

- 基线：已完成的 `runs/source_qk_adaptation`，目标 Q/K 可训练。
- 对照：复制同一个源检查点，设置 `adaptation.target_qk_trainable: false`。
- 冻结的是目标副本的 Q/K 低秩矩阵，**保留非零权重及其前向作用**。
- 目标 adapter、voxel tokenizer、spatial adapter、最后两个私有 block 和目标 head 仍训练。
- 因此这不是冻结全部注意力：私有 block 中的基础注意力权重仍可更新。
- 原有源参数始终不变；目标可训练参数减少 147456 个。
- seed=2026，Fold0，100%标签。配置来自旧 target `best.pt`，与旧 effective_config 完全核对；
  不从可能已修改的 YAML 继承链重新生成。除实验名称、输出位置、Q/K可训练性之外保持不变。

## 服务器运行

上传更新包至项目 `outputs`，在项目根目录执行：

```bash
tar -xzf outputs/qk_target_freeze_control_update_20261009.tar.gz &&
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -u -m scripts.run_qk_freeze_control --device cuda
```

默认依次执行 Fold0 的 IDH+WHO→survival、IDH+survival→WHO、WHO+survival→IDH。
只进行三次目标适配，源模型零次训练；每次训练后自动 Test、保存预测和 `best.pt`，
最终输出对旧“目标Q/K可训练”版本的配对差值。输出为 `runs/source_qk_target_frozen`。

只读计划（不写文件、不训练）：加 `--dry-run`。
只读汇总（不重复 Test）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.run_qk_freeze_control --report-only
```

如只剩部分任务未启动，用 `--target-task who idh` 等明确指定。已有输出目录拒绝覆盖，
不静默删除或重训；中断目录需先确认其完成情况。

## 核查与解释边界

启动前核查旧结果的完成记录、源及目标检查点、配置和预测文件哈希。
结束后核对患者划分、训练种子、源检查点及全部源权重，并逐张量核验目标 Q/K
确实与原源模型相等。源路径及原实验产物不被改写。
读报告会检查新产物哈希，不会重新选 epoch、训练或测试。

这仅是单折机制对照，不保证分数提高，不能证明五折有效，也不能独立确定源偏置的全部成因。
冻结前后容量不同是干预本身。比较依赖相同缓存与运行环境；缓存内容不在文件哈希核查范围内。
外层测试已被多轮查看，结果应标记探索性，不能按每折 Test 选择方法，也不能作为未经调参的最终证据。

更新包不含 MRI、缓存或训练权重。旧 voxel / Q/K 数据、YAML、结果均保留。
