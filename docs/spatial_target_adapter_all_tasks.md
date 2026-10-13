# 3D 空间 Adapter：五折、三个未见任务方向

## 本轮配置

在原始 `source_initialized_adaptation` 的 source-shared 方案上，仅给目标 Adapter 增加零初始化的三维深度卷积残差。复制源共享 Adapter 的原权重，沿用目标私有两层 Transformer、mean pooling 和原任务头。三个任务使用同一空间模块，不针对生存单独设计网络。

| 源任务 | 未见目标 | 执行折 |
|---|---|---|
| IDH + WHO | survival | 0–4 |
| IDH + survival | WHO | 0–4 |
| WHO + survival | IDH | 0–4 |

先用 100% 标签预算（包含拟合和内部验证标签），保持原患者协议、不做全预算 refit，由内部验证选择 checkpoint。WHO 沿用 `dataset_who_binary` 的既有定义（2/3 合并对 4），不改变标签定义。

原 Fold0 C-index=0.7038、iAUC=0.7753、IBS=0.2484 是比较起点，不是本次已达到的结果；原五折 C-index=0.6861±0.0336。新配置不沿用后续 attention、MAE bypass、token retention、crop=1 或训练策略修改，裁剪引导概率恢复为 0.5。

## 上传后执行

将 `outputs/spatial_adapter_update_20261001.tar.gz` 上传到服务器项目根目录，然后：

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf spatial_adapter_update_20261001.tar.gz || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
bash scripts/run_spatial_transfer_all.sh
```

按 fold0→fold4、每折 survival→WHO→IDH 顺序运行，共 15 个目标适配实验。每次结束自动打印和保存 `test_metrics`，无需额外调用测试，不能加 `--validation-only`。

- 复用 `runs/source_initialized_learning/fold{fold}_{source-pair}_holdout-{target}/best.pt`。
- 缺少对应源 checkpoint 时先按原源任务目标补训，再适配；这会增加运行时间，但不重训 MAE。源训练不创建空间模块，不使用未见任务标签。
- 源 checkpoint 必须匹配 fold/任务/患者协议，源裁剪概率为 0.5、token retention 为 0。不合格就停止，不静默替换。
- 遇到训练错误、已有目标目录或缺少 best.pt 的残留源目录立即停止，不删除或覆盖。完整旧目标结果也不自动跳过，防止混入不同配置。
- 所有折 CLI 均传基础 `--label-seed 2028`，记录的实际种子为 2028+fold，不要重复加 fold。

## 结果

结果根目录为 `runs/source_initialized_adaptation_spatial_adapter`，例如：

```text
fold0_idh-who_to_survival_private2_source-shared_labels_100pct_seed2028_spatial3d/
  best.pt
  last.pt
  checkpoints/epoch_XXXX_fold0.pt
  effective_config.json
  data_split.json
  training_history.json
  test_metrics.json
  test/survival/predictions.csv
```

新字段 `target_spatial_adapter: true` 贯穿 checkpoint、配置、结果和预测元数据；`source_parameters_unchanged: true` 检查旧任务参数未变。当前 survival 可训练参数由 4,453,260 增至 4,484,364，三个任务均增加 31,104 个空间参数。`gate_mean=0` 仍正常，因为本方案靠复制共享权重迁移，不开启源残差门控。

分类仍输出 ACC、AUC、F1、敏感性、特异性；生存仍输出 C-index、iAUC、1/3/5 年 AUC、IBS、log-rank χ²/p。预测及曲线输出、指标定义与删失参考患者保持原逻辑。不可估计的时点可能为 null，不能改成 0。

中断后保留已完成结果，对未启动场景使用普通 `scripts.adapt_source_guided` 命令，指定本 YAML、对应源 best.pt、fold、target-task、label-fraction=1.0、label-seed=2028 即可；参见启动脚本中的完整模板。残留的不完整目标目录需人工检查并指定新的 `--output-dir`，不要删除历史结果来强行重跑。

## 判断边界

五折已经在开发中多次查看，统一重跑不会使其重新成为独立确认性测试。所有方向/折均用锁定配置完整报告，不按 Test 结果挑 checkpoint、调某折参数或拼接跨版本最佳折。100% 结果不能证明低标签能力，后续仍需匹配的 fresh 对照、低标签预算及独立评估。

零初始化保证兼容的起点，不保证训练后提升。本地合成数据的单元/集成测试不是 UCSF 临床性能实验；新增局部交互只是待验证假设，不把 3D 卷积本身宣称为已证实的论文创新。
