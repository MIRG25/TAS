# 100% 标签实验：一致分层协议

## 目的与边界

旧协议的内层验证为不读取标签的患者随机划分，WHO 低级别验证病例只有 6–11 例。
本改动不改网络结构、损失、训练种子或选模指标，仅提供独立的新划分模式。
WHO 定义固定为 I/II=0、III/IV=1。平衡 WHO、IDH、survival_event 三个二元变量。

**Validation 和 Test 的患者必须不同；一样的是分配算法和比例约束，不是患者或预测成绩。**
保留原人数：Fold0–3 为 fit316/validation79/test99，Fold4 为316/80/98。
每位患者恰好出现在一次外层 Test；每折三个集合严格互斥。

外层五折及每折内层划分使用同一个整数分配器。各二元变量的计数必须为
所在母样本期望计数的向下/向上取整值；在这些硬约束下，最小化联合标签组合的计数偏差。
内层母样本只包含该折 outer-training，不读取该折 Test 的类别计数来匹配验证集。
不因联合类别稀少而退回忽略 WHO 的分层，也不复制、丢弃患者或搜索高分种子。
唯一划分种子2027；模型训练种子仍2026；预算采样种子仍2028+fold。

该协议读取全队列标签进行分配，明确标记 `full_cohort_labels_100pct_only`。
它只用于当前100%实验；低预算调用会报错，不能用于严格5%–20%标签效率主张。
模型梯度仍仅使用当前任务许可的训练标签，不会把目标标签作为源训练监督。
比例受整数约束无法完全相等，且并不保证年龄、扫描仪、随访时长等分布相同。
生存分层只平衡事件/删失状态，不保证生存时间分布一致。

新外层测试患者不同，因此与旧成绩只能作非配对探索性对照，不能把新旧同编号折作配对差值。
已反复用于开发的同一队列不会因重新划分恢复为独立确认性测试集。
既有结果和检查点不覆盖；源模型、目标模型必须重训，MAE及MRI缓存可复用。

## 服务器执行

```bash
cd /root/autodl-tmp/project/glioma_meta_learning
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m scripts.prepare_balanced_transfer_protocol
python -u -m scripts.run_cross_depth_transfer \
  --who-low-high --source-fit-weighting --stratified-splits \
  --fold-indices 0 1 2 3 4 --device cuda --dry-run
python -u -m scripts.run_cross_depth_transfer \
  --who-low-high --source-fit-weighting --stratified-splits \
  --fold-indices 0 1 2 3 4 --device cuda
```

这里保留上轮已实现的源阶段 `fit-patient-mean` 加权方式；目标损失不变。
如要使用旧源损失作匹配对照，可省略 `--source-fit-weighting`，入口会使用另一个独立目录。
不要据旧 Test 分数选择新划分或重新搜索 seed。

准备命令仅生成协议和审计表，不读取MRI。可重复执行；不同的已存在文件会被拒绝覆盖。
算法相同与固定种子不保证不同 SciPy/HiGHS 版本选择同一并列最优分配；锁定并复用首次生成的协议文件。
跨服务器直接复制同一 artifacts 子目录，不在开始训练后重新生成。
训练入口先检查全部指定任务的数据和配置；每个适配完成后自动输出 `test_metrics`。
只保存最佳检查点；完整且通过指纹校验的任务会跳过；未完成目录不会静默覆盖或假装支持断点续训。

## 输出

- 数据：`artifacts/dataset_who_low_high_stratified/`
- 类别审计：上述目录的 `split_distribution.csv`
- 源训练：`runs/source_learning_fit_weighting_stratified/`
- 目标适配：`runs/source_cross_depth_fit_weighting_stratified/`

```bash
python -m scripts.run_cross_depth_transfer \
  --who-low-high --source-fit-weighting --stratified-splits --report-only
```

所有3个迁移方向共享同一套患者划分。检查点绑定协议ID及标签指纹，旧源检查点不能混入。
缺失折不补零、不计为完成；只平均Test指标；IBS越低越好，log-rank p不平均。
更均衡的分层是评估设计改进，不是模型泛化提升的证据，也不承诺Test达到预定分数。
