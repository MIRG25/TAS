# 源知识关系桥（Source Knowledge Bridge，SKB）

**2026-10-05 状态：停止，保留记录，不作为下一轮执行方案。**
用户完成的三方向×五折实验未证明全局收益：生存 C-index 均值约 0.6665，
低于 spatial 基线约 0.6851。下面保留原设计与历史命令供复核，不建议重跑。
当前新实验见 [source patient contrast](source_patient_contrast.md)，其中 SKB 关闭。

## 基线和问题：以本次用户指定的版本为准

本实验继承 `source_initialized_spatial_adapter.yaml`，不是纯2-block、
attention或head-warmstart版本。旧结果只读保留，不再自动补跑失败方案。
源模型复用 `runs/source_initialized_learning`。MAE不重训，缓存不重做。

用户记录的 spatial survival 五折 Test C-index 为
0.709734、0.712701、0.713789、0.644386、0.6447473937（最后一项来自随后补齐的Fold4）。
均值约0.6851。若前三折不变，后两折平均需要超过约0.6819才能使五折均值超过0.70；
后两折均为0.70时总均值约0.7072。不能保证前三折不下降，也不能拼接不同方法的有利折。
WHO和IDH历史记录只有四折，不能声称已有完整五折对照。

**代码可确认的限制**：source-shared路径复制共享Adapter，但禁用source_residuals时不读取
源任务private Adapter。目标标签更新副本后，也不再读取原始冻结的源共享特征。
这说明存在可测试的信息利用缺口，不证明它就是Fold3/4差的原因。
既有实验既有正迁移也有负迁移，不能假定所有源专家信息必然有益。

## 新模块做什么

记当前MRI裁剪的目标特征为z，冻结源共享特征为s，源私有残差为
d_j = private_j(s) - s。源记忆仅为本患者的 M=[s,d_1,...,d_S]，不存其他患者数据。

1. 独立的冻结源前向按它训练时的路径产生M，强制eval/no_grad，再恢复模块原模式。
   不把已发生变化的目标特征直接喂给冻结的源private Adapter。
2. q=Wq LN(z)，k_j=Wk LN(M_j)，v_j=Wv LN(M_j)。
   对q与k的点积分数及一个可学习的null分数做softmax，得到患者条件的源信息权重。
3. r=sum_j a_j v_j；null的value严格为零，可衰减整个源分支。
4. 校正量 Δz=Wup GELU(Wrel [r, q⊙r])，输出z+Δz给原预测头。
   这是关系映射，不是原始源残差的固定标量加和；所有映射无bias，r=0时校正严格为零。
5. Wup零初始化；新模块创建时隔离随机状态，因此与基线共享权重、初始预测及后续随机流一致。
   第一次反向可学习Wup，之后梯度流入q/k/v和关系映射。零初始化仅保证初始等价，不保证泛化不下降。

模块与标签形式无关：IDH/WHO二分类及离散生存头沿用原损失和指标。
路由在单患者内部进行，不用batch统计。推理只读MRI，不读源/目标真值标签或临床变量。
原MAE、共享Adapter、源私有Adapter及源头全部冻结；目标Adapter、两个私有block、目标头和桥接模块训练。
384维、64维瓶颈时新增106,497个参数，所有任务相同。增加一次无梯度源前向，有额外计算成本；
没有额外的Transformer block，未在本地验证3090实际吞吐或峰值显存。

## 不重跑的历史方案与实质差异

| 已放弃/本次不再试探 | 本次处理 |
|---|---|
| head-warmstart、全12-block、学习率试探 | 预热头0轮、仅2个私有block、学习率/日程不变 |
| attention pooling | 仍用mean，路由选择的是源任务信息，不是MRI空间token权重 |
| MAE bypass | 不读取原始MAE旁路，不把额外MAE向量拼到预测头 |
| token retention、EMA、L2-SP、裁剪概率试探 | 不开启这些约束，裁剪固定0.5 |
| 旧gated-source / source-scale搜索 | 旧门控关闭；不做0/0.5/1验证搜索；改为患者条件检索及关系映射 |
| spatial adapter | 用户指定的现有基线组成，本次保留；不重新把它作为待测试的新贡献 |

旧gated-source也利用过源private残差，因此“第一次利用源专家”不是正确创新表述。
SKB是关于**如何变换和选择**源信息的新实验假设，不是宣称源融合本身全新或已有提升证据。
知识组合和冻结旧任务思想已有先例：
[AdapterFusion](https://aclanthology.org/2021.eacl-main.39/)、
[Progressive Neural Networks](https://arxiv.org/abs/1606.04671)。
尚未做系统新颖性检索，不能据此声称已满足IEEE TMI创新要求。

## 运行：仅新模块，三个方向×五折

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m scripts.run_source_bridge_transfer --dry-run
python -u -m scripts.run_source_bridge_transfer --device cuda
```

默认100%标签；先验证、选best，再自动test，输出test_metrics及患者级预测。
启动器只复用已有source best.pt，任何源检查点缺失会在整批启动前停止，绝不偷偷重训。
任一已存在目标目录也会停止，避免覆盖。每个子进程前检查至少2GiB可用空间；
该检查不是15组总空间保证。目标只保存best.pt，不保存optimizer/last/周期快照，不能精确断点续训。
日志自动保存在新root的logs下。输出子目录以`_spatial3d_sourcebridge`结尾。
独立test_source_guided可从best.pt恢复桥接模块，不依赖额外源文件。

已完成部分任务后，用`--fold ... --target-task ...`显式选择剩余任务；不重跑成功结果。
如果有中断目录，使用新的--output-dir保留它，不自动从不明状态恢复。

## 开关与对照

`adaptation.target_source_bridge: false`或`adapt_source_guided --no-target-source-bridge`
可关闭模块；除省空间的保存方式外，恢复spatial基线计算路径。开关两组的初始主路径权重和随机流一致。
本次启动器**不自动运行**关闭组、fresh组或任何旧方案。历史对照先核对患者ID、source检查点、
有效配置、种子和预测，不因某折成绩较低而重跑。模块增加参数和源前向，关闭组并非等容量/等算力；
将来若主结果支持它，需再设计容量匹配及源信息去除对照，才能归因于知识迁移。

所有折/任务使用同一规则，分类按验证AUC、生存按验证C-index选模；测试只评估，不参与epoch选择。
已反复查看的测试折属于探索性开发证据。100%标签实验不能证明5%–20%的低标签优势；
低预算需在方案锁定后用相同标注预算（包含验证标签）、划分和训练条件另行检验。

## 验证

单元测试覆盖零初始化等价、随机流、患者条件路由、null拒绝、batch独立、源任务顺序不变性、
可训练参数与优化器精确匹配、源模型权重/预测不变和严格加载。
微型MRI缓存端到端测试覆盖三个目标任务、屏蔽源标签、自动测试、独立重测和关闭模块的旧路径兼容。
启动器测试覆盖15组映射、缺源即停、拒绝覆盖和阻止混入已放弃方案。
这些是实现验证，不是性能达标的证据。

2026-10-03本地完整回归：`python -m pytest -q --disable-warnings`，
264 passed、2 subtests passed（180.62秒）；既有依赖警告未清零。
另通过CPU bfloat16 autocast前向/反向有限值检查，未代替服务器CUDA训练验证。
