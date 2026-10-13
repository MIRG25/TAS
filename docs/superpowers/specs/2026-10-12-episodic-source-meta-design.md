# Episodic 源端元学习实验设计

日期：2026-10-12。状态：用户已放弃当前版本，停止扩跑；代码和已有结果仅保留用于复核。

停止记录：已收到真实 Fold0 survival 结果，Validation C-index=0.594009、Test=0.596845；
对应旧 joint source-shared 为0.701721和0.630227。尚非完整五折/三个任务证据，不能据此
宣称所有 episodic 方法无效。当前实现沿用旧 epoch/LR 日程但每epoch外层更新数量更少，
属于需要复核的设计差异，不作为本轮继续调参的理由。用户已要求另想方法，不重跑本版本。

## 1. 已确认的目标与范围

用户同意停止 fixed-basis 冻结实验，不再补跑尚缺的四项，改尝试 episodic 元学习。
研究目标仍是从两个源临床任务学习初始化，再适配第三个未见临床任务；重点检查
survival C-index、IDH AUC/ACC，同时报告 WHO。上一轮停止的依据包括 source-shared
Validation survival C-index 均值下降、IDH 四折 Validation AUC 均值下降，而非仅凭 Test。

沿用用户选择的五折：Fold0–4，全量目标标签，三个方向：

- IDH + WHO → survival；
- IDH + survival → WHO；
- WHO + survival → IDH。

默认计划为 15 次新源训练 + 15 次新目标适配，0 次 MAE 训练。
符合第 7 节复用条件时，0 次新 fresh 训练；不得未经说明把次数增加到 45 或 60 项。

## 2. 方法选择及边界

采用一阶 MAML 的适配后 query 梯度，保留现有源共享梯度 PCGrad，记为
`episodic-fomaml-pcgrad-v1`。这是一阶近似与现有 PCGrad 的组合，不能称为标准二阶 MAML。

选择理由：MAML 直接优化经过少量更新后的泛化表现；一阶版本不保留支持集更新的
二阶计算图，适合当前 3D MRI 模型的计算约束。Reptile 的权重插值更简单，但本轮不
采用它，因为当前要直接检验独立患者 query 上的适配后损失。二阶 MAML 暂不纳入。

算法依据：[MAML](https://proceedings.mlr.press/v70/finn17a.html)、
[一阶元学习分析](https://arxiv.org/abs/1803.02999)。这些文献提供算法依据，不提供当前
胶质瘤 MRI 数据上的疗效或性能保证。

不沿用 fixed-basis 冻结规则，不开启 Q/K、桥接、组合专家、SAM、患者对比、token
retention、head warmstart 或 full-budget refit。MAE 始终冻结。

## 3. 患者协议与 episode

唯一患者协议仍为 `artifacts/dataset_who_low_high_stratified/transfer_protocol.json`，
folds/manifest 来自同一目录。494 例，fit 316 例、Validation 79/80 例、Test 99/98 例。
WHO 为 I/II=0、III/IV=1，IDH 标签及 endpoint-v2 生存结局保持不变。

Support 和 Query 均来自该折的 source fit，Query 是源训练患者，不是原 Validation。
源训练、episode 生成与源选模不得读取 held-out 临床任务标签。
原 Validation 仅按源任务指标选源 checkpoint；Test 不进入源阶段。

每个 meta epoch 的 episode 生成规则：

1. 独立随机数生成器以 `episode_seed = training_seed + 1000003 * fold + 1009 * epoch`
   打乱一次原有序 fit IDs；epoch 从 1 起。派生公式写入 provenance，不能用 Python
   随机 hash。MRI loader seed 为 `episode_seed + 10007 * episode_index`，index 从 0 起。
2. 按 32 人分块。小于 2 人的末块并入前一块，拒绝 fit 少于 2 人的输入。
3. 每块前半为 Support、后半为 Query；奇数块 Query 多 1 人。正常 316 例对应
   9 个 16/16 episode，加 1 个 14/14 episode。
4. 同一块的两个源任务使用相同患者角色。每个源任务都从同一个外层初始化独立出发，
   完成 Support 适配后，在本任务 Query 上求梯度；任务间不串联 fast weights。
5. 同一患者在该 epoch 只出现于一个 episode 的一个角色；下个 epoch 可变换角色。
   不按临床类别平衡采样，不新增患者排除规则，不重复采样少数类。

各源任务沿用现有标签 mask。若任一任务的某 episode Support 或 Query 全部无有效
标签，完整计划在训练前拒绝并给出任务和 episode，不允许以零损失静默冒充训练。
正常删失生存记录是有效标签，不要求每个微批次同时含事件和删失者。

每个 episode 保存有序 Support/Query IDs、任务、epoch、派生 seed；以规范 JSON
流式 SHA256 记录计划。写出患者覆盖、角色交集为零、有效标签数和可重建计划摘要。
使用独立 loader RNG，Validation 的调用不得扰动后续 episode 或 MRI 裁剪序列。

## 4. 内循环、外循环与模型状态

初始元参数与旧源训练的架构、模块数量完全相同：深层 shared adapters、两套源
private adapters 与各自 head。仅 shared adapters 被后续未见任务复制。

本轮固定超参数，不用 Test 或每折结果选择 inner steps/LR：

| 项目 | 设定 |
|---|---|
| inner steps | 3 |
| inner optimizer | SGD，momentum=0、weight_decay=0 |
| inner learning rate | shared/private/head 均为 0.001 |
| Support/Query 典型人数 | 各 16 |
| MRI 微批次 | 2，沿用现有配置 |
| inner gradient clipping | 沿用源 max_grad_norm |
| outer optimizer 与 LR | 沿用旧源 grouped AdamW 的分组、LR、weight_decay |
| source max meta epochs | 沿用实际旧源配置，预期上限 200 |
| outer schedule/early stopping | 沿用旧源 warmup、cosine、Validation interval/patience/min_delta |
| training / label seeds | 沿用 2026 与 2028+fold 的原规则 |

每步内循环对整个 Support 的任务损失求患者均值，用微批次梯度累积后只做一次 SGD
更新；3 步复用同一组已加载 Support MRI 张量，不在每步重新抽裁剪。最后不足微批
不能与完整微批等权平均。缺失标签按有效患者数归一，不改原任务损失定义。

fast 模型仅 shared adapters、本任务 private adapter 和本任务 head 可更新；另一个
源任务模块及 MAE 冻结。Support 不加入额外正则，Query 加原权重的本任务 shared/private
orthogonality。Query 任务梯度按任务有效患者数累积，正则梯度按全部 Query 患者数
累积，最后只做一次外层更新。正则为 `mean_i cosine(shared_i, private_delta_t,i)^2`，
沿用 eps=1e-6；不能把其他源任务的 private delta 计入本任务的 episode 正则。

对每个源任务 t，设初始参数为 θ，fast 参数为 φ_t：

    φ_t ← θ
    重复 3 步：φ_t ← φ_t − 0.001 × clipped_gradient(L_support,t)
    g_task,t ← ∂ L_query,t(φ_t) / ∂ φ_t
    g_reg,t  ← ∂ [0.01 × Orthogonality_query,t(φ_t)] / ∂ φ_t

这里 0.01 是当前预期值；执行以该折旧源 effective_config 的实际正则权重为准。
原源任务 loss_weights 经归一化后乘 Query 梯度。仅 shared 的任务损失梯度进入现有
PCGrad；正则梯度保留，不参与投影。任务 private/head 接收本任务加权 Query 与正则
梯度，未参与 episode 的模块不得收到 weight decay 更新。

g 从 fast 参数复制到 θ 的同名参数，忽略 ∂φ/∂θ 的二阶项。Support 更新仅作用于
fast 参数，Query backward 必须结束后才 reset fast 模型。全局 θ 只由外层 optimizer
更新；不得把 Support 末态当成 meta checkpoint，也不得把 Support 梯度误作外层梯度。

fast 模型可复制轻量 adapter/head 并共享同一只读 MAE，避免复制整个编码器。
不能在 activation checkpoint backward 尚未结束时切换权重；临时 functional_call
恢复参数后再重计算会存在使用错误权重的风险，若采用此路线必须有专门等价测试。

梯度累积、task/regularizer 分解、PCGrad 都用 FP32。保留 CUDA AMP，但缩放梯度必须
在 SGD 更新、PCGrad/裁剪前正确还原；任何非有限阶段要跳过整个外层更新并记录，
不得把半完成 episode 的梯度写入 meta 参数。epoch 无成功 outer update 时硬错误。

## 5. 任务损失与源 checkpoint 选择

分类 Support/Query 都沿用 source fit-patient-mean CE；权重仅由完整 source fit 计算
一次，沿用 IDH/WHO class-weight power。不得按 episode 重算类别权重。
生存 Support/Query 均沿用固定 endpoint-v2 cutpoints 的删失感知 NLL。

源 checkpoint 仍以原 source Validation 主指标平均值选择：IDH/WHO AUC、survival
C-index。评估未适配的 meta 初始化及其源头/源 private；Validation 不做内循环，
也不成为 Query 训练数据。这保留旧源选模流程，但未适配 Validation 与优化的
适配后 Query 目标并非完全相同，解释结果时须明确该限制。

保存全部来源字段：模型/encoder state、源任务和 held-out、fold、原患者协议、
selected epoch、源 Validation、class weights/manifest digest、训练 runtime、
episodic 设置、计划 digest、成功/跳过更新次数、训练计算计数。
checkpoint method 保持现有 source 模型家族标识，另加严格校验的
`source_training_objective=episodic-fomaml-pcgrad-v1`，不能借此伪装成旧联合训练完成项。

## 6. 目标适配保持原方案

从 meta checkpoint 的 shared adapters 初始化目标副本，完整训练 LayerNorm/down/up。
继续采用 spatial3d adapter、voxel tokenizer、两个 MAE 私有 Transformer blocks、
cross-depth [4,8,10] rank64。损失、LR、裁剪、标签预算、seed、选模和最终 Test 流程
全部沿用对应折旧 source-shared 的 effective_config。

源阶段的 private/head 不转移到目标；新目标 head 仍是原规则的随机初始化。
meta training 的源头结构与目标两块私有 Transformer/readout 并不相同；本轮优化
的是共享初始化的迁移性，不宣称完整模拟了真实 target adaptation 架构。

## 7. 对照、成本与成功判断

每折报告三组：旧 joint source-shared、新 episodic source-shared、旧 matched fresh。
fixed-basis 不纳入新方法的初始化或训练配置。

旧 fresh 复用前须证明：

- 同一不可变 MAE 的 tensor bytes/digest 完全一致；
- 原 source 架构、模块构造顺序、目标 seed 与新配置一致；
- 以旧 source 和新 meta source 分别重建 fresh 时，目标前向涉及的所有初始参数
  和 buffer 逐张量相同，forward 路径不读取 frozen source adapters/private/head；
- 按顺序的 fit/Validation/Test IDs、全部目标训练配置、runtime、label seed 一致；
- 旧 fresh 的 best.pt、history、split、metrics、完成标记均通过原完整性校验。

任一条件不成立，运行计划在新源训练前拒绝复用；不静默加跑 fresh。需要新 fresh
时另行明确实际新增工作量。新源训练开始前先验证可以预判的配置/架构/协议条件；
训练后验证新 MAE digest 及重建参数，随后才能运行新目标适配。

本设计沿用 source epoch 上限，因此计算量不等于旧联合训练。在正常完整标签场景，
每 32 人 episode 对两个源任务分别进行 3×16 Support 与 16 Query 前向，约 128
患者前向，旧联合训练对 32 人共享一次 encoder 前向，约为其 4 倍。
报告真实 patient forwards、Support steps、outer updates、wall time 与峰值 GPU memory；
不要宣称此对照严格隔离了算法与训练计算量，也不要宣称省计算或公平 compute budget。
若本轮出现收益，计算量匹配的 joint control 是后续归因验证，不在本轮偷偷增加。

目标 Validation 用于 checkpoint 选择；最终 Test 只评估一次并进行同折描述性比较。
已有 Test 已被多轮检查，因此新结果属于探索性证据。不得按 Test 选 inner 超参数、
方法、epoch、seed、阈值或单独选择每折版本；不平均 log-rank p。

全五折均完整时，分别打印 Validation 下新 meta − 旧 joint 和新 meta − fresh 的
逐折、平均差值。survival 主指标 C-index、分类主指标 AUC，IDH 同时报告 ACC/F1。
预设继续信号为 survival 和 IDH 的平均 Validation 主指标均高于旧 joint，WHO
平均 Validation AUC 不降低；另外分别报告是否超过 fresh，不能把优于 joint 当成
正迁移已成立。浮点容差 1e-12 不是实际意义或统计显著性阈值。
Test 不决定继续信号，不自动启动新的调参/seed 实验。

## 8. 代码边界与交付界面

计划新增独立的 episodic helper、源训练入口与实验 orchestrator；不重构旧 joint
训练循环。模型架构保持原样。现有目标入口只增加源 objective/provenance 的校验和
结果记录，旧 checkpoint 缺少 episodic 字段时仍按旧 objective 处理。

预期新文件边界：

- `glioma_meta/training/episodic_source.py`：计划、fast 状态隔离、一阶梯度、meta audit；
- `scripts/train_episodic_source.py`：源训练/Validation/选模、兼容 checkpoint；
- `scripts/run_episodic_cross_depth.py`：输入校验、15+15 调度、fresh 复用证明、配对报告；
- 对应的 CPU/CUDA 单元测试和 tiny MRI 全流程集成测试；
- 人类可读的上传与执行说明，仅打包实际改动文件。

新目录：

    runs/source_learning_episodic_fit_weighting_stratified/
    runs/episodic_cross_depth_fit_weighting_stratified/

旧 joint/fresh/fixed-basis 目录只读。部分目录拒绝覆盖；只有输入输出 hash、objective、
选模信息、患者列表及冻结 MAE 都核验通过的完成任务可以跳过。异常退出不创建完成
标记。dry-run 不写配置、不训练、不推理。report-only 不重新 Test。

实现后的命令如下；上传更新包后执行：

    python -u -m scripts.run_episodic_cross_depth --fold-indices 0 1 2 3 4 --device cuda --dry-run
    python -u -m scripts.run_episodic_cross_depth --fold-indices 0 1 2 3 4 --device cuda
    python -m scripts.run_episodic_cross_depth --fold-indices 0 1 2 3 4 --report-only

## 9. 必需验证与完成标准

- 解析梯度的小型可手算模型：FO Query 梯度等于 adapted 参数的梯度，并有别于联合训练
  或误用 Support 梯度；3 步确实执行，fast 与 outer 参数无 storage alias。
- 真实 MRI 模型：每个任务对应梯度覆盖正确，另一个源 private/head、MAE 不变；
  重置 fast 后预测和初始权重完全恢复；activation checkpoint 开/关等价。
- 患者计划：完整覆盖、每 episode/epoch 角色互斥、可重建 digest、尾块、mask、
  Validation/Test 排除、held-out 标签扰动不影响计划/训练/选模。
- loss：整个 Support/Query 与微批累积等价，尾微批/缺失标签正确归一，fit class weights
  固定；endpoint-v2 事件、删失和零信息生存微批不静默污染 meta 梯度。
- CPU 与 CUDA AMP：非有限梯度不产生部分 outer update，冻结 MAE 前后 bytes 一致，
  PCGrad 仅投影任务梯度、保留正则；episode 次序不串联 fast 状态。
- 完整 tiny 临床方向源训练 → meta checkpoint → 原目标适配 → 独立 Test 重载；
  源/目标匹配 fresh 初始参数证明；legacy joint/fresh 回归测试通过。
- orchestrator：全计划预检、部分目录拒绝、输入/输出篡改拒绝、verified skip、只读
  报告、不同 runtime/协议/初始化拒绝；源码中无 Test 控制训练决策路径。

测试通过只证明代码行为。真实 cohort 训练由服务器执行；不给性能提升保证。

## 10. 设计自审

已明确：方法名称与一阶梯度定义、患者角色、最后不足块/微批、标签掩码、源与目标
权重归属、源/目标选模、fresh 复用条件、计算量限制、五折任务次数、旧结果保护。
未实施任何新训练，也没有修改临床协议或把 fixed-basis 的未完成项作为前置条件。
实施计划已保存并由当前会话执行，见 ../plans/2026-10-12-episodic-source-meta.md。
