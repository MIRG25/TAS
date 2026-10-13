# 原版 source-shared 基线上的 3D 空间 Adapter 设计

日期：2026-10-01。状态：用户要求直接实施；最新范围为五折及全部三个迁移方向。执行证据见项目 outputs/spatial_adapter_execution_20261001/progress.md。

## 1. 目标与证据边界

研究目标不变：利用标签充足的源临床任务学习 MRI 表征和初始化，使用有限目标标签适配源阶段未见的任务，同时保留已有任务。输入仍为四模态 MRI；目标推理不得输入源任务真值或临床变量。

固定基线是原始 `source_initialized_adaptation` 的 source-shared 方案，而不是后续 attention、MAE bypass 或 token retention 分支。用户提供的 Fold0 测试结果为 C-index 0.7037815126、iAUC 0.7752723220、IBS 0.2483862981；原始五折 C-index 为 0.6861 ± 0.0336。这两个统计层级不能混用。

已核查：现有 `ResidualFeatureAdapter` 对每个 token 独立做 LayerNorm、降维、GELU、升维及残差相加。Transformer 已有空间交互；本方案不是修复“完全没有空间建模”，而是验证增加可训练的三维局部交互是否改善低标签适配。此前日志不能证明这是性能瓶颈，也不能保证本改动使五折均值超过 0.70。

## 2. 唯一结构改动

只扩展目标任务的 12 个 Adapter。保留其从源共享 Adapter 复制的 LayerNorm、down、up 权重及现有目标私有两层 Transformer。

对每层 Adapter：

1. `h = down(norm(tokens))`，沿用原来的降维结果。
2. 分离 CLS；将空间 token 按编码器既有 flatten 顺序恢复为 `[B, bottleneck, D, H, W]`。
3. 空间特征执行 `h_spatial_new = h_spatial + DWConv3d(h_spatial)`。
4. 恢复 token 顺序并接回未经该卷积处理的 CLS 降维特征。
5. 沿用 `tokens + up(GELU(h_new))`，最终仍为 mean pooling 和原任务头。

DWConv3d 固定 kernel=3、stride=1、padding=1、groups=bottleneck、bias=False。全部卷积权重初始化为零；固定恒等支路不参与权重衰减。不能改成“可训练卷积中心初始化为 1”再对该中心施加权重衰减，否则引入额外的基线收缩效应。

当前 128³ 输入、16³ patch 对应 8³ 空间 token，bottleneck=96。每层增加 96×27 个参数，12 层共增加 31,104 个参数。这个操作只混合已有 patch 特征，不能恢复 patch 内被压缩掉的细节，也不扩大 MRI 裁剪视野。不得把视频的时间轴解释套用到 MRI 空间轴上。

## 3. 初始化、随机性与旧任务保护

- 新增空间核不得改变原 Adapter、任务头的初始化随机序列。以零参数直接构造或严格隔离构造 RNG；不能仅在普通 Conv3d 随机初始化后清零而忽略其随机数消耗。
- 同一源 checkpoint、同一随机种子下，开启空间模块的初始目标输出应与关闭版本在 FP32 数值容差内一致。保持数据加载 RNG 隔离机制。
- 正常目标适配仍更新目标 Adapter 副本、目标私有两层 Transformer 和新头；新增卷积使用目标 Adapter 的原学习率和 weight decay。不得改变优化器其他组、梯度累积或 warmup 策略。
- 源共享 Adapter、源私有 Adapter、源任务头、原 MAE 编码器均不改变。空间模块只属于目标副本。源/目标不共享可训练参数存储。
- fresh 对照使用相同空间模块、相同容量和优化设置；其区别仍仅为不复制源共享 Adapter 权重。开启空间模块不意味着源训练已经学习了空间核。

## 4. 基线配置锁定

新增独立配置，计划命名 `configs/foundation/source_initialized_spatial_adapter.yaml`；不修改已有实验配置，不删除历史结果。新配置引用当前基础配置，但必须显式覆盖已漂移的设置。

| 项目 | 本实验固定值 |
|---|---|
| 初始化 | source-shared |
| 源模型 | 各折各方向原 `runs/source_initialized_learning/fold{fold}_{source-pair}_holdout-{target}/best.pt`；仅补训缺失者 |
| 源 token retention | 0；不得换用 retention 训练出来的源模型 |
| 目标私有 Transformer | 2 层 |
| Pooling / MAE bypass | mean / false |
| 源残差、门控、源强度搜索 | 关闭 |
| 随机裁剪引导概率 | 0.5；不是当前父 YAML 的 1.0 |
| 输入与 MRI 缓存 | 沿用原四模态缓存与 128³ 裁剪 |
| 学习率 | Adapter 1e-4、目标私有 block 1e-5、head 1e-4 |
| Weight decay / batch size | 0.05 / 2 |
| 最多 epoch / 每轮验证 | 200 / 是 |
| 原学习率 warmup | 5 epoch；不新增 head-only warmup |
| 早停 | 原 validation 主指标、patience 30、min_epochs 30、min_delta 1e-4 |
| 标签预算 | 第一轮 100%，包含目标拟合与验证标签 |
| 标签基础 seed | 2028；原代码按 fold 派生，不另挑 seed |
| 患者协议 | 原 transfer_protocol.json；fit/validation/test 患者不重新划分 |
| 全预算 refit | 关闭，测试 validation 选定的 checkpoint |
| 新输出根目录 | `runs/source_initialized_adaptation_spatial_adapter` |

以已有原版 effective config、split 和 checkpoint 的记录核对实际设置；历史 effective config 中的 label_fraction 可能保留未被旧 CLI 写回的默认值，预算以运行结果和患者 ID 记录交叉确认。不得仅凭一个字段错误判定或静默替换患者集合。

新增 `adaptation.target_spatial_adapter` 布尔开关，默认 false；可提供对应 CLI 开关。缺少该字段的旧配置与 checkpoint 必须继续采用原路径。内核尺寸、位置及范围本轮固定，不暴露更多调参选项。

## 5. 保存、恢复及出错处理

- 新增卷积属于各目标 Adapter，自然纳入目标 Adapter 优化器参数组；不得漏训或重复计入。
- 从旧源 checkpoint 建模时先严格加载原源权重，再创建原版目标副本并严格复制其 LN/down/up，最后附加零空间核。不把含新卷积键的整个目标模块直接与旧 Adapter 状态做不匹配的严格加载，也不使用全局 `strict=False` 隐藏不匹配。
- 新目标 checkpoint 保存空间开关及完整权重；独立 `test_source_guided` 按保存配置恢复并严格加载。开关、元数据、状态键不一致时明确报错。
- 结果 JSON、checkpoint、有效配置、患者预测元数据和启动日志记录空间开关；启用时运行目录增加明确的空间模块后缀，避免与旧结果碰撞。
- 旧 checkpoint 缺少开关按 false 处理；不增加旧模式的状态键，不影响旧模式参数或前向。
- 空间 token 数与 grid_shape 不匹配、错误布尔配置或检查点缺失等情况应在正式训练前尽早报错，不猜测形状或悄悄改配置。
- 验证选择、C-index/iAUC/IBS/KM/log-rank 的计算与参考患者保持现状。本轮不改变评价定义。

## 6. 验证标准

实施时先建立行为测试，再写生产代码。至少覆盖：

1. 关闭开关时复现旧图、参数键和初始化随机序列。
2. 开启零卷积时，源初始化目标输出与旧版本一致；旧 LN/down/up/head 权重逐项一致。
3. 非立方小网格与 CLS 分离测试，确认轴顺序正确；构造邻居脉冲证明非中心卷积核确实建立空间交互而非只混通道。
4. 真实优化器下空间核获得有限梯度并能更新，目标其他参数按原机制更新；旧源参数及预测不变。
5. source-shared/fresh 两组保持容量匹配和头部初始化一致，不因构造新模块改变采样 RNG。
6. IDH、WHO、生存三任务的小规模训练—验证—自动测试；保存患者预测和指标，并验证 test-only 重载预测一致。
7. 开关、状态键与元数据不匹配被拒绝；旧 checkpoint 可严格恢复。
8. 新 YAML 相对原版只改变空间模块、实验名称/输出路径及明确的 CLI 标签预算；已失败模块均关闭。
9. 运行完整已有测试套件，记录实际通过/失败及环境限制。单元测试通过不代表 UCSF 性能提高。

fresh Adapter 的 up 原本为零，空间核首步梯度可能为零；测试需允许 up 开始学习后再检查空间核更新，不能把这个初始化性质误判为断梯度。开启 AMP 时还需确认前向、梯度有限以及 checkpointed backward 可正常执行。

不在本地启动完整临床训练，不自动连接服务器、不上传患者数据、不删除任何 MRI、checkpoint 或历史结果。

## 7. 服务器实验与判断

按用户最新指令，第一批运行 Fold0–4 的全部三个 source-shared 100% 迁移方向：IDH+WHO→survival、IDH+survival→WHO、WHO+survival→IDH，共15个目标实验。复用匹配原源 checkpoint；仅当缺失时使用原源训练目标补训，不重训 MAE。训练完成自动加载 validation 选定的 best.pt，输出并保存 `test_metrics`、患者预测和 checkpoint，不加 `--validation-only`。命令见 docs/spatial_target_adapter_all_tasks.md，更新包在项目 outputs/ 下。

这些折已经在开发阶段多次查看，不能视为新的独立确认性测试。报告全部折及方向的变化，不凭某一折过0.70宣布成功，不拼接跨版本最优折，不针对外层测试调 epoch、学习率、卷积层数或挑种子。

容量匹配的 spatial-fresh 和低标签实验后续另行安排；本批已覆盖统一配置的全部五折、全部三个方向。100% 标签结果本身不证明少标签适配能力。

本轮不对数值提升作承诺。空间 Adapter 是有既有研究基础的工程假设，不将“加 3D 卷积”本身表述为已确立的论文创新。

## 8. 文献与代码依据

- 现有实现：`glioma_meta/models/multitask.py` 的 ResidualFeatureAdapter；`glioma_meta/models/source_guided.py` 的 source-shared 初始化及目标路径；`glioma_meta/models/foundation_transfer.py` 的 patch 网格和 mean pooling。
- 原版配置证据：`outputs/fold34_audit_20260930/runs/source_initialized_adaptation/` 内保存的有效配置和患者记录。
- [Convpass 原论文](https://arxiv.org/abs/2207.07039)：卷积适配器的局部归纳偏置依据，不是本数据集的效果证明。
- [ST-Adapter 原论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a92e9165b22d4456fc6d87236e04c266-Abstract-Conference.html)：轻量三维卷积适配思路的相关工作；其视频任务实验不能直接外推到胶质瘤 MRI。

本目录不是 Git 仓库，直接在指定项目实施，保留改前文件备份；不创建仓库或声称已经提交。
