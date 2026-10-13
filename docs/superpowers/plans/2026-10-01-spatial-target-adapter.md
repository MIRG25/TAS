# 3D Spatial Target Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在原版 source-shared 适配中增加零初始化 3D 空间 Adapter，保留旧行为、源任务与自动测试。按最新用户指令，执行 Fold0–4 的全部三个迁移方向，共15个目标实验。

**Architecture:** 保留目标 Adapter 的 norm/down/up，在 down 与 GELU 之间对空间 token 加零初始化深度卷积残差；CLS 不经过卷积。新增布尔开关默认关闭，训练与 test-only 从同一保存配置构建模型，沿用原源 checkpoint 和既有训练策略。

**Tech Stack:** Python >=3.10、PyTorch >=2.2、pytest、PyYAML；不增加运行时依赖。Windows 本地验证，Linux CUDA 服务器训练。

**Spec:** [已批准设计](../specs/2026-10-01-spatial-target-adapter-design.md)。执行者先完整读取设计与本计划。

## Global Constraints

- 项目根目录：`C:/Users/mingh/my/MTLearning/glioma_meta_learning`；不是 Git 仓库，不初始化仓库、不执行不存在的提交流程。所有编辑用 apply_patch。
- 唯一新结构：12 个目标 Adapter 各加 kernel=3、stride=1、padding=1、groups=bottleneck、bias=False 的零初始化卷积残差。
- 当前 bottleneck=96，总增量 31,104 参数；原 Adapter、目标私有两层和头部仍按原规则更新。源路径与原 MAE 冻结。
- 新开关 `adaptation.target_spatial_adapter` 默认 false；关闭时状态键、随机流及旧前向不变。
- 新实验固定 mean pooling、MAE bypass=false、token retention=0、source_residuals=false、crop probability=0.5；无 head-only warmup、EMA、L2-SP 或梯度累积新增项。
- Adapter/head LR=1e-4，private-block LR=1e-5，weight decay=0.05，batch=2；200 epoch 上限、原 LR warmup=5、每 epoch 验证，patience=30、min_epochs=30、min_delta=1e-4。
- 原患者协议、评价定义、源 checkpoint、MRI 缓存保持；100% 预算包含拟合和验证标签，基础 label seed=2028；无全预算 refit。
- 第一批执行全部五折、全部三个 source-shared 迁移方向，100%标签；不自动跑 fresh。复用匹配原源 checkpoint，仅补训缺失者，不重训MAE。每个目标适配结束自动输出并保存 test_metrics、预测、best.pt。
- 不删除历史结果，不以外层测试挑参数或拼接跨版本最佳折；本地行为验证不代表临床指标提升。

## Review Focus

- 非立方网格、CLS 和非连续张量：轴顺序不丢失，token 数错时明确报错（Task 1）。
- 构造新模块的 RNG 与 fresh 的零 up：不能改变原头部随机初始化，不能把 fresh 首步零卷积梯度误报为故障（Task 1）。
- 保存配置缺字段、字符串布尔值、CLI 覆盖：旧值默认 false，非法值拒绝，真实开关贯穿输出（Task 2）。
- 新旧 checkpoint 结构不匹配、单独 test-only 重载、旧 refit 分支：不得忽略权重，新增元数据不能只更新某个出口（Task 2）。
- AMP、gradient checkpointing 和优化器参数所有权：新参数只入组一次且能更新，源参数及旧任务预测不变（Tasks 1/3）。

---

## 文件职责

| 文件 | 操作与职责 |
|---|---|
| `glioma_meta/models/spatial_adapter.py` | 新建；目标 Adapter 包装器，保留旧权重命名并实现局部空间残差 |
| `glioma_meta/models/source_guided.py` | 修改；开关验证、目标 Adapter 扩展及 getter，不改变源模型构造 |
| `scripts/adapt_source_guided.py` | 修改；CLI/config 接入、日志、目录及所有保存出口 |
| `scripts/test_source_guided.py` | 修改；元数据校验、结构恢复、test-only 输出 |
| `configs/foundation/source_initialized_spatial_adapter.yaml` | 新建；独立、显式锁定原版设置的实验配置 |
| `tests/test_spatial_target_adapter.py` | 新建；算子、初始化、优化与冻结行为 |
| `tests/test_spatial_target_adapter_integration.py` | 新建；配置、三任务自动测试、独立重载及兼容性 |
| `scripts/run_spatial_transfer_all.sh` | 新建；五折/三个方向顺序运行、复用源模型、补训缺失源模型、出错停止 |
| `tests/test_spatial_launcher.py` | 新建；执行真实Shell流程，检查15个场景、复用和失败传播 |
| `docs/spatial_target_adapter_all_tasks.md` | 新建；操作说明、预期日志字段和服务器命令 |
| `README.md` | 修改；增加此对照实验链接，说明旧分支不是本轮起点，不清除历史文档 |

## Task 1: 空间算子与模型接入

**Files:** Create `glioma_meta/models/spatial_adapter.py`, `tests/test_spatial_target_adapter.py`; modify `glioma_meta/models/source_guided.py`。不改公共 ResidualFeatureAdapter 的现有实现。

**Interfaces:**
- `SpatialResidualFeatureAdapter(base: ResidualFeatureAdapter, grid_shape: Sequence[int])`；复用传入目标副本的 norm/down/up 子模块，不重新构造它们，不另存 base 为子模块导致重复状态键。
- 新状态参数 `spatial_weight: nn.Parameter`，形状 `[bottleneck, 1, 3, 3, 3]`；以原 down 权重的 device/dtype 创建全零张量，不消耗 RNG。
- `forward(tokens: torch.Tensor) -> torch.Tensor` 接受 `[B, 1+D*H*W, feature_dim]`。
- `SourceGuidedFoundationModel.add_target_task(..., target_spatial_adapter: bool = False) -> None` 增加末尾可选参数；`target_spatial_adapter(task: str) -> bool` 返回真实结构状态，未知任务报错。

- [ ] **1. 写失败测试：零残差、网格语义和非法输入。** 测试 `test_spatial_adapter_zero_init_preserves_base_and_rng`、`test_spatial_neighbor_and_cls_semantics`、`test_invalid_grid_or_token_count_rejected`。用 `(2,3,4)` 网格、随机非零 up，验证构造前后 torch RNG 相同，norm/down/up 张量逐项相同；初始输出 `torch.testing.assert_close(..., rtol=1e-6, atol=1e-7)`。将核某个非中心权重设为 1，以显式三维邻居索引计算期望值，证明改变邻居而不是通道；CLS 结果不变。覆盖非连续输入、错误 N 和不合法 grid。

```python
def test_spatial_adapter_zero_init_preserves_base_and_rng():
    base = ResidualFeatureAdapter(24, 8)
    torch.nn.init.normal_(base.up.weight, std=0.02)
    tokens = torch.randn(2, 25, 24)  # 1 + 2*3*4
    expected = base(tokens)
    rng_before = torch.get_rng_state().clone()
    adapter = SpatialResidualFeatureAdapter(base, (2, 3, 4))
    assert torch.equal(rng_before, torch.get_rng_state())
    assert adapter.spatial_weight.shape == (8, 1, 3, 3, 3)
    assert torch.count_nonzero(adapter.spatial_weight) == 0
    torch.testing.assert_close(adapter(tokens), expected, rtol=1e-6, atol=1e-7)
```

- [ ] **2. 运行新增测试观察红灯。** 使用下方本地 Python 的 `-m pytest tests/test_spatial_target_adapter.py -q`，确认失败来自缺少新增类/方法而非依赖环境。
- [ ] **3. 实现包装器。** 先降维，再将非 CLS token `transpose/reshape` 为三维网格；`F.conv3d(..., stride=1, padding=1, groups=bottleneck)` 的输出加回输入；恢复 token 顺序后接原 GELU/up/residual。固定恒等项不作为参数，不 detach，不能原地修改原输入或 CLS。grid 长度、正整数、输入 rank/通道/token 数明确检查。
- [ ] **4. 写模型行为失败测试。** 三目标方向参数化 `test_target_spatial_initialization_and_source_safety`，比较无字段、显式 false、true 三种构造：原共享权重先通过真实源优化一步成为非零；开启与关闭的初始目标输出相同、head 相同、RNG 相同；源状态键完全相同；开关 false 不出现 `spatial_weight`。`test_spatial_fresh_and_source_shared_capacity_matches` 证明两组新增容量相同且头部相同。
- [ ] **5. 接入模型。** 原 add_target_task 先创建普通目标 Adapter 并按原方式严格复制 shared 权重，然后仅在开关 true 时逐层包装；grid 从 `encoder.grid_shape` 获取。严格要求 bool，getter 依据目标 Adapter 类型/实际结构判定。没有新源状态键，原目标前向与冻结方法通过同一 target_adapters 容器自然生效。
- [ ] **6. 写并运行真实优化测试。** 使用生产 `_optimizer`，断言每个空间参数只出现一次，属于 target_adapters decay 组，lr=1e-4、weight_decay=0.05；source-shared 在反传后空间梯度有限且有更新。fresh 允许多步后更新；原源参数逐项 `torch.equal`、原源 eval 预测不变。小模型 depth=2、bottleneck=8 的参数差必须为 `2*8*27`；实际12层96维的公式为31,104。
- [ ] **7. 绿灯与局部回归。** `python -m pytest tests/test_spatial_target_adapter.py tests/test_source_guided.py tests/test_source_initialized_transfer.py -q`。全部通过后核对未改源路径。没有 Git，记录命令/输出，不生成虚假 commit。

## Task 2: 训练、配置、保存与重载

**Files:** Modify `scripts/adapt_source_guided.py`, `scripts/test_source_guided.py`; create new YAML and `tests/test_spatial_target_adapter_integration.py`。

**Interfaces:**
- 消费 Task 1 的 add_target_task 参数及 getter。
- CLI：`--target-spatial-adapter` / `--no-target-spatial-adapter`，`argparse.BooleanOptionalAction`、default=None；覆盖后将有效布尔值写回 `config['adaptation']`。
- 保存字段统一名 `target_spatial_adapter`；开启时运行目录末尾增加 `_spatial3d`，关闭时原目录命名不变。独立输出根目录为 `runs/source_initialized_adaptation_spatial_adapter`。

- [ ] **1. 写配置与校验失败测试。** `test_spatial_config_is_original_baseline_plus_one_module` 用 load_config 检查 spec 的数值，并与原父配置显式恢复 crop=.5 后逐项比较，允许差异仅开关、实验名、输出路径和首轮 label_fraction=1。`test_spatial_flag_metadata_rejected` 覆盖 true/false 冲突、字符串 `'false'`、整数0、null；`test_legacy_absent_spatial_flag_loads` 覆盖两处缺字段或 false。基于 `_protocol_fixture`、`_selected_checkpoint_metadata`，但模型池化改为 mean。

```python
def test_spatial_config_core_constraints():
    config = load_config('configs/foundation/source_initialized_spatial_adapter.yaml')
    settings = config['adaptation']
    assert settings['target_spatial_adapter'] is True
    assert settings['target_pooling'] == 'mean'
    assert settings['target_mae_bypass'] is False
    assert settings['source_residuals'] is False
    assert settings['target_initialization'] == 'source-shared'
    assert settings['target_private_blocks'] == 2
    assert settings['refit_on_full_label_budget'] is False
    assert settings['label_fraction'] == 1.0
    assert config['source_training']['mae_token_retention_weight'] == 0.0
    assert config['data']['random_crop']['tumor_guided_probability'] == 0.5
```

- [ ] **2. 观察失败。** `python -m pytest tests/test_spatial_target_adapter_integration.py -q`；确认错误是新 YAML/字段/校验尚不存在。
- [ ] **3. 实现配置与训练接入。** 新 YAML 继承 source_initialized_transfer，但显式恢复 crop=.5、mean、bypass=false、retention=0、source-shared、residuals=false、private2、refit=false、label_fraction=1；新增开关 true、独立 experiment/output。其余 spec 锁定值显式固定，禁止继承 attention/token-retention YAML。命令参数在任何输出写入前校验。`_initialize_target_model` 传入新参数，优化器不新增第二组以免重复入组。
- [ ] **4. 接通所有输出出口。** 用 getter 得到实际开关，在 `_payload`、有效配置、启动日志、validation-only 结果/预测、legacy refit payload、最终 test 结果/预测中保存。CLI false 必须能覆盖 YAML true；新结果仍使用既有 method 名，靠 experiment/新字段区分，避免破坏 method 白名单。现有 test-only 入口无需源 checkpoint 文件。
- [ ] **5. 实现 test-only 恢复。** `_validate_checkpoint` 检查 config 和顶层布尔值一致，缺字段按 false；重建 add_target_task 时传入该值，再 strict=True 加载。状态键不匹配不能忽略；getter 值写入测试元数据。禁止 test-only 构造优化器、改变 checkpoint 或重新划分患者。
- [ ] **6. 写三任务自动测试与重载测试。** `test_spatial_training_auto_tests_and_reloads` 对 IDH/WHO/survival 和 CLI true/false 参数化。复用 Task 3 下述 fixture 方案：真实小缓存、1 epoch 目标训练、按保存 best.pt 自动测试；断言 outer_test_evaluated=true、source_parameters_unchanged=true、正确开关和 `_spatial3d` 后缀、所有 test patient ID 恰等于协议测试池。独立 test-only 运行后 checkpoint bytes 不变，离散预测一致，连续预测 rtol=1e-6/atol=1e-7，三个任务均检查。
- [ ] **7. 兼容与失败关闭测试。** 缺新字段的旧目标 checkpoint strict roundtrip；主动删/加 spatial_weight、交换元数据开关时必须失败。已有 validation-only/refit 不变：对新开关启用的两个出口各增加小规模测试，确认元数据及结构保留，协议仍拒绝 refit。
- [ ] **8. 绿灯。** `python -m pytest tests/test_spatial_target_adapter_integration.py tests/test_source_initialized_transfer.py tests/test_source_token_retention_integration.py -q`。CLI帮助必须出现两个新开关；不得启动服务器训练。

## Task 3: 跨任务回归、交付与操作说明

**Files:** 新建 `docs/spatial_target_adapter_all_tasks.md`、全场景启动脚本及其测试；修改 README；补充新增测试；生成代码更新包和文件哈希清单到 outputs。

**Interfaces / fixture dependencies:**
- 从 `test_source_initialized_transfer` 导入 `_tiny_model(sources)`、`_protocol_fixture(tmp_path)`、`_selected_checkpoint_metadata(split)`。
- 复用 `test_source_token_retention_integration._experiment(tmp_path, target)`，其返回 `(manifest, split, sources, encoder_state, config)`；**必须在调用 `_train_source(tmp_path, monkeypatch, config, output_name)` 前**设置 source retention=0、target pooling=mean、bypass=false。其原测试仅对 survival 继续适配，不能当成三任务适配已测；新测试必须对三个任务都执行到 test-only。
- tiny source 真实训练仅是本地合成测试，不是重新训练临床源模型。空间核训练测试保持 target Adapter LR 足够使非零 up 的真实梯度可观察，不依赖脆弱的指标提高断言。

- [ ] **1. 完成 AMP/gradient-checkpointing 失败测试。** `test_spatial_amp_checkpointed_backward` 用小模型打开 gradient_checkpointing；可用 CUDA 时测试 float16 autocast 与 scaler，CPU 用 bfloat16 autocast 验证有限值并另标记 CUDA 检查是否实际运行。断言空间梯度有限、源状态不变；不得将 CUDA skip 表述为 GPU 已验证。
- [ ] **2. 完整本地回归。** 从项目根目录执行下面的 PowerShell 命令，记录实际通过数、耗时、警告/skip/失败。若失败，定位新增改动或现存问题，不删除失败测试、不偷改对照定义。

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& 'C:/Users/mingh/anaconda3/envs/cmh3.10/python.exe' -m pytest -q
```

- [ ] **3. 检查实际加载配置和命令行。** 本地 `python -m scripts.adapt_source_guided --help`、`python -m scripts.test_source_guided --help`，load_config 检查全局约束。可用本地 CUDA 做 tiny smoke，不用完整 MRI 跑真实训练。审查新/旧权重 namespace、bool 验证、源码中所有 metadata 出口。
- [ ] **4. 编写文档并链接 README。** 说明假设不是已证实原因、无性能保证、老任务保留；列出预期 `target_spatial_adapter=True` 和增加31,104参数，讲清恢复crop=.5和复用各折原源best.pt。列出下方命令。没有模块实现和验证前，不向用户宣称这些命令已可运行。
- [ ] **5. 独立最终代码审查。** 审查人检查 Task 1/2 的接口、初始化随机流、源参数保护、test-only 权重恢复和协议隔离；主执行者验证建议并修正必要问题，再重跑受影响测试及全套。设计/计划自审由主执行者完成，不替代为此审查。
- [ ] **6. 生成与检查更新包。** 路径 `outputs/spatial_adapter_update_20261001.tar.gz`；仅加入实际修改/新增的模型与脚本、配置继承链、两新增测试及其两 fixture 文件、操作文档。保留项目相对路径，不打包 MRI、runs、缓存、旧模型权重或整个 outputs。用 tar 列表检查无绝对/越界路径，解包到新建临时目录比对文件 SHA-256，生成同名 manifest；不覆盖旧更新包，冲突时改用明确新版本名。
- [ ] **7. 交付。** 报告实际修改、已运行检查及未运行的真实临床实验，提供更新包的绝对本地链接和服务器命令。不要声称 C-index 将提高。由用户上传代码包后执行全部五折、全部三个方向。

### 交付时的服务器命令（实现验证前不要运行）

上传生成的更新包至服务器项目根目录。解压失败必须立即退出，防止运行旧代码：

```bash
cd /root/autodl-tmp/project/glioma_meta_learning || exit 1
tar -xzf spatial_adapter_update_20261001.tar.gz || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
bash scripts/run_spatial_transfer_all.sh
```

不能加 `--validation-only`。自动测试输出分类或生存全部现有指标，同时保存患者预测和 best.pt。此命令不启动 fresh 对照，不更换已存在的原源 best.pt，不覆盖原版结果；仅补训缺失方向/折的源模型。

## 计划自审与执行交接

- 范围覆盖：设计1/4/7对应全局约束、Task2和交付；设计2/3对应Task1；设计5对应Task2；设计6对应三个任务的测试；设计8及研究边界进入操作说明。
- 接口统一：模型参数/getter、YAML、CLI、checkpoint和预测元数据全部使用 `target_spatial_adapter`；唯一新状态参数为 `spatial_weight`。
- 不做源损失修改或性能承诺；最新范围允许仅补训缺失源模型，未完成的真实 GPU 临床实验如实列出。
- 推荐执行方式：由当前主执行者按顺序实现（Native），最后独立审查。三个任务接口相连且改动集中，这样比逐任务交接更省上下文和时间。
- 用户已要求直接实施，不再重复确认。实现与验证记录见 outputs/spatial_adapter_execution_20261001/progress.md；该记录是实际完成状态，以上复选框保留原任务清单。
