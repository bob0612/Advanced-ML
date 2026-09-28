# MP1 小型语言模型作业完成计划 / Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> 初稿为执行计划。执行进度：2026-09-28 已完成任务 1、任务 2 的本地基线实验，以及任务 3 的基线分析和假设设计；首次网站报分尚未执行，任务 4–8 尚未执行。见 [环境记录](../../../MP1_student_starter/code/ENVIRONMENT.zh-CN.md)、[基线结果](../../../MP1_student_starter/code/BASELINE_RESULTS.zh-CN.md)及[报告草稿](../../../MP1_student_starter/report/REPORT.md)。

**Goal:** 从零训练一个符合课程规则的小语言模型，完成有机制依据的改进、公平对照与消融，提交真实、可复现的分数、代码、模型和不超过 10 页的报告。

**Architecture:** 保留原始 GPT 和固定评分流程，先建立基线，再优先研究训练期间的参数指数移动平均（EMA）。同一次训练导出原始权重和平均权重，在相同训练目标数下比较；只用验证集选择设置，冻结后测试。EMA 是待检验的候选方案，不预先承诺它会改善成绩。

**Tech Stack:** Python 3.12；作业锁定的 PyTorch 2.7.1、NumPy 2.5.3、tokenizers 0.21.4；当前 Mac 按课程说明使用 CPU、4 线程、FP32。

**Spec:** [原始作业指南](../../../MP1_student_starter/GUIDE.md)、[原始运行说明](../../../MP1_student_starter/code/README.md)、[中文运行说明](../../../MP1_student_starter/code/README.zh-CN.md)、[项目导读](../../../MP1_student_starter/PROJECT_OVERVIEW.zh-CN.md)；同时核对了 [课程提交页面](https://xudongwu-0.github.io/courses/dase7506/)（2026-09-28）。

## Global Constraints：全程必须遵守

- 排名使用完整测试集的 **BPB，越低越好**；正式评估使用 **FP32**，并且能够在 **CPU** 上复现。
- 资源限制为 **CPU scoring time ≤5× baseline、peak evaluation RAM ≤4 GiB、uncompressed inference assets ≤64 MiB**。
- 固定 `context=256`、`vocab=2048`、数据、分词器、窗口及评分协议；不修改 `common.py`、`evaluate.py` 和 `data/`。
- 保留 `model.py` 与 `configs/baseline.json`，使原始基线始终能单独运行。
- 只用训练文本学习权重与统计量；验证集用于开发和选择；方法冻结后才评估测试集。不得根据测试集成绩继续挑选设置。
- 不使用外部训练文本、预训练权重、缓存的验证/测试答案、未来 token、跨窗口状态或评估时的联网服务。
- 必须包括：初始基线、相同累计训练目标数的对照、关键机制消融，以及质量与计算成本分析。一组成对实验可以同时完成后两项。
- 披露随机种子、累计训练目标数、检查点来源、所有搜索与训练成本。重复使用检查点不能抹掉已有训练成本。
- 最终报告 **at most 10 pages including figures, tables and references**；仓库说明实质性 AI 帮助和复用来源。
- 最终提供不可变代码链接及匹配的完整模型包，别人能够不重新训练直接评分。

## 截止日期：先处理这个差异

本地 `GUIDE.md` 写的是：**9 月 29 日前**先提交学号和完整测试集 BPB；**9 月 30 日结束前（UTC+8）**最终提交，且最后一次提交必须包含代码及模型链接。

2026-09-28 查到的[课程页面](https://xudongwu-0.github.io/courses/dase7506/)则写分数截止 **2026 年 9 月 30 日（UTC+8）**，并表示链接当前可选、之后需要。两者对首次提交时间和链接时间的说明不同，不能自行判断哪一个覆盖另一个。

**保守安排：9 月 28 日先提交一个冻结版本的真实完整测试分数；9 月 30 日前把代码、模型、报告和最终分数全部补齐，同时查看教师是否发布了澄清。** 若只来得及基线，首次分数可以反映已冻结的基线；这仅解决早期报分，不代表已完成算法改进。后续更新的方案仍只能用验证集选择，每个报分版本都要独立冻结并留档，不能根据其测试成绩反向调参。

## 先用通俗语言理解作业

可以把模型看成“读到前半句话，猜下一小段文字”的程序。例如输入一串 token `[A, B, C]`，分别学习预测 `[B, C, D]`。token 是固定分词器切出来的单位，不一定是一个完整单词。

- **训练集**像练习册：用来更新模型。
- **验证集**像模拟考试：用来判断改动是否值得保留。
- **测试集**像正式考试：在方法确定后给最终成绩。
- **BPB**可以直观理解为“模型给这些文字编码，每个原始字节平均需要多少位”；越小越好。提交的是评估 JSON 的 `bpb`，不是 `token_ppl`。
- **公平对照**：避免“新方法学得更久，所以成绩更好”被误写成算法优势。
- **消融**：关闭你声称有用的机制，其他条件尽量一致，看效果是否改变。

推荐顺序：**跑通 → 基线 → 找问题 → 实现一个机制 → 公平比较 → 冻结 → 测试与资源测量 → 报告与复现 → 提交**。

## 文件职责与预计改动

以下相对路径以工作区根目录为起点；命令另行注明在 `code/` 下运行。

| 文件 | 计划如何处理 | 作用 |
|---|---|---|
| `MP1_student_starter/code/model.py` | 保留 | 原始 GPT 对照 |
| `MP1_student_starter/code/configs/baseline.json` | 保留 | 原始结构配置 |
| `MP1_student_starter/code/student.py` | EMA 路线可保留现有工厂 | EMA 是训练方法改进，不要求为了“有改动”而重写模型 |
| `MP1_student_starter/code/train.py` | 后续增加可关闭的 EMA 选项、双检查点导出及日志 | 默认不开 EMA，维持原始训练配方 |
| `MP1_student_starter/code/ema.py` | 后续新增 | 管理平均权重，避免把训练器写得过于复杂 |
| `MP1_student_starter/code/tests/test_ema.py` | 后续新增 | 验证平均公式、独立性和保存/加载 |
| `MP1_student_starter/code/tests/test_contract.py` | 保留并运行 | 因果性、归一化、样本/窗口独立性、梯度和窗口计数 |
| `MP1_student_starter/code/EXPERIMENTS.csv` | 后续从 `RUN_LOG_TEMPLATE.csv` 复制 | 逐次填写实验记录；训练器不会自动填写 |
| `MP1_student_starter/code/runs/` | 训练自动产生 | 各次实验的权重与 JSON，每次使用新目录 |
| `MP1_student_starter/REPRODUCE.md` | 后续新增 | 准确的安装、训练、评估及文件对应说明 |
| `MP1_student_starter/report/REPORT.md`、`REPORT.pdf` | 后续新增 | 报告源稿与最终不超过 10 页的 PDF |
| `MP1_student_starter/submission/ASSET_MANIFEST.csv` | 后续新增 | 最终推理文件列表、解压后字节数和 SHA-256 |

检查时，当前目录尚不是 Git 仓库；`code/` 未看到 `.venv/` 或 `runs/`，且当前 shell 的 PATH 中未找到 `python3.12`。这只说明还需要检查和准备环境，不证明电脑其他位置一定没有 Python。

## Review Focus：容易漏掉的五类问题

1. EMA 权重必须独立保存，不能因共享引用偷偷改变正在训练的原始模型；任务 4 检查原始参数更新前后不被 EMA 操作改动。
2. 原模型的输入词嵌入与输出层共享权重；EMA 导出、加载后必须保持一致；任务 4 检查共享权重与保存前后的输出。
3. EMA 初始状态、训练步号及非法参数必须明确定义；任务 4 检查边界与具体数值，不平均随机初始权重。
4. 短序列、尾部填充及未来输入不能导致错误或信息泄漏；任务 1、4、6 运行固定接口测试，并检查完整测试的目标数。
5. 模型在原开发目录能运行不代表他人能复现；任务 7 在干净目录只用发布文件加载评分，并检查全部推理资源与实际进程 RAM。

---

## 任务 1：准备环境，让最小流程能跑通

**预计主动操作时间：** 30–90 分钟；安装耗时取决于网络与现有环境，不是保证。

**Files:** 读取 `code/requirements.txt`；建立本地 `.venv/`；不改模型与评分代码。

**Interfaces:** 使用现有 `student.build_model(config)`；其 `forward(ids)` 返回 `[B,T,2048]` logits，`predict_log_probs(ids)` 返回同形状的有限、归一化自然对数概率。

- [x] 确认能运行 Python 3.12；实际在项目内安装 Python 3.12.14，系统 Python 保持原样。
- [x] 从当前工作区进入代码目录，建立虚拟环境并安装课程锁定依赖。以下保留通用 Mac 安装示例；本机环境已经建立，后续直接激活 `.venv/`：

```bash
cd MP1_student_starter/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1
python -m pip install -r requirements.txt
python --version
python -m pip freeze
```

- [x] 已记录安装中的网络限制及一次下载中断；使用 PyPI 重试成功，三个原始锁定版本均保持不变，`pip check` 通过。
- [x] 运行 `python -m unittest discover -s tests -v`，原包 5 项测试全部通过。
- [x] 运行 10 步小试验，再从磁盘加载权重在验证集评分：

```bash
python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 10 --run-dir runs/smoke-s17
python evaluate.py --checkpoint runs/smoke-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

**为什么这样做：** 先确认环境、训练、存盘、重新加载和评分能连起来，避免长时间训练后才发现安装或接口错误。这里选验证集即可，不必为安装检查查看测试集。

**完成标准：** 测试通过，产生 `checkpoint.pt`、`metrics.json`、`validation_cpu_fp32.json`，BPB 为有限数值。10 步成绩不用于声称“复现了完整基线”。

**实际结果（2026-09-28）：** 完成；10 步共处理 81,920 个训练目标，独立加载评分的验证 BPB 为 **3.5798219683823453**，与训练结束时一致。未执行测试集评分。详见 [环境与结果记录](../../../MP1_student_starter/code/ENVIRONMENT.zh-CN.md)。

## 任务 2：跑完整基线，建立可信的参照

**Files:** 原始 `model.py`、`configs/baseline.json`；新增 `EXPERIMENTS.csv`；输出到新的 `runs/baseline-s17/`。

**Interfaces:** `train.py --implementation model`；评估使用原始 `evaluate.py`；从 `metrics.json` 读取实际训练时间、参数量和验证指标。

- [x] 将原始文件留档：已保存包含代码、配置、数据与指南的源文件快照及逐文件 SHA-256；当前未建立 Git 仓库，未伪造提交编号。
- [x] 从 `RUN_LOG_TEMPLATE.csv` 创建 `EXPERIMENTS.csv`，现已记录安装检查和完整基线两次训练。
- [x] 按完整默认预算运行基线：

```bash
python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 300 --run-dir runs/baseline-s17
python evaluate.py --checkpoint runs/baseline-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

- [x] 记录参数量、训练与验证时间、最终及中途验证 BPB、Apple M5 / 16 GiB 硬件、线程数、依赖版本与检查点 SHA-256。完整测试峰值 RSS 已实际测量，没有用 GPU 字段代替 CPU 内存。
- [x] 核对累计训练目标数为 `1200 × 32 × 256 = 9,830,400`。这些是累计处理的目标，可能重复抽到同一段文本。
- [x] 为较早报分准备，将基线作为独立版本冻结后运行完整 CPU FP32 测试；得到 BPB **2.1012651958438786**，对应权重和源文件已留档。后续开发只使用验证集。
- [ ] 可选的首次网站报分待执行：学号尚未提供，本轮没有创建网站提交或 GitHub issue。本地已准备好真实 BPB；网站动作须另行完成，不能把实验完成视为已提交。

**完成标准：** 有一组完整、可复现的基线记录。约 2.10 是原材料给出的测试参考值，不是承诺值；约 311 秒训练是参考服务器的测量，不是本机预计完成时间。

**实际结果（2026-09-28）：** 本地基线交付已完成；验证 BPB **2.0710874333793914**，测试 BPB **2.1012651958438786**；训练 **334.006 秒**，完整测试评分 **7.493 秒**（单次），测试峰值 RSS **1.352 GiB**。完整证据及命令见 [基线实验记录](../../../MP1_student_starter/code/BASELINE_RESULTS.zh-CN.md)。

**特别注意：** `--eval-every 300` 只增加曲线记录，不保存“验证最好的”中间权重；当前 `checkpoint.pt` 仍是最后一步。若后续确实要选中间权重，应先明确增加保存机制，不能只根据日志声称保留了那个模型。

## 任务 3：读懂结果，提出一个可以被验证的假设

**Files:** 阅读 `model.py`、`train.py` 及基线的 `metrics.json`；把判断写入报告草稿。

**Interfaces:** 使用 `history` 与 `validation_history`；它们分别是训练损失记录和验证 BPB 记录，尺度不同，不直接相减。

- [x] 已在报告草稿中解释 embedding、因果注意力、前馈网络、输出层与下一 token 交叉熵，并区分训练损失和验证 BPB 的单位。
- [x] 已分析日志：四个验证点均下降，300→1,200 步下降 0.259997 BPB；12 个训练小批次损失记录均有限。现有证据不支持把过拟合或后期不稳定写成事实，也不足以认定完全收敛。
- [x] 已明确候选假设：**在相同训练轨迹与目标数下，EMA 可能比最后一步原始权重取得更低的验证 BPB。** 随机更新影响只是待验证的可能解释；同时记录“模型仍在改善，较早权重可能使平均滞后”的竞争假设。
- [x] 首轮继续检验 EMA，保留初稿的 decay=0.99、start=600、seed=17；同次训练的 E0/E1 为主要对照，不混入延长训练或增大模型。

**实际交付（2026-09-28）：** [报告草稿](../../../MP1_student_starter/report/REPORT.md)已包含模型解释、验证趋势、证据边界、候选比较、EMA 公式、同预算消融设计及成功/失败判定；[分析数据](../../../MP1_student_starter/report/baseline_analysis.json)保存计算和源文件哈希。本轮没有新增训练或测试评分，EMA 仍未实现。

**为什么优先选择 EMA：** 原始规则明确允许自训练权重平均；它能用同一训练过程得到“开启机制/关闭机制”的一对结果，公平性容易解释，代码范围较小，也便于控制最终文件体积。它不是多个模型一起预测：最终只加载一套平均权重。

**停止条件：** 如果基线首先出现 NaN、无法学习或接口错误，先解决那个明确问题；不要用 EMA 掩盖训练故障。如果 EMA 最后没有改善，仍完整报告这个负结果；若有时间，再基于验证现象提出一个新假设，不用测试集筛选。

## 任务 4：实现 EMA，先证明实现正确

**预计主动操作时间：** 1–3 小时；只实施这一项主要机制。

**Files:** 新增 `code/ema.py`、`code/tests/test_ema.py`；修改 `code/train.py`；保留基线模型和固定评分器。

**Interfaces（已实现）：**

- `EMA(model: torch.nn.Module, decay: float)`：克隆模型当前状态，不持有会被原模型原地修改的共享状态。
- `EMA.update(model: torch.nn.Module) -> None`：浮点状态按平均公式更新，非浮点状态复制当前值。
- `EMA.state_dict() -> dict[str, torch.Tensor]`：返回可供原模型 `load_state_dict` 使用的完整状态。
- 训练参数 `--ema-decay`：默认 `0` 表示关闭；开启时要求 `0 < decay < 1`。
- 训练参数 `--ema-start`：默认 `600`。开启时要求 `1 <= ema_start < steps`；完成第 `ema_start` 次优化器更新后初始化 EMA，此后每次优化器更新后更新 EMA。
- 默认仍输出原始 `checkpoint.pt`；开启 EMA 时额外输出 `checkpoint-ema.pt`。两者保留评分器需要的 `protocol`、`implementation`、`config`、`model`、`seed`、`train_tokens` 字段。

算法只有一条核心公式：

```text
新的平均权重 = decay × 旧的平均权重 + (1 - decay) × 当前训练权重
```

例如 `decay=0.99`，每次给当前权重 1% 的比例；模型仍正常训练，旁边单独维护一份平滑后的权重。先用 `decay=0.99, start=600`，这是预先确定的起点，不是已证实的最优设置。

- [x] 先写 `test_ema_arithmetic`：旧参数为 2、当前参数为 4、decay 为 0.9，更新后应为 2.2；确认未实现时该测试失败。
- [x] 写 `test_ema_does_not_mutate_training_model`：更新前后原始模型的参数完全不变，改变原始参数也不能回写已有 EMA 状态。
- [x] 写 `test_ema_copies_integer_buffers`：非浮点 buffer 保留当前精确值，不做浮点平均。
- [x] 实现上述 `EMA` 接口，使这些数值测试通过。
- [x] 为训练器添加上述开关与检查；默认关闭时不额外改变学习率、采样、随机数调用或原始模型更新。
- [x] 写 `test_ema_tracking_preserves_raw_training_trajectory`：相同初始化、随机种子、小批次顺序和优化器设置下，短训练分别关闭/开启 EMA 跟踪，逐个比较原始模型最终参数应一致；额外跟踪不得回写原始模型或消耗影响训练的随机数。
- [x] 写 `test_ema_initialization_and_cli_bounds`：在指定步数先完整复制当前权重，下一步才开始平均；非法 decay 与 `ema_start >= steps` 产生清楚的错误。
- [x] 写 `test_ema_checkpoint_round_trip`：用小 GPT 保存平均权重后重建加载，在相同输入上输出一致；输入 embedding 和输出 head 继续共享权重；输出形状正确且对数概率归一化。
- [x] 将原始模型和 EMA 的验证 BPB、EMA 参数与初始化步数、训练目标数、两个文件哈希写入日志。比较时重新构建 EMA 评估模型，避免把平均权重写回正在训练的原始模型。
- [x] 运行所有测试：`python -m unittest discover -s tests -v`。同时保留原来的 5 项检查；额外检查实际 EMA 加载后的模型也满足短序列、因果性和状态独立性。
- [x] 做开启 EMA 的短流程检查。以下命令**只能在新参数实现后运行**：

```bash
python train.py --implementation student --device cpu --precision fp32 --threads 4 --seed 17 --steps 10 --ema-decay 0.99 --ema-start 5 --run-dir runs/ema-smoke-s17
python evaluate.py --checkpoint runs/ema-smoke-s17/checkpoint-ema.pt --device cpu --precision fp32 --threads 4 --split validation
```

**完成标准：** 公式、文件加载与接口检查均通过；原始和平均权重都能通过原始评分器评估。只有 10 步检查通过才进行完整训练。


**实际交付（2026-09-28）：** EMA 实现位于代码提交 `ba21033`。18 项测试通过，包含原有 5 项及 EMA 载入后的同等契约检查；10 步 EMA smoke 的独立重新加载验证 BPB 为 3.599032586776047，与训练日志完全一致。固定评估器、模型、配置与数据哈希均未改变。

## 任务 5：完成最小但充分的实验组合

**Files:** 新运行目录、`EXPERIMENTS.csv`、报告结果表；不修改评分器。

**Interfaces:** 所有正式对照均为 1,200 步、batch size 32、context 256、seed 17、同一个 GPT 结构与相同原始优化器配方；改变的是导出与评估的权重。

| 编号 | 实验 | 训练目标数 | 目的 |
|---|---|---:|---|
| B0 | 任务 2 的完整原始基线 | 9,830,400 | 报告必须包含的初始基线 |
| E0 | EMA 开启训练产生的原始最终权重 | 9,830,400 | 同一次训练中的“关闭平均机制”对照 |
| E1 | 与 E0 同次训练的 EMA 权重 | 9,830,400 | 检验平均机制的作用 |
| R0/R1，可选 | 换一个预先确定的 seed，同时导出原始/EMA 权重 | 每条训练轨迹各 9,830,400 | 检查效果是否只是偶然，不挑最好看的种子 |

E0 和 E1 共用一条训练轨迹，因此搜索总成本计算一次训练；但每个模型本身都继承了全部 9,830,400 个训练目标，不能声称 EMA 没有训练成本。B0 与这次新训练的成本分别记录并相加。

- [x] 固定上述预算后运行完整 EMA 试验。以下命令依赖任务 4 的实现：

```bash
python train.py --implementation student --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 300 --ema-decay 0.99 --ema-start 600 --run-dir runs/ema-d099-s17
python evaluate.py --checkpoint runs/ema-d099-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/ema-d099-s17/checkpoint-ema.pt --device cpu --precision fp32 --threads 4 --split validation --output runs/ema-d099-s17/validation-ema.json
```

- [x] 为 E1 明确指定输出文件名。两份权重在同一个目录，若都用默认文件名，第二次评估会覆盖第一次结果。
- [x] 对比 B0 和 E0，确认打开 EMA 跟踪没有意外改变原始训练流程；若存在差异，检查采样、随机状态和原始参数是否被改动，再解释硬件数值差异的可能性。
- [x] 计算 `验证改进 = E0 验证 BPB - E1 验证 BPB`。正值才表示平均权重更好，同时报告原始数值。
- [x] 填完种子、耗时、预算、硬件、参数量、验证 BPB、文件哈希与选择依据。测试 BPB 此时留空。
- [x] 有余力时用 seed 23 再做一对，报告两对结果；这是稳健性补充，原始要求没有强制必须两种种子。
- [x] 如果尝试第二个 decay，事先限定一个额外候选并记录全部试验。不能仅展示获胜结果，隐藏其余搜索成本。
- [x] 根据验证 BPB 和资源预检确定最终候选；若 EMA 没有改善，如实记录失败机制与局限。可以保留较好的基线作为排名模型，但必须在报告中清楚区分提交模型和已实施、已消融的候选改进，不能把基线冒称为创新。

**完成标准：** 至少有 B0、E0、E1 的可追溯结果，能够回答“效果是否来自平均机制”“是否增加了训练数据量”“代价是什么”。E0/E1 这一对同时满足同预算对照与机制消融；不必为了数量再堆许多重复实验。

## 任务 6：冻结最终方法，完成正式评分与三项资源检查

**Files:** 最终代码快照、最终检查点、唯一命名的测试 JSON、资源测量记录和资产清单。

**Interfaces:** 使用未修改的 `evaluate.py`；CPU、FP32、4 线程；完整 test 应包含 **428,405 targets、1,292,013 UTF-8 bytes**。

- [x] 先在验证集完成推理时间、内存和文件体积预检，给资源限制留出余量，再冻结代码、配置、权重、依赖和所需资源；记录对应哈希。
- [x] 在 `REPRODUCE.md` 明确写出提交的是哪一个文件。以下示例假定验证阶段选择了 E1；若选了其他文件，替换为那个已冻结检查点：

```bash
python evaluate.py --checkpoint runs/ema-d099-s17/checkpoint-ema.pt --device cpu --precision fp32 --threads 4 --split test --output runs/ema-d099-s17/final-test-cpu-fp32.json
```

- [x] 从输出核对 split、FP32、targets、utf8_bytes 和 checkpoint 哈希；提交 JSON 中完整精度的 `bpb`，不手工改写成绩。
- [x] 在同一台 CPU、相同线程数和精度下，分别测冻结基线与最终模型的完整测试评分时间。推荐各重复 3 次、分别指定不同输出名，报告全部时间和中位数及比值。重复仅用于计时和复现，不据此改模型。
- [x] 检查 `最终模型评分秒数 / 基线评分秒数 <= 5`。比较 JSON 中评分阶段的 `seconds`；另行报告包含加载的进程时间，不混用两种口径。
- [x] 用操作系统工具记录整个评估进程的峰值 RAM；Mac 可用以下包装命令，保留原始输出并根据本机 `man time` 确认单位：

```bash
/usr/bin/time -l python evaluate.py --checkpoint runs/ema-d099-s17/checkpoint-ema.pt --device cpu --precision fp32 --threads 4 --split test --output runs/ema-d099-s17/resource-test-cpu-fp32.json
```

- [x] 按字节换算核对 RAM 不超过 `4 × 1024^3` 字节。现有 `device_metrics()` 的 CPU 显存数值为 0，不能替代 RAM 测量。
- [x] 清点全部必需推理资产的**解压后**体积，包括最终权重、额外表格/词典/平均模型之外的任何必需资源；不要只看压缩包大小或 `.pt` 的一个文件。把评分会用到的提供文件也单列大小，明确总量口径，并对整个必需文件集合做保守核对，目标不超过 `64 × 1024^2` 字节。

**测量口径说明：** 本地 README 的资源测量部分只有三条限额，没有完整测量步骤。上述“同机同线程、多次计时、进程峰值 RAM、完整资产清单”是可复查的实施建议，不应在报告中声称它就是老师已公布的唯一口径；如课程另有正式细则，按其执行。

**完成标准：** 最终模型的分数、三项资源数据和代码/权重指纹都可对应。若发现资源不合格，停止宣称该模型可提交；回到验证阶段解决资源问题，并如实记录此前测试，不把测试成绩作为新方案调参依据。


**实际结果（2026-09-28）：** 已完成 seed 17 与 23 的配对实验。EMA 分别使验证 BPB 增加 0.003963995、0.003732096，未带来改善；选择原始 B0。一次中止运行的目标数/耗时不确定性单列披露。最终测试 BPB 2.1012651958438786，3 次一致；CPU 时间中位数比值 1.1035，峰值 RSS 1.4535 GiB；全部源码、数据和对照权重保守共 33.7677 MiB，三项约束通过。详见 `submission/` 下选择、冻结、成本及资源证据。未尝试第二个 decay。

## 任务 7：写报告并做一次“别人视角”的复现

**预计主动操作时间：** 3–5 小时；建议从任务 2 开始同步填表，而不是最后再找记录。

**Files:** `report/REPORT.md`、`report/REPORT.pdf`、`REPRODUCE.md`、`submission/ASSET_MANIFEST.csv`，以及独立的模型下载包。

**Interfaces:** 代码版本必须重建发布检查点所需的模块；模型包解压后无需训练即可运行原始评估命令。

- [x] 报告按以下提纲写，预留空间，包含图、表和参考文献总计不超过 10 页：

| 内容 | 建议篇幅 | 需要回答的问题 |
|---|---:|---|
| 任务与基线 | 1 页 | 预测什么、怎样评分、原始模型是什么？ |
| 观察与假设 | 1 页 | 实际看到了什么，为什么尝试 EMA？哪些只是推测？ |
| 方法 | 1–2 页 | 平均公式、开始时机、参数、导出方式是什么？ |
| 实验设计与结果 | 2 页 | B0/E0/E1 是否同预算？关键数值、消融结论是什么？ |
| 成本、局限与分析 | 1–2 页 | 时间、RAM、文件大小、总搜索成本，哪些结论证据不足？ |
| 复现、引用和 AI 帮助 | 1 页 | 谁能用什么命令复现？复用了什么？AI 帮助了哪些部分？ |

- [x] 至少放一张核心结果表：方法、seed、累计目标数、参数量、验证 BPB、已冻结版本的测试 BPB、训练时间、CPU 评分时间、峰值 RAM、资产大小。没有运行的测试填写“未评估”，不编造。
- [x] 可画训练损失曲线和验证 BPB 曲线，分别使用清楚的坐标轴；图用于支持结论，不要求为了美观增加不必要图表。
- [x] 对负结果也给出克制解释。例如“该预算下未观察到 EMA 稳定改善”，不能凭一个种子断言“EMA 永远无效”。
- [x] 在仓库 README 或清晰链接的说明中披露实质性 AI 帮助，并确认自己能够解释代码和实验；保留数据来源及分发声明。
- [x] 写准确的环境、安装、训练、评估命令，指定工作目录、checkpoint 文件名、预期 BPB 与合理数值误差说明。不要把候选命令当成最终命令。
- [x] 在一个新的目录，仅放入准备发布的代码与模型包，重新安装/使用记录的环境，运行接口检查，再直接进行完整 CPU FP32 测试；禁止依靠开发目录里的临时文件或重新训练来补救缺失文件。
- [x] 核对重现分数、资源记录及 SHA-256 与提交版本相符；打开最终 PDF 检查页数、表格、公式、文字和图是否截断。
- [x] 生成不可变代码链接，例如 GitHub 的 `/tree/<commit-sha>`；配套的模型包也保留固定版本和校验值。注意 `runs/` 和 `*.pt` 默认被 `.gitignore` 忽略，代码 push 成功不代表模型已经发布。

**完成标准：** 他人拿到两个链接就能直接评估；报告读者能看懂机制、公平性、效果和代价；最终 PDF 不超过 10 页。


**本地交付：** 6 页中文 PDF 已逐页检查，公式和表格未截断，中文字体嵌入。独立临时目录仅复制发布代码与模型包，使用已记录依赖环境通过全部 18 项测试，完整测试分数与哈希一致；证据为 `submission/evidence/CLEAN_REPRODUCTION.json`。可选曲线未绘制，采用结果表。远端代码、报告和模型包已发布到 bob0612/Advanced-ML 的固定提交 0659d1eb28752ced329b3c94ac460b2628c4b527；匿名下载全部 HTTP 200，哈希一致。课程 Issue 已准备，等待明确授权公开学号和分数。

## 任务 8：完成网站提交与之后的互评

**Files:** 使用已经验证的代码与模型链接、最终完整测试 JSON。

**Interfaces:** [课程提交页面](https://xudongwu-0.github.io/courses/dase7506/)生成 GitHub 提交内容；需在 GitHub 上完成创建才算提交。

- [ ] 输入学号和准确的完整测试 BPB；最终一次提交同时附上不可变代码链接和对应模型包链接。
- [ ] 如果更新已有报分，使用同一个 GitHub 账号，并按网页要求填写原始提交编号；保留提交编号和最终记录。
- [ ] 不能只点击网页按钮便关掉：确认生成的 GitHub issue 已实际创建，且内容与最终版本一致。
- [x] 检查下载链接权限，确保老师和同学无需向你索要权限就能取得规定文件；截止后保持冻结版本可下载。
- [ ] 链接公开后，在 7 天互评期内选择一位同学，使用其固定代码和 checkpoint 跑完整 CPU FP32 评分，提交实际复现的分数。
- [ ] 互评时保存命令、环境和输出；小数值差异或安装失败本身不代表违规。奖励由教师确认的差异报告决定，不把奖励当作必然获得。

**完成标准：** 最终提交记录包含正确分数和两个匹配链接；之后按课程要求完成互评。

## 结合当前日期的实际排期

制定计划时本机时间为 **2026-09-28 晚间，香港时间**。以下是工作顺序建议，训练时间应根据本机实测调整。

| 时间 | 优先工作 | 当天应留下的成果 |
|---|---|---|
| 9 月 28 日晚 | 环境、5 项测试、10 步检查、完整基线；核对截止差异；按较早要求报一个真实冻结版本的分数 | 可运行环境、基线权重和日志、早期提交记录 |
| 9 月 29 日 | 实现并测试 EMA；跑 E0/E1；填表；有余力再做第二个 seed | 一个完整的同预算对照与消融实验，报告方法和结果初稿 |
| 9 月 30 日前半天 | 最终候选资源预检、冻结、测试和正式资源测量；定稿报告 | 最终 JSON、资源数据、不超过 10 页的报告 |
| 9 月 30 日余下时间 | 干净目录复现、发布固定代码与模型、更新并核对提交 | 可用的两个链接、匹配的最终提交记录；留出上传与修复时间 |
| 公开后 7 天 | 复现另一位同学并提交互评 | 真实复现分数及可选证据 |

**时间不够时的删减顺序：** 先删额外结构尝试、大范围调参、第二个 decay 和第二个 seed；保留完整基线、至少一个机制实现、同预算对照、消融、资源测量、真实报分、报告和可复现模型包。不要把关键交付物换成更多实验次数。

## 最后逐项打勾

- [ ] 我能解释下一 token 预测、因果性和 BPB。
- [x] 原始完整基线有权重、日志和成本记录。
- [x] 做了一个真实的训练/结构/记忆机制改动，并有代码与实验证据。
- [x] 有相同累计训练目标数的对照和关键机制消融。
- [x] 所有开发选择来自验证集，未通过测试集筛选模型或种子。
- [x] 最终分数来自固定评分器、完整测试集、CPU FP32。
- [x] 三项资源限制均有实际测量，单位清楚。
- [x] 代码版本与模型包匹配，干净目录无需重训就能复现。
- [x] 报告不超过 10 页，披露搜索成本、复用来源和实质性 AI 帮助。
- [ ] 提交 issue 已创建，最终记录包含代码和模型两个有效链接。

## 本计划自查记录

初稿已逐项对照原始指南和 README，覆盖基线、同预算对照、消融、固定评分、资源、成本披露、报告、链接、截止日期及互评。新增 EMA 参数均标为后续实现，未冒充现有功能；区分课程硬性规定、当前代码行为、实施建议与待验证假设。后续已完成任务 1、任务 2 的本地实验及任务 3 的分析设计，证据已补入相应任务；任务 4–6 与任务 7 本地交付也已完成；独立审查与远端发布已完成；课程报分等待明确授权，互评需等课程公开同学链接。
