# MP1 项目结构与代码导读

本导读依据当前目录中的全部 Python 源码、配置、清单和两份英文文档编写。它解释当前实现，不代表额外的课程规则；没有执行依赖安装、模型训练或测试集评估。

完整翻译见 [GUIDE.zh-CN.md](GUIDE.zh-CN.md) 和 [code/README.zh-CN.md](code/README.zh-CN.md)。

## 1. 这个项目要做什么

这是一个**从随机初始化开始训练小型语言模型**的课程作业。模型读到一段文本前缀后，为下一个 token 给出概率。你要改进模型结构或训练方法，并用实验解释改进的作用及代价。

三个容易混淆的概念：

| 概念 | 在本项目中的含义 |
|---|---|
| token | 分词器生成的文本单位，不一定是一个完整单词。固定词表有 2048 个 token。 |
| 训练目标 | 需要预测的下一个 token。默认每个序列提供 256 个训练目标。 |
| BPB | 每字节比特数，是提交和排名使用的指标；越小表示给真实文本分配的概率整体越高。 |

当前 `student.py` 直接调用 `model.py` 的 `GPT`。因此，**现有 student 实现与基线使用相同的模型结构**，文件里还没有独立的模型改进。目录中也尚无 `runs/`、训练检查点或结果日志。

## 2. 当前目录结构

```text
当前工作区/
├── .DS_Store                         macOS 目录元数据
└── MP1_student_starter/
    ├── .DS_Store                     macOS 目录元数据
    ├── GUIDE.md                      英文作业指南
    ├── GUIDE.zh-CN.md                 新增：作业指南完整中文译文
    ├── PROJECT_OVERVIEW.zh-CN.md      新增：本导读
    └── code/
        ├── README.md                 英文安装、运行与技术规则
        ├── README.zh-CN.md           新增：README 完整中文译文
        ├── student.py                学生模型构造入口
        ├── model.py                  基线 GPT
        ├── train.py                  训练入口
        ├── evaluate.py               固定评分器
        ├── common.py                 数据、设备、模型加载及窗口工具
        ├── requirements.txt          Python 依赖版本
        ├── RUN_LOG_TEMPLATE.csv      实验记录表头
        ├── PACKAGE_MANIFEST.json     发布包文件哈希清单
        ├── .gitignore                Git 忽略规则
        ├── configs/
        │   └── baseline.json         基线结构配置
        ├── tests/
        │   └── test_contract.py      模型接口与因果性等测试
        └── data/
            ├── manifest.json         数据协议、来源版本及哈希
            ├── tokenizer.json        已拟合的固定 BPE 分词器
            ├── wikitext_train.txt    训练文本
            ├── wikitext_validation.txt
            │                         验证文本
            └── wikitext_test.txt     测试文本
```

`.DS_Store` 与模型无关。`.gitignore` 忽略 `.venv/`、`__pycache__/`、`*.pyc`、`runs/` 和 `*.pt`；因此常规 Git 提交不会自动包含训练结果和权重，需要另外准备可下载的检查点文件包。

## 3. 你正在看的五个文件

### `student.py`：主要模型开发入口

当前核心代码只有：

```python
from model import GPT

def build_model(config):
    return GPT(config)
```

`build_model` 是模型工厂：输入配置，输出 PyTorch 模型。使用 `--implementation student` 时，训练器动态导入这个模块；评估器以后也会根据检查点里保存的模块名重新调用它。

开头英文注释的含义是：

> 在这里实现你的算法。默认实现是完整、可运行的基线。要求诊断一个局限，并实现结构、训练或记忆机制方面的改进；解释方法、测量成本，并对机制做消融实验。仅给基线改名，或报告一个碰巧成绩好的随机种子，不构成算法贡献。保留两个模型接口的前提下，可以完全替换这个工厂或模型。

这两个接口及模型属性为：

| 接口/属性 | 要求 | 使用方 |
|---|---|---|
| `model.context` | 等于 `256` | `common.make_model` |
| `forward(ids)` | 返回形状 `[B, T, 2048]` 的未归一化 logits | 训练器，通过 `model(ids)` 调用 |
| `predict_log_probs(ids)` | 返回同形状、有限且归一化的自然对数概率 | 评分器与部分测试 |

`B` 是批量大小，`T` 是输入序列长度。代码测试会使用短于 256 的序列，因此模型也应能处理较短输入。

### `train.py`：控制模型如何学习

按执行顺序，它会：

1. 读取命令行参数，设置设备、精度、CPU 线程数和随机种子。
2. 检验并读取数据，载入配置，通过 `make_model` 构造模型。
3. 建立 AdamW 优化器。
4. 从训练文本的 token 序列中随机抽取连续的 257 个 token；前 256 个作为输入，后 256 个作为目标。
5. 计算交叉熵、反向传播、裁剪梯度，并更新参数。
6. 按选定间隔计算验证集指标；训练结束后再进行一次验证。
7. 保存最终权重和运行指标。

例如，抽出的序列是 `[A, B, C, D]` 时，输入为 `[A, B, C]`，目标为 `[B, C, D]`。A 位置预测 B；B 位置可以看到 A、B，用于预测 C。真实目标由训练器传给损失函数，不额外传给模型。

默认设置：

| 设置 | 默认值或行为 |
|---|---|
| 实现模块 | `student` |
| 结构配置 | `configs/baseline.json` |
| 输出目录 | `runs/baseline-s17` |
| 设备 / CPU 线程 | `cpu` / 4 |
| 随机种子 | 17 |
| 更新步数 | 1200 |
| batch size | 32 |
| 每个样本的目标数 | 256 |
| 优化器 | AdamW，权重衰减 0.1 |
| 学习率 | 基础值 0.001，乘以前 100 步预热系数和余弦衰减系数 |
| 梯度范数上限 | 1.0 |
| 训练精度 | `auto`；CPU 用 FP32，支持 BF16 的 CUDA 默认用 BF16 |
| 中途验证 | `--eval-every 0`，即默认仅在训练结束后验证 |

默认累计训练目标数是：

```text
1200 × 32 × 256 = 9,830,400
```

这是**累计处理量**，随机抽取时可能重复看到同一段文本，不能把它理解为 9,830,400 个互不重复的 token。

该脚本目前没有恢复训练的命令行参数，也不会自动保存验证成绩最好的中间模型。`--eval-every 300` 只增加验证记录；最终 `checkpoint.pt` 仍是最后一步的模型。检查点中不保存优化器状态。

每次使用一个空的或不存在的输出目录；目录已经含有文件时，脚本会停止，避免混合不同实验的结果。

### `RUN_LOG_TEMPLATE.csv`：跨实验比较的记录表

这个 CSV 当前只有表头，没有实验记录。训练器不会自动填写它。

| 字段 | 中文含义 |
|---|---|
| `run_id` / `code_commit` | 实验编号 / 对应代码提交版本 |
| `method` / `config_path` / `seed` | 方法名 / 配置路径 / 随机种子 |
| `parent_checkpoints` | 继承或复用的检查点 |
| `steps` / `batch_size` / `context` | 更新步数 / 批量大小 / 上下文长度 |
| `processed_targets_including_ancestry` | 累计处理的训练目标数，包含来源检查点的训练成本 |
| `parameters` | 参数量 |
| `hardware` / `threads` / `train_precision` | 硬件 / 线程数 / 训练精度 |
| `train_seconds` / `validation_seconds` / `test_seconds` | 训练 / 验证 / 测试耗时 |
| `peak_ram_gb` / `peak_gpu_allocated_gb` | 峰值 RAM / GPU 已分配显存 |
| `validation_bpb` / `test_bpb` | 验证集 / 测试集 BPB |
| `checkpoint_sha256` | 检查点文件指纹 |
| `selected_on` / `notes` | 选择依据 / 备注 |

开发过程中用验证集选择方法，测试结果留到方法冻结后记录。CSV 的字段名使用 `gb`，作业内存限制使用 GiB；填写和比较时需要明确单位，`1 GiB = 2^30` 字节，`1 GB = 10^9` 字节。

### `requirements.txt`：锁定依赖版本

当前内容是：

```text
torch==2.7.1
numpy==2.5.3
tokenizers==0.21.4
```

- `torch`：神经网络、自动求导、优化器、设备计算和权重保存。
- `numpy`：评估器用它保存每个窗口的损失数组。
- `tokenizers`：读取 `tokenizer.json`，把文本编码成 token ID。

Python 3.12 的要求在 README 中，未写入此文件。本次仅解释这些锁定版本，没有检查软件源可用性或安装兼容性。

### `README.md`：实际操作手册

主要回答“怎样安装、运行、评估、提交”，同时定义模型接口和评分口径。作业目标、截止日期和互评规则在上一级的 `GUIDE.md`。

## 4. 其余代码和资源如何配合

### `model.py`：完整基线 GPT

模型的数据流如下：

```text
输入 token ID [B, T]
    ↓
token embedding + 可学习的位置 embedding [B, T, 128]
    ↓
4 个 Transformer Block
    每块：LayerNorm → 因果自注意力 → 残差相加
          LayerNorm → 两层前馈网络（GELU）→ 残差相加
    ↓
最终 LayerNorm
    ↓
输出投影 → logits [B, T, 2048]
    ↓ 评估时再做 log_softmax
自然对数概率 [B, T, 2048]
```

- 每个 Block 有 4 个注意力头，每头维度为 `128 / 4 = 32`。
- 前馈网络先把维度从 128 扩大到 512，再缩回 128。
- `is_causal=True` 使当前位置只能关注自己及之前的 token。
- `self.head.weight = self.token.weight` 让输出投影和输入词嵌入共享权重。
- 当前采用可学习的绝对位置嵌入；位置编号在每次输入时从 0 开始。
- 这个文件也定义了 `build_model`，因此 `--implementation model` 可以直接训练原始基线。

### `configs/baseline.json`：结构参数

| 键 | 值 | 含义 |
|---|---:|---|
| `vocab` | 2048 | 词表大小，固定 |
| `width` | 128 | 隐藏表示维度 |
| `heads` | 4 | 注意力头数 |
| `depth` | 4 | Transformer Block 数量 |
| `context` | 256 | 最大上下文长度，固定 |

此配置不包含学习率、优化器或训练步数；这些在 `train.py` 中定义。新实验可以另建配置并通过 `--config` 指定，保留原始基线配置用于比较。

### `common.py`：固定的公共逻辑

| 函数/常量 | 作用 |
|---|---|
| `ROOT` / `PROTOCOL` | 定位 `code/` 目录 / 标记固定基准协议 |
| `sha` | 计算文件 SHA-256，用作内容指纹 |
| `load_data` | 按数据清单核验哈希，再编码三个数据划分；返回 token 张量和原始字节数 |
| `setup` | 设置 CPU/CUDA、精度与线程；当前不接受 Apple MPS |
| `autocast` | 按设置启用或关闭 BF16 自动混合精度 |
| `make_model` | 动态导入实现模块，调用工厂，检查 context 和词表大小，记录模块哈希 |
| `windows` | 构造固定评估窗口及右移一位的目标，处理末尾短窗口 |
| `device_metrics` | 返回设备信息，以及 CUDA 峰值已分配/保留显存 |

`load_data()` 会读取并分词训练、验证和测试三个划分。默认训练循环的梯度更新只使用 `data['train']`，训练期间的评分只使用验证集；加载测试文件本身不等于用测试集训练或调参。

`windows()` 把不足 256 个目标的最后一段补齐，并用 `-100` 标记无效目标；评分器会忽略这些填充位置。相邻窗口共享一个边界 token，但模型状态不跨窗口传递。

### `evaluate.py`：固定评分器

执行过程为：加载检查点 → 核对协议 → 按模块名和配置重建模型 → 加载权重 → 读取数据 → 调用 `score()` → 保存结果。

`score()` 关闭梯度计算，临时把模型切换为评估模式，逐批调用 `predict_log_probs(x)`。它检查输出形状、数值是否有限、概率是否归一化，然后用真实目标对应的对数概率计算损失。模型接口只收到输入 `x`。

评分公式可写为：

```text
NLL = 对所有有效目标累加 -ln P(真实的下一个 token | 当前窗口内可见前缀)
BPB = NLL / ln(2) / 整个数据划分的原始 UTF-8 字节数
token_ppl = exp(NLL / 有效目标数)
```

BPB 的分母是字节数，`token_ppl` 的分母是目标 token 数；两者不能混用。评分从每个数据划分的第二个 token 开始，分母仍包含完整原文的所有字节。

评估器默认 `--split test`，所以开发阶段应显式加 `--split validation`。正式提交使用完整测试集结果中的 `bpb`，且评估精度为 FP32。

### `tests/test_contract.py`：五项基本检查

| 检查 | 排查的问题 |
|---|---|
| 改变未来输入，不改变之前的预测 | 是否偷看未来 token |
| 对数概率归一化，且样本单独运行与批量运行一致 | 输出是否是概率分布、样本间是否相互影响 |
| 两次处理同一输入之间插入另一窗口，结果仍一致 | 是否跨窗口残留状态 |
| 下一 token 交叉熵产生有限、非零梯度 | 是否能正常进行梯度训练 |
| 包括最后短窗口在内，每个目标恰好出现一次 | 评估窗口是否漏算或重复计算 |

测试使用较小的随机初始化模型，不需要训练数据下载。它们检查基本行为，不衡量训练后的 BPB，也不等于已满足全部课程约束。

### `data/`：固定数据和分词器

| 文件 | 内容及用途 |
|---|---|
| `wikitext_train.txt` | 用于学习模型参数和其他训练统计量；10,951,562 字节 |
| `wikitext_validation.txt` | 用于开发、参数及检查点选择；1,148,007 字节 |
| `wikitext_test.txt` | 方法冻结后的最终评估；1,292,013 字节 |
| `tokenizer.json` | 2048 项词表的 BPE 分词器，使用 ByteLevel 预分词与解码 |
| `manifest.json` | 协议名、上游数据版本、构建说明以及四个数据/分词器文件的 SHA-256 |

分词器不是预训练语言模型，它负责把文本转换成固定词表中的 ID。不要重新拟合或替换它，否则改变了本作业的评分基准。

### `PACKAGE_MANIFEST.json`：发布文件清单

它列出源文件、文档、配置和数据的发布哈希。实际运行时，`load_data()` 校验的是 `data/manifest.json`；当前代码并不读取 `PACKAGE_MANIFEST.json` 来校验整个包。

## 5. 一次训练和评估会产生什么

```text
配置 JSON + 模块名
    ↓ common.make_model()
student.build_model(config) 或 model.build_model(config)
    ↓
train.py ← 训练 token
    ├── forward(ids) → 交叉熵 → 参数更新
    ├── evaluate.score() ← 验证 token
    └── runs/某次实验/
        ├── checkpoint.pt
        └── metrics.json
                ↓
evaluate.py --checkpoint ... --split validation 或 test
    ├── 按检查点中的模块名和配置重建模型
    ├── predict_log_probs(ids) → 评分
    └── 在默认输出位置生成
        ├── validation_cpu_fp32.json 或 test_cpu_fp32.json
        └── 对应的 *.window-nll.npy
```

| 产物 | 保存的内容 |
|---|---|
| `checkpoint.pt` | 协议、实现模块名、配置、模型权重、种子、本次累计训练目标数 |
| `metrics.json` | 参数量、训练精度与耗时、验证指标、训练/验证历史、PyTorch 版本、线程数、哈希、设备指标等 |
| `*_cpu_fp32.json` | BPB、token 困惑度、NLL、目标数、字节数、评分时间、文件哈希和设备信息等 |
| `*.window-nll.npy` | 每个窗口的累计自然对数负对数似然；保存的是窗口总损失，不是逐 token 损失 |

检查点不包含 Python 模型源代码。因此仅分享 `.pt` 文件不够，必须同时提供能被导入的实现模块、相关辅助代码及推理所需文件。训练器记录的 `implementation_sha256` 只对应所选实现模块自身，不是整个代码目录的合并哈希。

## 6. 适合按什么顺序阅读和使用

1. 先读 `GUIDE.zh-CN.md`，明确评价指标、实验要求和提交内容。
2. 再读 `code/README.zh-CN.md`，准备环境并运行接口测试。
3. 从 `student.py` 进入 `model.py`，看懂输入、输出和因果注意力。
4. 阅读 `train.py`，理解训练目标、采样方式、优化器及保存逻辑。
5. 阅读 `common.windows` 和 `evaluate.score`，理解固定评分方式。
6. 复现基线并记录成本，再修改 `student.py`、训练方案和新配置；开发阶段用验证集比较，最终冻结后再测试。

按原文约束，主要可修改区域为 `student.py`、`train.py` 和新增辅助文件；保留 `model.py` 与 `configs/baseline.json` 以便公平比较，保持 `common.py`、`evaluate.py`、数据及分词器不变。

“相同训练目标数对比”要求考虑 batch size、训练步数及来源检查点的已用训练量，不能只比较步数。“消融实验”是关闭或移除你提出的关键机制，保持其他条件尽量一致，观察效果是否随之变化。

## 7. 当前材料中的几个具体细节

- **文档路径不一致：** 英文文档和发布清单指向 `guide/GUIDE.md`，实际指南在材料包根目录。中文译文已使用当前目录下有效的链接。
- **发布清单与现有文件存在差异：** 在新增中文文档之前检查发现，`code/README.md` 的 SHA-256 与发布清单记录不同，清单中的 `guide/GUIDE.md` 路径也不存在。这仅说明当前文件/布局与该清单不完全一致，无法据此判断原因。本次没有改写清单或英文文件。
- **资源测量说明不完整：** 原始 README 只列出三项资源上限。`score()` 的 `seconds` 统计评分阶段，模型/数据加载在计时前完成；`device_metrics()` 的峰值字段是 CUDA 显存，CPU 下为 0，不能把这个 0 当作 CPU 峰值 RAM。现有脚本也不会自动汇总所有推理文件大小。
- **参考耗时不是本机实测：** 约 311 秒训练、5.92 秒评分和约 2.10 BPB 均来自英文 README。本次没有运行这些实验，也不能直接把参考机器的 5.92 秒乘五当作所有机器的固定时间上限。
- **Mac 当前走 CPU：** 公共设备设置只接受 CPU 或 CUDA，传 `--device mps` 会被拒绝；按提供的 macOS 说明使用 CPU。

此次新增的三个中文 Markdown 文件用于阅读，不参与训练、评分或数据哈希校验。
