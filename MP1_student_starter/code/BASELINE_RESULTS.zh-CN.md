# 完整基线实验记录

2026-09-28 已完成原始 GPT 的 1,200 步训练、重新加载后的验证评分，以及冻结后的完整测试评分。实验编号：`baseline-s17`。

**完整测试 BPB：2.1012651958438786。** 这是原始基线的真实成绩，可用于首次报分；它不代表已经完成算法改进。本轮没有向课程网站或 GitHub 提交内容。

## 实验设置

| 项目 | 实际值 |
|---|---|
| 实现 / 配置 | 原始 `model.py` / `configs/baseline.json` |
| 结构 | 4 层 GPT，width=128，heads=4，context=256，vocab=2048 |
| 参数量 | 1,088,256 |
| 初始化 / 随机种子 | 随机初始化 / 17 |
| 训练步数 / batch size | 1,200 / 32 |
| 累计训练目标数 | **9,830,400** |
| 优化器与学习率 | 原始 AdamW 配方，未改动；基础学习率 0.001，100 步预热及原始余弦衰减 |
| 设备 / 线程 / 精度 | CPU / 4 / FP32 |
| 硬件 | Apple M5，10 个物理/逻辑核心，16 GiB 系统内存 |
| 环境 | macOS 27.0 ARM64，Python 3.12.14，PyTorch 2.7.1，NumPy 2.5.3，tokenizers 0.21.4 |
| 权重选择 | 预先指定原始配方的第 1,200 步权重，没有根据测试集选择配置或检查点 |

这里的 16 GiB 是机器总内存，不是评估进程的峰值 RAM。所有统计量按实际输出记录。

## 成绩和验证曲线

| 训练步数 | 验证 BPB |
|---:|---:|
| 300 | 2.3310841919657976 |
| 600 | 2.1841632253775156 |
| 900 | 2.1061162454589173 |
| 1,200 | 2.0710874333793914 |

最终权重保存后，独立运行 `evaluate.py`，验证 BPB 仍为 **2.0710874333793914**，与训练结束时完全一致。

随后于 **2026-09-28 20:14:17（UTC+8）**生成冻结记录，再运行完整测试，得到：

| 测试检查 | 实际值 |
|---|---:|
| BPB | **2.1012651958438786** |
| 有效预测目标数 | 428,405 |
| 原文 UTF-8 字节数 | 1,292,013 |
| 评估窗口数 | 1,674 |
| 设备 / 精度 | CPU / FP32 |

验证集包含 376,599 个目标、1,148,007 字节和 1,472 个窗口。两份评分的目标数、窗口数、逐窗口损失总和、BPB 计算及检查点哈希均已核对。

四个验证点依次下降，目前没有观察到验证反弹。仅凭这四个点不能声称基线已经过拟合或训练后期不稳定；后续讨论 EMA 时，需要把“可能有益的假设”与“实际观察到的问题”分开。后续方法和参数选择继续只使用验证集。

## 耗时、内存与文件体积

| 项目 | 实际测量 |
|---|---:|
| 训练阶段耗时，不含准备和验证 | **334.006 秒，约 5 分 34 秒** |
| 训练脚本记录的完整进程耗时，含准备及其中的验证 | 377.009 秒，约 6 分 17 秒 |
| 本次所有验证评分阶段累计耗时，含独立重载评分 | 41.190 秒 |
| 独立验证评分阶段耗时 | 6.591 秒 |
| 完整测试评分阶段耗时 | **7.493 秒** |
| 完整测试进程墙钟时间，含加载 | 16.22 秒 |
| 完整测试进程峰值 RSS | **1,451,671,552 字节，约 1.352 GiB** |
| macOS 同时报告的 peak memory footprint | 1,712,064,432 字节，约 1.594 GiB |
| 检查点体积 | **4,370,805 字节，约 4.168 MiB** |
| 归档的全部代码、文档、数据与检查点未压缩合计 | **17,926,532 字节，约 17.096 MiB** |

资源测量使用 `/usr/bin/time -l`。本机 `man 2 getrusage` 明确 `ru_maxrss` 单位为字节。实验表 `peak_ram_gb` 按字段名用十进制 GB 填写，即 `1.451671552`；本页换算成二进制 GiB。RSS 和 memory footprint 是不同的系统指标，两者均保留原始数值。

体积统计保守计入了归档中的训练数据、指南等文件，因此范围大于最小推理文件集合；不把已安装的 Python、第三方库和虚拟环境算作模型资产。全部文件见 [资产清单](runs/baseline-s17-records/asset-manifest.csv)。

这是一轮实际 CPU 计时，尚不是最终改进模型的多次计时比较。之后检查“≤5 倍基线”时，应在同一机器和相同条件下重复测量基线及冻结的改进模型；不能把本次单次测量当成其他机器的固定时限。

任务 1 的 10 步安装检查也已加入实验表。迄今两次训练合计处理 **9,912,320 个训练目标**，训练阶段合计 **337.013 秒**；不是只披露完整基线而忽略安装试验的训练成本。

## 复现命令

以下命令在 `MP1_student_starter/code/` 下执行。

直接复评保存的模型，无需训练：

```bash
.venv/bin/python evaluate.py --checkpoint runs/baseline-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
.venv/bin/python evaluate.py --checkpoint runs/baseline-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

若需重新训练，使用一个新的、尚无结果的目录：

```bash
.venv/bin/python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 300 --run-dir runs/baseline-s17-reproduce
```

课程训练器只保存最后一步权重；300、600、900 步的验证记录不代表保存了对应模型。

## 产物与版本记录

- [实验总表](EXPERIMENTS.csv)，包含安装检查与完整基线两行。
- [完整基线权重](runs/baseline-s17/checkpoint.pt)。
- [训练指标及曲线数据](runs/baseline-s17/metrics.json)。
- [验证评分 JSON](runs/baseline-s17/validation_cpu_fp32.json)。
- [完整测试评分 JSON](runs/baseline-s17/test_cpu_fp32.json)，首次报分取这里的 `bpb`。
- [冻结记录](runs/baseline-s17-records/FROZEN_BASELINE.json)，创建于测试评分前。
- [原始代码、配置及数据快照](runs/baseline-s17-records/source-snapshot.tar.gz)。
- [逐文件源码哈希](runs/baseline-s17-records/source-manifest.json)。
- [训练原始日志](runs/baseline-s17-records/train.log)。
- [测试资源测量原始输出](runs/baseline-s17-records/test-resource.txt)。
- [成本统计](runs/baseline-s17-records/costs.json)及[核验结果](runs/baseline-s17-records/verification-summary.json)。

代码快照 SHA-256：

```text
9e01f4fc50dd35da80263f9a5ee96b2d9638a1779b23defe7c41e02b068efbb2
```

检查点 SHA-256：

```text
5bb600b4dc65e8a8fa0eaa6841955b99754bbb61b060fb626d0855940d26206a
```

当前目录尚未建立 Git 仓库，实验表的 `code_commit` 留空，以真实快照和哈希记录版本，没有伪造提交编号。快照中的 18 个文件与本次运行时文件逐一匹配，原始模型、训练器、评分器及数据未修改。

独立验证评分已成功，但默认沙箱禁止系统计时工具读取 `kern.clockrate`，使那次外层计时命令以非零状态退出；因此没有填写验证进程的峰值内存。随后获准执行系统资源读取，完整测试和资源测量均以状态码 0 完成，未因测量问题修改评分器。

首次网站报分仍待执行：需要学号，并实际完成网站生成的 GitHub issue。此处已备齐真实分数与本地对应文件，但不应把本地实验完成误认为网站已提交。
