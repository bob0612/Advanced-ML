# 第一步环境准备与运行记录

状态：**2026-09-28 已完成任务 1**。本次完成依赖安装、全部接口测试、10 步训练及重新加载权重后的验证集评分。

## 当前环境

| 项目 | 实际值 |
|---|---|
| 平台 | macOS 27.0，Apple ARM64 |
| Python | 3.12.14 |
| PyTorch | 2.7.1 |
| NumPy | 2.5.3 |
| tokenizers | 0.21.4 |
| pip | 25.0.1 |
| 训练与评分设备 | CPU，4 线程，FP32 |
| 虚拟环境 | `code/.venv/` |

原始 `requirements.txt` 中的三个版本均成功安装，`pip check` 通过。实际安装的全部依赖版本保存在 [requirements.macos-py312.lock.txt](requirements.macos-py312.lock.txt)，它是这次 Mac 环境的记录，不替代课程原始依赖文件。

初始系统 Python 为 3.9.6，未找到 3.12。通过项目内的 uv 0.12.19 安装了独立 CPython 3.12.14，再用其 `venv` 模块创建虚拟环境；uv 的 Python 安装机制见 [官方说明](https://docs.astral.sh/uv/guides/install-python/)。

- 工具位于工作区根目录 `.tools/uv/`。
- Python 位于工作区根目录 `.tools/python/cpython-3.12.14-macos-aarch64-none/`。
- **保留 `.tools/python/`：当前虚拟环境依赖这里的解释器，删除或移动它会使 `.venv/` 失效。** 换目录或换电脑时，应重新创建虚拟环境。
- 未修改系统 Python、全局 PATH、shell 启动配置或原始课程依赖版本。
- 根目录 `.gitignore` 排除了 `.tools/`；课程自带 `.gitignore` 已排除 `.venv/`、`runs/` 和 `*.pt`。

## 以后怎样使用

从当前工作区根目录运行：

```bash
cd MP1_student_starter/code
source .venv/bin/activate
python --version
```

应显示 `Python 3.12.14`。每次新开终端，重新激活即可。也可不激活，直接使用 `.venv/bin/python` 运行命令。

在 VS Code 中，Python 解释器选择 `MP1_student_starter/code/.venv/bin/python`。

重新创建同类 Mac 环境时，先准备 Python 3.12，创建 `.venv/`，再安装 `requirements.txt`；如需精确重建此次全部传递依赖，可使用记录的 `requirements.macos-py312.lock.txt`。不要把整个 `.venv/` 当作可跨机器复制的发布包。

## 实际完成的检查

以下命令均在 `code/` 下执行：

```bash
.venv/bin/python -m pip --disable-pip-version-check --no-cache-dir check
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 10 --run-dir runs/smoke-s17
.venv/bin/python evaluate.py --checkpoint runs/smoke-s17/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
```

已有 `runs/smoke-s17/` 包含结果，再训练时必须换一个新的 `--run-dir`；直接重评现有检查点则可以继续使用它。

| 检查 | 结果 |
|---|---|
| 依赖兼容性 | `No broken requirements found.` |
| 原包接口测试 | 5/5 通过 |
| 模型参数量 | 1,088,256 |
| 训练步数 / batch size | 10 / 32 |
| 累计训练目标数 | 81,920 |
| 训练阶段耗时 | 3.006749 秒，不含准备和验证 |
| 验证集 BPB | 3.5798219683823453 |
| 从磁盘重新加载后验证 BPB | 3.5798219683823453，与训练结束时一致 |
| 独立评分阶段耗时 | 8.339955 秒，不含加载 |
| 验证目标数 / 原文 UTF-8 字节数 | 376,599 / 1,148,007 |
| 原始文件保护检查 | 13 个源代码、依赖、配置、测试和数据文件的 SHA-256 均未改变 |

**这个分数仅证明流程可运行。10 步远少于完整基线的 1,200 步，不应把它当作完整基线或最终提交成绩。此次没有执行测试集评分。**

## 结果与日志

- [10 步检查点](runs/smoke-s17/checkpoint.pt)
- [训练指标](runs/smoke-s17/metrics.json)
- [重新加载后的验证评分](runs/smoke-s17/validation_cpu_fp32.json)
- [接口测试日志](runs/setup-step1/contract-tests.log)
- [训练日志](runs/setup-step1/smoke-train.log)
- [验证日志](runs/setup-step1/smoke-validation.log)
- [环境信息](runs/setup-step1/environment.json)
- [初始文件哈希](runs/setup-step1/original-sha256.json)

检查点 SHA-256：

```text
3d01a051e2db1a90ca60929d6fc5f76ae8dd266cc1d2cb4fbbb964c560b0e29d
```

安装时先遇到默认沙箱 DNS 限制，获准联网后安装继续。第一次依赖下载以状态码 120 提前退出且没有明确诊断信息；保留相同版本，采用无缓冲输出、关闭进度条和详细日志重试后成功。没有据此修改课程代码或锁定版本。
