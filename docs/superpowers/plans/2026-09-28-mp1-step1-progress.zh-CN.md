# 执行记录 — plan: 2026-09-28-mp1-completion-plan.zh-CN.md

执行范围：用户要求“开始第一步”，本轮仅执行任务 1：环境准备、接口测试、10 步训练与验证集评分。

- Ruling: 用户限定第一步，因此不继续完整基线、算法修改或提交；其余任务保持待办。
- Ruling: 当前目录不是 Git 仓库，本步也不改生产代码；直接在当前项目创建独立 Python 与虚拟环境，使用文件哈希留存原始状态，不初始化 Git 或创建 worktree。
- Pre-flight: 任务 1 输出 Python 3.12 虚拟环境及 smoke 权重/指标；任务 2 可直接复用环境。固定评分接口和数据不变。
- 初始检查：系统 Python 为 3.9.6，arm64；在 PATH 与常见安装位置未找到 Python 3.12；无现成 `.venv/`。
- 已记录 13 个源代码、依赖、配置、测试及数据文件的初始 SHA-256，位于 `MP1_student_starter/code/runs/setup-step1/original-sha256.json`。
- 默认沙箱内访问 PyPI 出现 DNS 失败；联网安装申请获准后，uv 0.12.19 已安装到工作区 `.tools/uv/`。未修改系统 Python 或 shell 配置。
- Python 3.12.14 已安装到 `.tools/python/cpython-3.12.14-macos-aarch64-none/`，通过 `python3.12 -m venv` 创建 `MP1_student_starter/code/.venv/`；pip 为 25.0.1。
- 首次依赖安装在下载期间以状态码 120 退出，日志未提供明确原因；检查发现虚拟环境当时只有 pip，磁盘可用空间充足。使用相同依赖、无缓冲输出、关闭下载进度条并开启详细日志重试；没有改变依赖版本。
- 相同锁定版本重试成功：PyTorch 2.7.1、NumPy 2.5.3、tokenizers 0.21.4；`pip check` 通过，全部传递依赖记录于 `code/requirements.macos-py312.lock.txt`。
- Task 1: complete — 原包 5/5 接口测试通过；10 步 CPU FP32 训练成功（81,920 targets）；从磁盘重载后完整验证集评分成功，BPB 为 3.5798219683823453，与训练结束时一致。
- 验证覆盖 376,599 个目标、1,148,007 字节；未执行测试集评分。原始 13 个文件哈希全部保持一致。
- 输出：`code/runs/smoke-s17/checkpoint.pt`、`metrics.json`、`validation_cpu_fp32.json` 及逐窗口损失；环境说明见 `code/ENVIRONMENT.zh-CN.md`，原始日志见 `code/runs/setup-step1/`。
- Ruling: 本轮没有实现新功能、修复模型或形成开发分支，只执行环境和原包检查；保留原测试，无新增实现测试，也不进行整份计划结束时的代码评审、合并或发布。执行记录保留供后续任务继续使用。
