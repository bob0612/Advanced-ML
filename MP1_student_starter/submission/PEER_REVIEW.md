# 公开后的互评：待课程发布同学的模型

当前不能把互评标为完成。2026-09-28 查询课程公开快照时，
`publication.projects` 为空；指南规定 9 月 30 日之后统一公开链接，
之后有 7 天互评期。

公开后执行：

1. 在课程榜单选一位同学，记录提交编号、固定代码版本、检查点 URL 和报告分数。
2. 在独立目录按对方文档安装所需环境，保存文件 SHA-256；核对其使用原始评分器。
3. 不重训，运行完整 CPU FP32 测试（4 线程），保存真实 JSON 和命令：

   ```bash
   python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output peer-test.json
   ```

4. 核对 428,405 targets、1,292,013 UTF-8 bytes；记录实际 `bpb`，不根据对方声称的分数修改结果。
5. 在该同学结果旁选择 Peer Review Report，填写必需的实际复现分数，并完成生成的 GitHub issue。
6. 保存互评 issue 链接。小数值差异、安装失败本身不能证明违规；任何差异结论都应有可检查证据。

尚未选择同学、运行其模型或提交互评，本文是后续执行说明，不是已完成的互评报告。
