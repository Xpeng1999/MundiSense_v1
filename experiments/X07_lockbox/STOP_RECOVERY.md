# X07 标签隔离自检停止与修正提案

**状态：停止，等待明确授权。此文件和补丁不是恢复入口。**

## 已发生的事

L1 协议在 `6d9d07c` 提交并推送；L2 全部通过。2026-10-14 13:06:54（北京时间）在 L3 CPU 准备结束附近主动停止。指标脚本尚未运行，valid/test 都没有 GPU 推理；SIMS-test 的媒体与标签未打开。

问题在 `src/emotioncore/x07/metrics.py`：读取 meta.csv 的全部 label 列之后才筛选 valid。虽然该实现只会计算 valid 指标，它会不必要地提前构造 test 标签数据，不符合本轮最保守的封存访问要求。这是执行 agent 的实现缺口；现有 105 项测试覆盖了全部推理完成之前不能运行指标，但没覆盖混合 CSV 中其他划分的标签读取。

原 `configs/x07_lockbox_protocol.json` 与全部登记代码/配置均未修改。原目录 `/mnt/h/emotionAI/emotioncore_runs/X07`、STOP、输入与日志保留。核验见 [停止回执](results/stop_evidence.json) 和 [准备审计](results/preparation_audit.json)。

## 可审阅的修正

[proposed_label_isolation.patch](proposed_label_isolation.patch) **仅为提案，未应用到分支中的冻结源文件或正式执行目录**：

1. 首先只读取 video_id / clip_id / mode，验证当前划分的精确 ID 集。
2. 根据这一无标签清单构造 CSV skiprows，再只解析当前划分的 label。第二次读取仍位于该划分推理完成和一次性指标锁之后。
3. 校验“一条 CSV 记录对应一物理行”；不满足就拒绝读取标签，避免多行记录破坏 skiprows 对齐。
4. 新增 valid/test 双向测试：另一划分放入无法转为数字的标签哨兵，确保不被解析；另测多行 CSV 必须在标签读取前拒绝。原来的提前执行及二次运行拦截继续保留。

补丁只在隔离目录 `/mnt/h/emotionAI/emotioncore_runs/X07/proposed_fix_check` 测试，未调用正式 stages/prepare/infer/metrics 入口，测试使用构造样本，没有真实 SIMS 标签或 GPU 前向。**108 项 CPU 测试通过**，见 [日志](results/proposed_fix_tests.txt)。

## 请求授权的精确范围

允许应用上述标签读取修正，并为修正后的同一冻结系统建立**新的替代协议 `configs/x07_lockbox_protocol_v2.json` 与独立执行目录 `/mnt/h/emotionAI/emotioncore_runs/X07_v2`**。原协议、执行目录、停止与失败记录保留；先提交新协议，再重做 L2，按原门槛完成 L3，通过后才执行 L4。GPU 仍按 X07 总预算 4 小时控制，现已使用 0 秒。

新协议只变更标签访问代码、代码 SHA、协议溯源和执行路径；模型、参数包、人脸/VAD 配置、指标定义、bootstrap、缺媒体处理、验收门槛均不变。不得用本次 7 条原生帧解码失败来修改解码或选参；它们按冻结规则登记。

在得到明确授权前，不应用提案、不改变原协议、不重启任何阶段。X07 PR 保持 Draft、未合入。
