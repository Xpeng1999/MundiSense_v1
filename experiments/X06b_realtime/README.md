# X06b 实时化与切句重测

状态：全部测量及服务器最终审计于 2026-10-14 02:11:24（北京时间）完成，等待 reviewer 验收。R3、R4 达到对应条件；R5 积压达标，墙钟比、计划端点延迟和内存增长未达标。PR #7 已按 X06a 有条件通过意见合入；本轮分支 `x06b-realtime`，PR #8 保持 Draft、不合入。

全部模型、v1.1 参数包、FSMN/0.3/0.4 秒、α=0 冻结。新代码在 `emotioncore.x06b`，产物仅写 `/mnt/h/emotionAI/emotioncore_runs/X06b`。X06a 及更早目录只读。R2 只用 FMH；ET-test 只报告；MER-test、SIMS-valid 不用；CH-SIMS test 零推理。

预登记见 [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) 和 `configs/x06b_registration.json`。原协议与验收意见保存在 `protocol/`。R2 按 FMH 结果选中 1/2；服务器 `r2/freeze.json` 原始字节已在 `958da60` 提交为 `configs/x06b_face_freeze.json`，随后才启动后续阶段。含恢复控制的 70 项 CPU 回归测试通过；FMH 真实片段集成检查通过，详见 [报告](REPORT.md)。

R2、R3、R4、AISHELL-4 CPU 补测、R5、曲线和最终审计均已完成。旧 ET-test 格式异常仅重做独立 CPU 汇总，未重复推理。原 STOP、失败日志和未结束 run_meta 保留，以 [恢复回执](results/resume_authorization.json)引用。恢复代码为 `277462b`，七个 GPU 子进程均正常退出，累计 **7.519318 / 14 小时**，每阶段预算重估均在门禁内。[最终审计](results/delivery_audit.json)明确区分历史失败与重试成功；详见 [恢复记录](STOP_RECOVERY.md)。

一次性续跑驱动 `emotioncore.x06b.resume` 已完成，见 [阶段回执](results/stages_resume1.json)；不得再次启动。历史 `stages/summary.json` 的 stopped 状态与原 STOP 指向旧尝试，没有新增停止回执。

| 验收项目 | 实测 | 原门槛 | 达到条件 |
| --- | --- | --- | --- |
| R3 FMH OP-N 差区间下界 | −0.011781 | ≥−0.02 | 是 |
| R4 FMH VAD−正确切句 F1 | −0.015382 | ≥−0.03 | 是 |
| R5 墙钟/媒体 | 1.001054 | ≤1.0 | 否 |
| R5 积压 P95 | 0.998267 秒；后半程未增大 | ≤2 秒且后半程不增大 | 是 |
| R5 计划端点延迟 P50/P95 | 2.213491 / 2.804833 秒 | ≤1.5 / 2.5 秒 | 否 |
| R5 第10分钟到结束内存增长 | 11.038% | <10% | 否 |

完整结果及曲线见 [11 节报告](REPORT.md)，配置和入口见 [X07 交接](../../docs/X07_HANDOFF.md)。测量完成不代表整体验收通过；后续工作范围等待 reviewer 确认。

以下是登记入口；已完成的本轮目录不可重跑或覆盖：

- CPU 启动与选型：`python -m emotioncore.x06b.preflight`、`python -m emotioncore.x06b.r2`。
- 配置冻结后的顺序驱动：`python -m emotioncore.x06b.stages --launch`；依次执行 CPU 集成检查、R3、R4、R5，失败即停止。每次 GPU 阶段之后必须重估预算。
- 全速 H3 入口由 `gpu_guard r3_fmh` / `r3_et_test` / `r4_correct_fmh` / `r4_correct_et_test` / `r4_vad_fmh` / `r4_vad_et_test` 执行，先由 `prepare` 生成对应选定配置的输入。不可跳过依赖和冻结门禁。
- 实时入口：`python -m emotioncore.x06b.gpu_guard r5`，仅回放 X06a 相同四条 ET-test 流；不回放缓存概率。

GPU 统一上限 14 小时，按子进程完整墙钟计算；CPU-only 任务单列。每次正式运行记录干净提交、冻结 SHA、输入/输出回执。所有运行目录禁止覆盖。原始帧、事件、逐句结果留在服务器，仓库仅收集汇总及路径/SHA。
