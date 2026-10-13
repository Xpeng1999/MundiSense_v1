# X06b 实时化与切句重测

状态：执行中，尚未交付或验收。PR #7 已按 X06a 有条件通过意见合入；本轮分支 `x06b-realtime`，只开 PR、不合入。

全部模型、v1.1 参数包、FSMN/0.3/0.4 秒、α=0 冻结。新代码在 `emotioncore.x06b`，产物仅写 `/mnt/h/emotionAI/emotioncore_runs/X06b`。X06a 及更早目录只读。R2 只用 FMH；ET-test 只报告；MER-test、SIMS-valid 不用；CH-SIMS test 零推理。

预登记见 [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) 和 `configs/x06b_registration.json`。原协议与验收意见保存在 `protocol/`。R2 按 FMH 结果选中 1/2；服务器 `r2/freeze.json` 原始字节已在 `958da60` 提交为 `configs/x06b_face_freeze.json`，随后才启动后续阶段。53 项 CPU 回归测试和 FMH 真实片段集成检查通过，详见 [报告](REPORT.md)。

从干净提交运行的入口：

- CPU 启动与选型：`python -m emotioncore.x06b.preflight`、`python -m emotioncore.x06b.r2`。
- 配置冻结后的顺序驱动：`python -m emotioncore.x06b.stages --launch`；依次执行 CPU 集成检查、R3、R4、R5，失败即停止。每次 GPU 阶段之后必须重估预算。
- 全速 H3 入口由 `gpu_guard r3_fmh` / `r3_et_test` / `r4_correct_fmh` / `r4_correct_et_test` / `r4_vad_fmh` / `r4_vad_et_test` 执行，先由 `prepare` 生成对应选定配置的输入。不可跳过依赖和冻结门禁。
- 实时入口：`python -m emotioncore.x06b.gpu_guard r5`，仅回放 X06a 相同四条 ET-test 流；不回放缓存概率。

GPU 统一上限 14 小时，按子进程完整墙钟计算；CPU-only 任务单列。每次正式运行记录干净提交、冻结 SHA、输入/输出回执。所有运行目录禁止覆盖。原始帧、事件、逐句结果留在服务器，仓库仅收集汇总及路径/SHA。
