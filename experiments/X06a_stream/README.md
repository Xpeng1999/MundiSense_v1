# X06a 连续流原型

**状态：修订 2 执行中，尚未全部交付。** ET-test 1,929 句与 FMH 3,266 句正确切句已完成；FMH H3 S6b OP-N Δ=0.008625，95% CI [-0.001436, 0.018895]。已完成 GPU 作业累计 4.605242 / 18 小时。S7 范围为 H3 四条 ET-test 流、H_ref 仅说话人 21。当前事实和未完成项目以 [REPORT.md](REPORT.md) 为准。

启动凭据：[revision_2_budget_startup.json](results/revision_2_budget_startup.json)。前两次停止报告完整保存在 [REPORT_BEFORE_REVISION_1.md](REPORT_BEFORE_REVISION_1.md) 与 [REPORT_BEFORE_REVISION_2.md](REPORT_BEFORE_REVISION_2.md)。`REPORT.md` 按当前完成凭据更新；历史停止和未测项目均不记为当前验收通过。PR #7 保持 Draft、不合入。

协议快照在 `protocol/`；执行解释与兼容性问题在 `OPEN_QUESTIONS.md`。模型、融合、校准、中性偏置及拒识阈值均冻结。部署默认 H3，可选 H_ref；`calibration_bundle_v1.1.json` 仅改变 `gate_rules` 的字节区间。

执行主机为 `wsl`，新产物根目录为 `/mnt/h/emotionAI/emotioncore_runs/X06a`。从干净的仓库根目录运行，Python 使用 `/home/administrator/emotion-data/emotioncore_runs/X02/venv/bin/python`；`PYTHONPATH` 包含新根目录下的 `deps` 与仓库 `src`。依赖副本和模型缓存均留在 X06a 根目录，旧实验目录只读。

```bash
python -m emotioncore.x06.run --input /绝对路径/EmotionTalk片段.mp4 --system H3 --out events.jsonl
```

此实验入口接受冻结清单中的 FMH 或 ET-test 片段，拒绝其余来源。相对 `--out` 保存在 X06a 的 `offline_runs/` 下；绝对输出路径也必须位于 X06a 根目录。事件文件和同名 `.run` 目录内的运行凭据不得覆盖。`--realtime` 开启按原时间戳的 1 倍回放，默认全速。

修订后的顺序为启动预算检查 → ET-test 745 句续推并复用原 1,184 句 → FMH 正确切句与 S6b → 0.4 / 0.8 秒两种 S2 → S3 → 选定静音的 S5 → H3 四流 / H_ref 说话人 21 的 S7。最短静音只在 FMH 的 H3 情绪 macro-F1 上选择，0.005 内选 0.4；FSMN 与 0.3 阈值不变，原 `configs/x06_vad_freeze.json` 不覆盖。相同输入 S6a 已完成，不重跑。

新增的 `resume_et` 入口只推理剩余 745 句，把原 1,184 句原始字节直接拼入新的完整结果文件。`pre_s7` 按序执行至 S5，每阶段后写新的预算回执，遇失败立即停止；它不自行启动 S7。最短静音与 EMA 冻结文件提交后，才允许实时回放。

GPU 任务全部经过 18 小时总账本 watchdog，包含全部子进程墙钟和历史失败初始化。原停止回执保持不变；只有经过 SHA 核对的修订 2 及最新预算回执才能授权后续阶段。旧 S6 入口不能直接重跑。

全速阶段共享 H_ref/H3 的共同分量；实时阶段独立执行所选系统所需的前向，不回放缓存概率。常驻 Q 使用 NF4，专家使用 D1_fmhx。静默事件使用 H_V/OP-R80，标记 `visual_only`，不进入主指标。

实时延迟按 VAD 判定结束到 JSONL 写出并 flush 计时。原时间戳按 1 倍前进；若处理落后，录制文件保留待处理帧，不丢帧，另报输入积压与墙钟/媒体时长比。解码及人脸分项包括端点判定之前已完成的工作，不能直接与端点后延迟相加。

所有事件、逐句结果、输入缓存与完整日志留在服务器；仓库交付聚合 JSON、报告以及原产物路径和 SHA-256。
