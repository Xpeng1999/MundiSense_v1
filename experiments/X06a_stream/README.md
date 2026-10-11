# X06a 连续流原型

**状态：修订 1 启动预算门禁停止。** 预计累计 20.233545 GPU 小时（含 3.247282 小时保守余量），超过授权的 18 小时；实际仍为 1.166519 小时，本次 GPU 增量为零。S6a 复用相同输入诊断，H3/H_ref 均为 1.000；原 0.95 / 0.99 门槛作废，原停止回执保留。ET-test 剩余 745 句、FMH S6b、两种 VAD 情绪推理、平滑和 S7 均未启动。完整结果与缺测项见 [REPORT.md](REPORT.md)，新凭据见 [results/revision_1_budget.json](results/revision_1_budget.json) 和 [results/revision_1_stop_audit.json](results/revision_1_stop_audit.json)；旧报告见 [REPORT_BEFORE_REVISION_1.md](REPORT_BEFORE_REVISION_1.md)。

协议快照在 `protocol/`；执行解释与兼容性问题在 `OPEN_QUESTIONS.md`。模型、融合、校准、中性偏置及拒识阈值均冻结。部署默认 H3，可选 H_ref；`calibration_bundle_v1.1.json` 仅改变 `gate_rules` 的字节区间。

执行主机为 `wsl`，新产物根目录为 `/mnt/h/emotionAI/emotioncore_runs/X06a`。从干净的仓库根目录运行，Python 使用 `/home/administrator/emotion-data/emotioncore_runs/X02/venv/bin/python`；`PYTHONPATH` 包含新根目录下的 `deps` 与仓库 `src`。依赖副本和模型缓存均留在 X06a 根目录，旧实验目录只读。

```bash
python -m emotioncore.x06.run --input /绝对路径/EmotionTalk片段.mp4 --system H3 --out events.jsonl
```

此实验入口接受冻结清单中的 FMH 或 ET-test 片段，拒绝其余来源。相对 `--out` 保存在 X06a 的 `offline_runs/` 下；绝对输出路径也必须位于 X06a 根目录。事件文件和同名 `.run` 目录内的运行凭据不得覆盖。`--realtime` 开启按原时间戳的 1 倍回放，默认全速。

修订后的顺序为启动预算检查 → ET-test 745 句续推并复用原 1,184 句 → FMH 正确切句与 S6b → 0.4 / 0.8 秒两种 S2 → S3 → 选定静音的 S5 → H3/H_ref 两轮 S7。最短静音只在 FMH 的 H3 情绪 macro-F1 上选择，0.005 内选 0.4；FSMN 与 0.3 阈值不变，原 `configs/x06_vad_freeze.json` 不覆盖。相同输入 S6a 已完成，不重跑。

本次仅新增 CPU 预算审计与停止封锁，没有执行新的续推流程。`python -m emotioncore.x06.revision_budget` 仅供独立新运行复现，既有回执不允许覆盖；全部 GPU 入口会在创建运行前拒绝当前预算停止状态。旧 S6 模块属于历史实现，不能当作修订后的 745 句续推入口直接重跑。

GPU 任务必须经过 `emotioncore.x06.gpu_guard`，累计整个子进程墙钟，包含初始化、CPU 处理和回放等待；修订总上限为 18 小时。已有 watchdog 脱离 SSH 会话但不自动启动后续阶段。当前预算停止未解除，CLI 接口存在不代表可以直接启动实验。

全速阶段共享 H_ref/H3 的共同分量；实时阶段独立执行所选系统所需的前向，不回放缓存概率。常驻 Q 使用 NF4，专家使用 D1_fmhx。静默事件使用 H_V/OP-R80，标记 `visual_only`，不进入主指标。

实时延迟按 VAD 判定结束到 JSONL 写出并 flush 计时。原时间戳按 1 倍前进；若处理落后，录制文件保留待处理帧，不丢帧，另报输入积压与墙钟/媒体时长比。解码及人脸分项包括端点判定之前已完成的工作，不能直接与端点后延迟相加。

所有事件、逐句结果、输入缓存与完整日志留在服务器；仓库交付聚合 JSON、报告以及原产物路径和 SHA-256。
