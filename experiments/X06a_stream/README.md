# X06a 连续流原型

**状态：S6 已触发 0.95 停止门禁，后续实验未运行。** 实际流检查处理 1,184/1,929 句后，H_ref 已有 97 处差异，其全体一致率乐观上界为 94.971488%。所有 GPU 入口均被停止状态阻断。完整结果、逐句差异和缺测项见 [REPORT.md](REPORT.md)，停止凭据见 [results/stop_delivery_audit.json](results/stop_delivery_audit.json)。以下入口和执行顺序描述实现接口，不表示已完成全部阶段或可在本轮直接续跑。

协议快照在 `protocol/`；执行解释与兼容性问题在 `OPEN_QUESTIONS.md`。模型、融合、校准、中性偏置及拒识阈值均冻结。部署默认 H3，可选 H_ref；`calibration_bundle_v1.1.json` 仅改变 `gate_rules` 的字节区间。

执行主机为 `wsl`，新产物根目录为 `/mnt/h/emotionAI/emotioncore_runs/X06a`。从干净的仓库根目录运行，Python 使用 `/home/administrator/emotion-data/emotioncore_runs/X02/venv/bin/python`；`PYTHONPATH` 包含新根目录下的 `deps` 与仓库 `src`。依赖副本和模型缓存均留在 X06a 根目录，旧实验目录只读。

```bash
python -m emotioncore.x06.run --input /绝对路径/EmotionTalk片段.mp4 --system H3 --out events.jsonl
```

此实验入口接受冻结清单中的 FMH 或 ET-test 片段，拒绝其余来源。相对 `--out` 保存在 X06a 的 `offline_runs/` 下；绝对输出路径也必须位于 X06a 根目录。事件文件和同名 `.run` 目录内的运行凭据不得覆盖。`--realtime` 开启按原时间戳的 1 倍回放，默认全速。

正式顺序是：CPU 预检与流构造 → 原始单句服务诊断 → AISHELL-4 train 冻结 VAD → test 报告与七条流切分 → 实际段数预算复核 → 实际流输入上的正确切句等价性 → FMH 正确切句和分段情绪推理 → FMH 冻结 EMA → H3/H_ref 各自四条 ET-test 实时回放。实际流等价性是后续 GPU 工作的硬性前置条件；原始单句诊断不能替代它。VAD 冻结文件已提交为 `configs/x06_vad_freeze.json`，后续正式运行检查其 SHA。

GPU 任务必须经过 `emotioncore.x06.gpu_guard`，它锁定单个 GPU 作业、累计全部进程墙钟时间，并在 15 小时或存储门禁触发时终止任务。`emotioncore.x06.launch` 将单个受限 watchdog 与 SSH 会话分离，避免连接中断移除预算监控；不自动启动下一阶段。

全速阶段共享 H_ref/H3 的共同分量；实时阶段独立执行所选系统所需的前向，不回放缓存概率。常驻 Q 使用 NF4，专家使用 D1_fmhx。静默事件使用 H_V/OP-R80，标记 `visual_only`，不进入主指标。

实时延迟按 VAD 判定结束到 JSONL 写出并 flush 计时。原时间戳按 1 倍前进；若处理落后，录制文件保留待处理帧，不丢帧，另报输入积压与墙钟/媒体时长比。解码及人脸分项包括端点判定之前已完成的工作，不能直接与端点后延迟相加。

所有事件、逐句结果、输入缓存与完整日志留在服务器；仓库交付聚合 JSON、报告以及原产物路径和 SHA-256。
