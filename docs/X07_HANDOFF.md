# X07 交接：X06b 配置、入口与边界

当前状态：**执行中，尚未全部交付**。PR #8 保持未合入；测量完成与 reviewer 验收分别判断。完整结果见 [X06b 报告](../experiments/X06b_realtime/REPORT.md)。X06a 有条件通过，PR #7 已合入 main；历史未达标及停止回执保存在 [X06a 报告](../experiments/X06a_stream/REPORT.md)。

## 固定部署配置

| 项目 | 配置 |
| --- | --- |
| 全模态系统 | 明确传入 H3；本轮不测 H_ref |
| Q | X04b Q_mix，best_adapter_eval8 |
| 专家 | X05 D1_fmhx A3/V2/V1，权重及特征编码器原样复用 |
| 校准/拒识 | calibration_bundle_v1.1.json，语音 R90，静默视觉 R80 |
| VAD | FSMN，阈值 0.3，最短静音 0.4 秒 |
| 平滑 | α=0 |
| 解码 | 全局 5 fps，顺序读取必要原生帧，只取出需要帧；间隔保持末帧 |
| 检测 | 原 YuNet，输入缩小至 1/2；8 个独立检测线程，16 帧在途上界 |
| 跟踪/裁剪 | 原 X02 关联、目标排序、JPEG 与裁剪规则；裁剪取原分辨率帧 |
| 多脸 | 第二轨迹分数 ≥ 目标轨迹 0.5 倍时拒识 |
| 实时暂存 | 原始帧缓冲硬上界 32；超限或缺帧立即失败 |

人脸配置在 FMH 上选定，于 `958da60` 提交后才开始 R3–R5。其 SHA-256 为 `5617c5ae119af1ef491ab0b404350a05afe16b3261d48bec56b9013c7ae5aafa`。参数包及三份 VAD/平滑配置的原始 SHA 见 [登记文件](../configs/x06b_registration.json)，模型路径及 SHA 见参数包 components。参数包历史顶层 default_full_system 字段保持不变；入口显式传入 H3，并遵守 gate_rules 的 H_A/H_V 回退。

## 离线与实时入口

执行环境为服务器 RTX 5090 与既有 X02 venv；新代码目录为 `src/emotioncore/x06b`，旧模块只读复用。配置、输入、Git 状态、预算必须通过检查，每次运行写 run_meta，拒绝覆盖现有输出。

- 全速输入：`python -m emotioncore.x06b.prepare r3 fmh` 等；固定集合仅 fmh / et_test，R4 对应 r4_correct / r4_vad。
- 全速推理：`python -m emotioncore.x06b.gpu_guard r3_fmh`，以及 r3_et_test、r4_correct_fmh、r4_correct_et_test、r4_vad_fmh、r4_vad_et_test；不得直接绕过 watchdog 调用 infer。
- 实时回放：`python -m emotioncore.x06b.gpu_guard r5`，仅 X06a 同样的四条 ET-test 流，1 倍输入，真实 H3 前向。
- 本轮完整顺序由 `python -m emotioncore.x06b.stages --launch` 启动一次；每阶段后重估预算，后续实验不应重复调用本轮已完成目录。
- 审计：`python -m emotioncore.x06b.delivery`；原始逐句/事件、输入及曲线数据留 `/mnt/h/emotionAI/emotioncore_runs/X06b`，仓库记录摘要、路径与 SHA。

GPU 上限 14 小时，按子进程完整墙钟计费，包含初始化、CPU等待及实时等待。预算未来工作加20%及300秒；R4 GPU 前以实际冻结 VAD 段数取代估计。ET-test 仅报告；MER-test、SIMS-valid 不用，CH-SIMS test 零推理。

## 测量与已知局限

- R3 FMH OP-N F1 差 -0.005144，95% CI [-0.011781240570322375, 0.0016558892232379546]；非劣条件达到：True。

R4 对话时序流假设其他说话人的声音没有进入目标麦克风；其他句子只提供间隔时长，不渲染或推理。拼接流仍不能替代自然连续录音。X06a 短停顿流保留为最坏参考。

延迟以输出写出为终点，分别从实际 VAD 判定、计划 VAD 端点和真值句结束计算。一个真值句分成多事件时取最后贡献事件；未覆盖句单列。墙钟含末事件排空，GPU 总账另含初始化。内存基线固定为输入启动后第600秒首次采样，结束时补回生产者退出前 RSS；X06a 的基线口径不同，不能直接相减。

当前入口面向登记的实验流，并非任意视频上传服务。后续阶段不得隐式改动模型、校准、VAD 或人脸配置，也不得根据 ET-test 结果选参。

本轮已依 X06B_RESUME.md 获授权恢复。旧 ET-test 报表异常与原 STOP、失败 run_meta 保留；CPU 重试目录为 r3_analysis_et_test_retry1，恢复回执为 resume_authorization，续跑驱动为 stages_resume1。无需且不得重复启动原一次性驱动或重跑 R2/R3。后续顺序为 R4 → AISHELL-4 → R5 → 曲线 → 最终审计。新停止仍会拦截；恢复入口也只允许启动一次。详见 [恢复记录](../experiments/X06b_realtime/STOP_RECOVERY.md)。
