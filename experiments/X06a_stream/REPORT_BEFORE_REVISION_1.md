# X06a 连续流原型报告

## 1. 摘要

- 状态：S6 触发 0.95 停止门禁，实验已停止；X06a 未全部完成，不作验收通过声明。
- 实际流输入处理 1,184/1,929 句，H3/H_ref 分别有 92/97 处标签差异。
- 即使剩余 745 句全部一致，全体一致率也最多为 95.230689% / 94.971488%；H_ref 已不可能达到停止线。
- 移除全部门控后仍有 90/95 处差异，主要问题不能归结为新多脸规则。
- AISHELL test 帧级 F1 0.957709、边界 F1 0.400606；情绪切句损失、平滑及 S7 未运行。
- 参数包仅改 gate_rules，冻结模型未训练；20 项测试通过，全部 X06 进程已退出。
- 累计 GPU 用时 1.166519 小时（上限 15 小时）；报告、代码及停止凭据交由 reviewer 审核。

## 2. 第 0 步

已核对 X05 验收对象为 `d2036fb24d2e10df21de151f4806d26d57ed7bbb`，将 [PR #6](https://github.com/Xpeng1999/emotioncore-zh/pull/6) 转为 Ready，并于 2026-10-11 16:20:24（Asia/Shanghai，08:20:24 UTC）合入 `main`。合并提交为 `801320bc75f87aae736ae529c47180e8289fba30`，`x06-stream` 从该提交切出。凭据见 [step0.json](results/step0.json)。本轮 PR 不合入，等待 reviewer 验收。

## 3. 完成情况与偏离

| 工作 | 停止时状态 | 证据或限制 |
|---|---|---|
| S1 | 完成 | 3 条 FMH、4 条官方 ET-test，共 5,195 句 |
| S2 | VAD 选型、test 报告、切段和 2,944 段媒体准备完成 | 情绪推理未启动，切句损失未测 |
| S3 | 固定规则与 v1.1 已实现和测试 | 完整流的触发率及情绪指标未测 |
| S4 | 服务、生产者和 CLI 已实现 | 原始单句及实际流正确句界服务已运行；完整实时管线、CLI GPU 集成未完成 |
| S5 | 代码已实现，未运行 | 没有选择 α，也没有平滑冻结文件 |
| S6 | **门禁停止，1,184/1,929 句** | H_ref 的全体一致率乐观上界已低于 0.95 |
| S7 | 未运行 | 无实时延迟、完整稳定性或内存增长结论 |

所有新增执行代码位于 `emotioncore.x06`。未训练 Q、专家或 VAD；Q_mix 使用 `best_adapter_eval8`，专家使用 D1_fmhx。X02 人脸检测、关联、轨迹排序、裁剪操作和旧实验目录保持原样。新增的虚拟流输入层使用全局 5 fps 时钟；到达容器结尾但尚未到片段结束时，保持最后一帧的原始画面，避免 OpenCV 最近帧定位越过 EOF。

原 `equivalence_attempt2` 使用原 WAV 和从各句句首开始的抽帧，实际只是服务复现。发现范围不足后，补做 `stream_equivalence`，直接切取已冻结 S1 波形、读取全局抽帧，并将它设为后续 GPU 任务的前置条件。没有根据 ET-test 输出改变相位、对齐、阈值或模型。FMH 正确切句基线代码也改为相同流输入，但因 S6 停止没有运行，不能把预处理变化当成纯 VAD 切句损失。

2026-10-11 22:47:00.828898（Asia/Shanghai）记录 S6 停止。没有重试该检查或修订模型/参数来提高 ET-test 一致率；所有 GPU 入口现均在创建新运行前拒绝继续。停止与代码范围审计见 [stop_delivery_audit.json](results/stop_delivery_audit.json)，其中列出已完成阶段的 run_meta、路径和 SHA。

未用 MER-test、SIMS-valid；CH-SIMS test 零推理。AISHELL test 与 ET-test 仅报告。每个正式运行从干净提交启动，保存 `run_meta.json`、输入与输出 SHA-256。细化口径与变更登记见 [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md)。

## 4. 连续流构造与局限

| 说话人 | 用途 | 原句数 | 时长（小时） | VAD 语音段 | 静默窗口 |
|---|---|---:|---:|---:|---:|
| 06 | FMH | 965 | 1.417022 | 525 | 4 |
| 10 | FMH | 553 | 0.725395 | 261 | 1 |
| 11 | FMH | 1,748 | 2.605821 | 964 | 20 |
| 02 | ET-test | 414 | 0.677538 | 244 | 1 |
| 13 | ET-test | 379 | 0.542908 | 218 | 3 |
| 20 | ET-test | 525 | 0.833143 | 317 | 8 |
| 21 | ET-test | 611 | 0.933932 | 367 | 11 |

句序为 group、dialogue、utt_id；基础种子为 20261011，各说话人使用 seed + speaker_id。句间间隔独立均匀采样于 0.3–1.5 秒，噪声标准差取该句首尾各 100 ms 合并后的 RMS，静态画面保持前句最后一帧。真值区间使用原 MP4 时长，不把人工间隔算成真值。音频重采样为单声道 16 kHz，以采样点取整对齐并补齐或截断；逐句调整留在服务器构造凭据中。

全速处理与在线生产者使用同一全局 5 fps 取样和 X02 操作。七条流的在线 VAD 端点/静默窗口时序与全速切段完全一致；两句 FMH 的 CPU 集成检查还逐字节核对目标裁剪。

EmotionTalk 没有原始整段录像。拼接间隔比真实停顿干净，VAD 在这里偏容易；拼接产生的表情和场景跳变也不代表真实连续情绪。真实录音的切句质量应看 AISHELL-4，而非只看模拟流。原始 WAV 与容器时长不同，因此原始单句诊断也不能代替实际 S1 输入测试。

## 5. VAD 选型、AISHELL-4 结果与切句损失

只用第 0 通道；依据 RTTM 排除重叠区域，保留无语音帧为负类，在重叠边缘断开模型/切段状态。比较预训练 Silero VAD 6.2.3 与 FSMN-VAD；语音阈值取预登记的 {0.3, 0.5, 0.7}，最短静音为 {0.2, 0.4, 0.8} 秒，最短语音 0.3 秒、最长 15 秒。Silero 使用 512 样本块，FSMN 使用原在线前端的一秒输入块；两者后验交给同一切段状态机，不混用 FSMN 额外的原生启发式参数。

| 模型与参数 | AISHELL train 帧 F1 | train 边界 F1 |
|---|---:|---:|
| Silero 最优：阈值 0.3、静音 0.8 秒 | 0.847519 | 0.363651 |
| FSMN：阈值 0.3、静音 0.4 秒 | 0.940421 | 0.540495 |
| FSMN 最优并选中：阈值 0.3、静音 0.8 秒 | 0.946247 | 0.511194 |

0.4 秒配置距最优超过 0.005，不适用更短静音优先的并列规则。选型结果先保存并以提交 `88f706c` 冻结为 `configs/x06_vad_freeze.json`，SHA-256 为 `43cb46bc84417b9bff5cb20c7ab1edaa370ee98ccfaa21cfae4150d637cd6714`，随后才运行 AISHELL test。

| AISHELL test 指标 | 结果 |
|---|---:|
| 帧级 Precision / Recall / F1 | 0.940462 / 0.975600 / 0.957709 |
| ±0.25 秒边界 Precision / Recall / F1 | 0.553883 / 0.313774 / 0.400606 |
| 段数 / 平均段长 | 5,614 / 7.278630 秒 |

帧级 F1 较高不等于句子边界准确。边界评估包括重叠区间边缘与 15 秒强制切分，不能当成自然语义句界质量。全网格与数据凭据见 [vad_train.json](results/vad_train.json)、[vad_test.json](results/vad_test.json)。

由于 S6 停止，情绪切句损失未运行，不能判断 FMH 损失是否 ≤ 0.03。S2 实现先对所有重叠 VAD 段的各分量概率按重叠时长平均，再使用冻结参数包；无覆盖句记拒识并留在召回分母。全部句 macro-F1、接受子集 macro-F1、覆盖率、切碎率和合并率均没有情绪推理结果，不能以缺测记为通过。

## 6. 多脸规则与 v1.1 参数包

若第二条轨迹的目标分数 ≥ 第一条的 0.5 倍，拒识为 `ambiguous_multi_face`；否则只使用第一条。无脸但有语音走 H_A；有脸且无语音超过 2 秒，按每个 2 秒窗口执行 H_V、OP-R80，并标记 `visual_only`。默认完整系统 H3，H_ref 可选。

v1 SHA-256：`94a0805b9775676209306187a3880595b7e7592af70aaec8d50f3b2a7968979a`。v1.1 SHA-256：`e73befc23e833b25fdd26d0784368fedf14afc6aef86c0378865f0958455a5f7`。升级函数只替换顶层 `gate_rules` 值的原始字节区间，其余字节完全不动；外层 version 字段也保持原样。融合、校准、中性偏置和拒识阈值未重拟合。

原始单句 ET-test 诊断中有 23/1,929 句触发新规则，其中两系统各有 21 句原本在 X05 R90 下接受。这是原始单句诊断口径，不是全流 VAD 触发率。FMH 与 ET-test 全流的触发率及相应情绪指标未运行，不能判断 FMH 的 ≤ 5% 验收项。已完成的 FMH 原始单句人脸缓存属于准备资料，不替代真实流指标。

## 7. 等价性

正式 S6 使用实际 S1 波形与全局取样，在原句真值边界切段；比较 X05 部署逐句概率经过冻结偏置和 R90 阈值后的标签，包含拒识，计划分母为全部 1,929 句。处理到 1,184 句时，H_ref 累计 97 处差异，已经使“剩余全对”的全体一致率上界低于 0.95，故立即停止。其余 745 句没有推理；下面的前缀一致率不是完整 ET-test 一致率。

| 指标 | H3 | H_ref |
|---|---:|---:|
| 已处理 / 计划句数 | 1,184 / 1,929 | 1,184 / 1,929 |
| 已观察标签差异 | 92 | 97 |
| 已处理前缀一致率 | 0.922297 | 0.918074 |
| 全体一致率乐观上界（余下全部一致） | ≤ 0.952307 | **≤ 0.949715** |
| 仅移除多脸规则后的差异 | 90 | 95 |
| 移除全部门控后的差异 | 90 | 95 |
| 完整总体一致率 | 未完成，不能给出 | 未完成，不能给出 |

两系统各只有 2 处主差异在移除多脸规则后恢复；另有 90/95 处涉及流输入变化后的类别或 R90 阈值决策变化。原始单句诊断的无门控输出与 X05 完全一致，而实际流检查使用不同的音频对齐和抽帧时钟。现有对比没有分离这两项，**不能把每处变化进一步归因于单独的音频或视频因素**，也没有在停止后追加模型消融。

证据：[stream_equivalence.json](results/stream_equivalence.json)、[stream_equivalence_audit.json](results/stream_equivalence_audit.json)。完整逐句诊断位于服务器 `/mnt/h/emotionAI/emotioncore_runs/X06a/stream_equivalence_audit/mismatches.json`，SHA-256 `e1b513929791cdcfd233b6f8d93e16c57167adb542094917fdeefb3f28c88fb7`；其中保留新旧置信度、未拒识类别、分量概率最大变化及轨迹分数。

以下逐条列出所有主差异，共 189 个“句子 × 系统”差异，涉及 140 个句子。“类别改变”比较偏置后的未拒识类别；“跨 R90 阈值”表示类别相同但接受状态改变；这些是决策层表现，不是已证实的音频/视频根因。横线表示该系统在该句与 X05 一致。

| 句子 ID | H3：X05 → 实际流；表现 | H_ref：X05 → 实际流；表现 |
|---|---|---|
| G00003_11_02_006 | neutral → reject；跨 R90 阈值 | neutral → reject；跨 R90 阈值 |
| G00003_11_02_008 | neutral → surprise；类别改变 | reject → surprise；跨 R90 阈值 |
| G00003_11_02_018 | — | reject → surprise；跨 R90 阈值 |
| G00003_11_13_023 | — | reject → surprise；跨 R90 阈值 |
| G00003_12_02_008 | neutral → surprise；类别改变 | neutral → surprise；类别改变 |
| G00003_12_02_010 | reject → neutral；跨 R90 阈值 | reject → neutral；类别改变 |
| G00003_16_02_015 | — | reject → anger；跨 R90 阈值 |
| G00003_17_02_003 | neutral → reject；跨 R90 阈值 | — |
| G00003_17_02_005 | — | neutral → reject；跨 R90 阈值 |
| G00003_17_02_023 | — | reject → anger；跨 R90 阈值 |
| G00003_17_13_014 | neutral → reject；跨 R90 阈值 | — |
| G00003_17_13_020 | — | reject → anger；跨 R90 阈值 |
| G00003_17_13_022 | anger → reject；类别改变 | — |
| G00003_18_13_012 | reject → anger；跨 R90 阈值 | — |
| G00003_19_13_027 | — | neutral → reject；跨 R90 阈值 |
| G00003_22_02_004 | fear → reject；多脸拒识，移除后恢复 | fear → reject；多脸拒识，移除后恢复 |
| G00003_22_13_007 | — | reject → anger；跨 R90 阈值 |
| G00003_23_02_006 | reject → fear；跨 R90 阈值 | — |
| G00003_23_02_012 | neutral → reject；跨 R90 阈值 | — |
| G00003_23_02_016 | reject → anger；跨 R90 阈值 | — |
| G00003_24_02_026 | neutral → anger；类别改变 | neutral → anger；类别改变 |
| G00003_24_13_005 | — | reject → joy；跨 R90 阈值 |
| G00003_24_13_008 | reject → anger；类别改变 | reject → anger；跨 R90 阈值 |
| G00003_24_13_011 | anger → reject；跨 R90 阈值 | — |
| G00003_24_13_034 | reject → joy；跨 R90 阈值 | — |
| G00003_25_02_032 | surprise → reject；跨 R90 阈值 | surprise → reject；跨 R90 阈值 |
| G00003_25_02_050 | reject → sadness；跨 R90 阈值 | fear → reject；跨 R90 阈值 |
| G00003_25_13_015 | reject → surprise；类别改变 | — |
| G00003_25_13_019 | reject → anger；跨 R90 阈值 | reject → anger；跨 R90 阈值 |
| G00003_25_13_029 | anger → reject；跨 R90 阈值 | anger → reject；类别改变 |
| G00003_25_13_035 | — | anger → reject；跨 R90 阈值 |
| G00003_25_13_039 | surprise → reject；类别改变 | surprise → reject；类别改变 |
| G00003_26_02_009 | reject → neutral；跨 R90 阈值 | reject → neutral；跨 R90 阈值 |
| G00003_26_02_017 | anger → fear；类别改变 | — |
| G00003_26_13_004 | — | neutral → reject；跨 R90 阈值 |
| G00003_27_02_027 | — | reject → anger；跨 R90 阈值 |
| G00003_27_02_028 | — | reject → fear；类别改变 |
| G00003_27_02_031 | reject → fear；跨 R90 阈值 | reject → fear；跨 R90 阈值 |
| G00003_27_02_035 | anger → reject；类别改变 | anger → reject；跨 R90 阈值 |
| G00003_27_13_004 | reject → neutral；类别改变 | — |
| G00003_27_13_010 | reject → surprise；类别改变 | — |
| G00003_27_13_024 | reject → fear；跨 R90 阈值 | — |
| G00003_27_13_030 | — | surprise → reject；类别改变 |
| G00003_30_13_006 | joy → sadness；类别改变 | — |
| G00003_31_02_006 | anger → reject；跨 R90 阈值 | — |
| G00003_31_13_003 | sadness → neutral；类别改变 | — |
| G00003_33_02_013 | — | reject → joy；跨 R90 阈值 |
| G00003_33_02_015 | joy → reject；跨 R90 阈值 | — |
| G00003_33_02_027 | — | reject → neutral；跨 R90 阈值 |
| G00003_33_02_033 | — | joy → reject；跨 R90 阈值 |
| G00003_33_13_020 | — | reject → joy；跨 R90 阈值 |
| G00003_33_13_030 | joy → reject；跨 R90 阈值 | — |
| G00003_33_13_036 | joy → reject；跨 R90 阈值 | — |
| G00003_34_13_023 | neutral → joy；类别改变 | — |
| G00003_34_13_034 | reject → joy；跨 R90 阈值 | — |
| G00003_35_02_011 | neutral → reject；类别改变 | — |
| G00003_35_02_022 | reject → joy；跨 R90 阈值 | reject → joy；跨 R90 阈值 |
| G00003_35_13_005 | — | joy → neutral；类别改变 |
| G00003_36_02_015 | — | reject → joy；跨 R90 阈值 |
| G00003_38_02_013 | surprise → fear；类别改变 | surprise → fear；类别改变 |
| G00003_38_02_017 | reject → fear；跨 R90 阈值 | reject → fear；跨 R90 阈值 |
| G00003_38_02_021 | — | fear → reject；跨 R90 阈值 |
| G00003_38_13_012 | reject → surprise；类别改变 | — |
| G00003_41_02_031 | reject → neutral；跨 R90 阈值 | reject → neutral；跨 R90 阈值 |
| G00003_42_02_007 | neutral → reject；类别改变 | neutral → surprise；类别改变 |
| G00003_42_02_027 | surprise → reject；跨 R90 阈值 | surprise → anger；类别改变 |
| G00003_43_02_015 | reject → anger；类别改变 | reject → anger；类别改变 |
| G00003_43_02_025 | anger → disgust；类别改变 | anger → disgust；类别改变 |
| G00003_43_13_005 | — | reject → neutral；跨 R90 阈值 |
| G00003_43_13_016 | — | reject → anger；跨 R90 阈值 |
| G00003_44_02_004 | anger → reject；跨 R90 阈值 | — |
| G00003_44_02_006 | — | anger → reject；跨 R90 阈值 |
| G00003_44_02_011 | — | reject → anger；跨 R90 阈值 |
| G00003_44_02_016 | — | anger → reject；跨 R90 阈值 |
| G00003_45_02_008 | disgust → neutral；类别改变 | reject → neutral；类别改变 |
| G00003_46_02_018 | disgust → reject；跨 R90 阈值 | disgust → reject；跨 R90 阈值 |
| G00003_46_02_033 | surprise → reject；跨 R90 阈值 | surprise → reject；跨 R90 阈值 |
| G00003_46_02_039 | — | reject → disgust；跨 R90 阈值 |
| G00003_46_13_013 | reject → neutral；跨 R90 阈值 | — |
| G00003_46_13_038 | reject → anger；跨 R90 阈值 | — |
| G00003_47_13_031 | — | reject → surprise；跨 R90 阈值 |
| G00003_50_13_018 | surprise → anger；类别改变 | surprise → anger；类别改变 |
| G00003_54_02_013 | — | reject → sadness；类别改变 |
| G00003_54_13_007 | surprise → reject；跨 R90 阈值 | — |
| G00003_56_02_003 | reject → neutral；跨 R90 阈值 | — |
| G00003_56_02_017 | reject → sadness；跨 R90 阈值 | reject → sadness；类别改变 |
| G00015_02_20_017 | joy → neutral；类别改变 | — |
| G00015_04_20_020 | joy → reject；跨 R90 阈值 | — |
| G00015_05_20_013 | neutral → reject；类别改变 | neutral → reject；类别改变 |
| G00015_05_20_030 | — | reject → disgust；跨 R90 阈值 |
| G00015_05_20_032 | — | reject → anger；类别改变 |
| G00015_06_20_023 | anger → reject；跨 R90 阈值 | — |
| G00015_06_20_031 | disgust → reject；跨 R90 阈值 | — |
| G00015_07_20_002 | disgust → reject；跨 R90 阈值 | — |
| G00015_08_20_023 | disgust → reject；跨 R90 阈值 | disgust → reject；跨 R90 阈值 |
| G00015_08_20_026 | — | neutral → reject；跨 R90 阈值 |
| G00015_08_20_030 | reject → disgust；跨 R90 阈值 | reject → disgust；跨 R90 阈值 |
| G00015_11_20_015 | — | neutral → reject；跨 R90 阈值 |
| G00015_11_20_016 | disgust → reject；跨 R90 阈值 | — |
| G00015_11_20_020 | neutral → reject；类别改变 | neutral → reject；类别改变 |
| G00015_11_20_026 | disgust → anger；类别改变 | reject → anger；跨 R90 阈值 |
| G00015_11_20_029 | reject → joy；类别改变 | reject → joy；跨 R90 阈值 |
| G00015_12_20_003 | neutral → reject；类别改变 | neutral → reject；类别改变 |
| G00015_12_20_011 | neutral → anger；类别改变 | — |
| G00015_12_20_017 | — | reject → neutral；类别改变 |
| G00015_12_20_022 | disgust → anger；类别改变 | — |
| G00015_13_20_007 | — | reject → disgust；跨 R90 阈值 |
| G00015_13_20_012 | — | anger → disgust；类别改变 |
| G00015_13_20_020 | — | neutral → reject；跨 R90 阈值 |
| G00015_15_20_001 | — | reject → surprise；跨 R90 阈值 |
| G00015_15_20_009 | neutral → reject；类别改变 | neutral → reject；类别改变 |
| G00015_17_20_009 | anger → reject；跨 R90 阈值 | anger → reject；跨 R90 阈值 |
| G00015_18_20_010 | reject → neutral；跨 R90 阈值 | reject → neutral；跨 R90 阈值 |
| G00015_18_20_016 | reject → fear；跨 R90 阈值 | — |
| G00015_18_20_018 | — | anger → reject；跨 R90 阈值 |
| G00015_19_20_004 | neutral → reject；类别改变 | neutral → reject；跨 R90 阈值 |
| G00015_19_20_018 | reject → fear；跨 R90 阈值 | — |
| G00015_19_20_025 | neutral → reject；跨 R90 阈值 | neutral → reject；跨 R90 阈值 |
| G00015_19_20_037 | fear → sadness；类别改变 | fear → sadness；类别改变 |
| G00015_19_20_041 | reject → fear；跨 R90 阈值 | reject → fear；跨 R90 阈值 |
| G00015_20_20_008 | — | reject → joy；跨 R90 阈值 |
| G00015_21_20_001 | — | reject → neutral；跨 R90 阈值 |
| G00015_21_20_004 | — | neutral → reject；跨 R90 阈值 |
| G00015_22_20_018 | — | reject → disgust；跨 R90 阈值 |
| G00015_24_20_012 | — | neutral → reject；跨 R90 阈值 |
| G00015_25_20_007 | reject → joy；跨 R90 阈值 | reject → joy；跨 R90 阈值 |
| G00015_26_20_007 | reject → sadness；跨 R90 阈值 | — |
| G00015_27_20_025 | joy → neutral；类别改变 | joy → neutral；类别改变 |
| G00015_29_20_003 | — | reject → anger；跨 R90 阈值 |
| G00015_29_20_013 | reject → anger；跨 R90 阈值 | — |
| G00015_30_20_003 | joy → reject；多脸拒识，移除后恢复 | joy → reject；多脸拒识，移除后恢复 |
| G00015_31_20_002 | joy → neutral；类别改变 | — |
| G00015_31_20_044 | reject → neutral；跨 R90 阈值 | reject → neutral；类别改变 |
| G00015_31_20_067 | reject → anger；类别改变 | reject → anger；跨 R90 阈值 |
| G00015_32_20_005 | neutral → surprise；类别改变 | neutral → surprise；类别改变 |
| G00015_35_20_001 | — | reject → fear；跨 R90 阈值 |
| G00015_35_20_013 | — | reject → sadness；类别改变 |
| G00015_35_20_015 | reject → fear；跨 R90 阈值 | — |
| G00015_35_20_019 | — | reject → fear；跨 R90 阈值 |
| G00015_36_20_016 | reject → fear；跨 R90 阈值 | reject → fear；跨 R90 阈值 |

原始单句诊断（完整 1,929 句）保留如下，不能替换正式 S6：

| 比较口径 | H3 | H_ref |
|---|---:|---:|
| 带新多脸规则及路由 | 0.989114（21 句差异） | 0.989114（21 句差异） |
| 仅移除多脸规则，保留 H_A 回退 | 0.999482（1 句差异） | 0.999482（1 句差异） |
| 移除全部门控、按完整系统决策 | 1.000000 | 1.000000 |

| 原始单句诊断 ID | H3：X05 → 新规则 | H_ref：X05 → 新规则 | 原因 |
|---|---|---|---|
| G00003_12_13_021 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00003_22_02_004 | fear → reject | fear → reject | ambiguous_multi_face |
| G00015_30_20_003 | joy → reject | joy → reject | ambiguous_multi_face |
| G00015_30_20_005 | joy → reject | joy → reject | ambiguous_multi_face |
| G00015_31_21_034 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_013 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_015 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_016 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_019 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_020 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_021 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_022 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_024 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_32_21_028 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_33_21_001 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_35_21_007 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_35_21_009 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_35_21_010 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_38_20_008 | neutral → reject | neutral → reject | ambiguous_multi_face |
| G00015_39_20_007 | fear → reject | fear → reject | ambiguous_multi_face |
| G00015_39_21_025 | neutral → reject | neutral → reject | ambiguous_multi_face |

该原始单句诊断的 21 句差异均在移除多脸规则后恢复。反事实中的另一个差异 `G00015_32_21_023` 不在这 21 句中：它无可用 Q 人脸裁剪，主检查双方均拒识；仅移除多脸规则才触发 H_A 并输出 neutral。不能将两个口径的差异句集合混为一谈。路径、SHA 和逐句归因见 `equivalence_audit_v2` 的服务器凭据，聚合入口为 [equivalence_audit.json](results/equivalence_audit.json)。

## 8. 平滑

因 S6 停止，未运行 α 选择；未运行不等于 α=0。候选为 {0, 0.2, 0.4}，只在 H3 的 FMH 流上按 macro-F1 选择，0.005 内选更小值；H_ref 复用该 α。所有偏置和阈值固定。

S5 先逐段校准、再 EMA，与 S2 先聚合各分量、再校准的顺序不同，二者不可交换。实现单列 S5 的 α=0 同顺序对照，避免把聚合顺序变化误报成平滑收益。语音拒识段仍推进 EMA，静默窗口不推进；中性误报和相邻标签翻转率没有本轮结果。

## 9. 延迟与资源

S7 尚未运行，无端到端 P50/P95/P99、实测实时率、完整运行内存增长或在线瓶颈排序。原始单句服务计时与 CPU 人脸准备耗时只用于预算，不作为 S7 延迟。

实现按 VAD 实际判定结束到事件 JSONL 写出并 flush 计时；另报输入时间戳积压和预定端点到写出的时间。若处理落后，录制文件保留全部帧，不丢帧；不能据此声称处理速度已跟上 1 倍实时。解码和人脸分项可能在端点前完成，Q 总耗时也包含其预处理和视频解码，因此这些分项不可重复相加。RSS 范围为主进程及活跃子进程，基线在模型加载后、输入开始前，结束值保留生产者退出前的驻留内存，峰值为逐事件/一秒等待时采样值。

首次预算估计 13.611545 GPU 小时；实际 VAD 段数复核为 12.747556 小时；补充实际流正确切句后为 14.689533 小时。计费口径为 GPU 子进程全部墙钟时间，包括模型初始化、CPU 处理和实时回放等待。最终累计 **4,199.466696 秒 = 1.166519 GPU 小时**，其中初始化失败也完整计费；没有训练。

| GPU 作业 | 状态 | 计费秒数 |
|---|---|---:|
| 原始单句服务第一次初始化 | timm 缺失，预测前失败 | 51.234218 |
| 原始单句服务诊断重试 | 完成 1,929 句 | 2,144.709704 |
| 实际流正确切句 S6 | 1,184 句后按门禁停止 | 2,003.522774 |
| 合计 | 低于 15 小时上限 | 4,199.466696 |

执行主机为 wsl，GPU 为 RTX 5090。停止审计确认没有活跃 X06 进程，GPU 作业锁已释放，VAD 情绪推理、FMH 实际流正确切句基线、平滑和两轮实时回放目录均未创建。20 项协议测试通过，其中新增检查确保 S6 停止后所有 GPU 入口均无法创建新作业。完整账本见 [stop_delivery_audit.json](results/stop_delivery_audit.json)，测试凭据见 [protocol_tests.json](results/protocol_tests.json)。

## 10. 异常

- 初次预检按说话人选样时包含了官方 valid 片段，1,929 句数量断言在任何模型推理前拦截；随后固定为官方 test。失败目录保留。
- 第一次 GPU 服务在加载 V1 时缺少 timm，尚未产生预测；失败的 51.234218 秒计入 GPU 账本。将旧视觉环境中 timm 0.9.16 的文件逐一核对 SHA 后复制到 X06a 独立依赖目录，CPU 加载通过，再开始新的等价性运行；未修改旧环境。
- AISHELL train 串行执行完成 35 条后改用八进程，仅复用已完整保存且核对过的概率缓存；旧运行标记为 `superseded_cpu_execution`，选型只在完整 130 条结果汇总后进行。
- CPU 集成发现容器最后不足一帧区间的 OpenCV 定位越过 EOF；修复虚拟源的末帧保持，X02 检测和裁剪操作未改。选定 FSMN 的两句集成复测通过，所有裁剪字节一致。
- 原始单句等价性检查范围不足，已明确降为诊断并补做实际流检查；未以其 98.911% 替代正式 S6。
- 为优先完成 S6 输入，暂缓后恢复下游 CPU VAD 媒体准备，保存进程身份与暂停/恢复凭据；2,944 段媒体准备在停止前已全部完成，但没有进入情绪推理。
- 实际流 S6 于 2026-10-11 14:47:00.828898 UTC 记录停止，GPU 子进程以预期的非零码退出；这是协议中止，不是 OOM 跳过。未发生 OOM 跳过，没有执行后续实验。停止后的工作仅包括 CPU 结果审计、测试、报告和 PR。

## 11. 给 reviewer 的问题

- 实际流 S6 已触发 0.95 停止条件。请先审查真实流输入与 X05 参考输入的对应关系，以及波形对齐、全局抽帧和原始单句抽帧的定义；本轮没有根据 ET-test 指标选择替代处理方式。
- 新多脸规则与 0.99 验收线的兼容性仍需单独判断，但它只能解释本次已处理部分每系统的 2 处差异，不能据此豁免其余差异。
- 下一轮若授权修复与复测，应明确开发集、允许改动的输入/人脸接口和剩余 GPU 账本；当前没有继续运行。切句损失、平滑与实时延迟仍未交付，不能作为已通过的 X06a 或已测得的 X06b 性能基线。
