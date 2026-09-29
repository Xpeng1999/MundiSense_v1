# X03 融合架构对照：执行中报告

## 1. 摘要

X02 第 0 步已补齐。Reviewer 已将 Q 的 GPU 上限提高至 55 小时，并批复固定 700 句内验证、43 条缺脸回退、T1r SIMS-valid 补做和取消 P 臂。两份新配置在 fold 0 前冻结 SHA-256。Q fold 0 已完成，最佳内验证 AV macro-F1 **0.6966**；按实测重估总 GPU 时长 **50.51 小时**，通过前两项门禁。最长 20 条 AV 输入的显存门禁仍在准备缓存。E1/E2 已完成；尚无跨臂结论。

## 2. 第 0 步完成情况

- X02 代码经 `x02-unimodal` PR #2 合入 `main`，合并提交 `1969a90ce72c1fe93b506b2b85420452338bddb0`；X03 从该提交切出 `x03-fusion`。
- [X02 补充说明](../X02_unimodal/ADDENDUM.md) 重算 CER：CV 池原口径 0.434，新口径 0.293；全部 19,250 句新口径 0.282。句级质量文件位于服务器 `/home/administrator/emotion-data/emotioncore_runs/X02/quality/utterance_quality.jsonl`，SHA-256 `80330bb718e31a8155793ab83e396bd4741a87347858f0342fd039cdb098bb32`，疑似截断 4,121 句。X02 原报告正文未改。
- X02 的 37 个 bundle 和 21 个临时 worktree 已移到 `/home/administrator/emotion-data/emotioncore_runs/X02/archive`，原工作树路径以符号链接保持可读。工作树通过 `git worktree move` 更新注册；reviewer 已接受该做法。未删除文件。

## 3. 完成情况与偏离

已完成 X02 交付、E1 五组合及 T1r SIMS-valid、ET 19,250 条人脸视频预处理、Q 模态冒烟和预算预检。固定内验证 [x03_inner_val_subsets.json](../../configs/x03_inner_val_subsets.json)（SHA-256 `be2cd442346cfa978e22bea8ab2ae3776d52187b2581fe93e377fc4fb58d3cfd`）按七类分层，每折从内验证说话人抽取 700 句，种子 20260929。固定 [x03_face_missing_policy.json](../../configs/x03_face_missing_policy.json)（SHA-256 `ce65415b5f7129b62ef5edc7eda5b20874ed88931f5a7cc295c276b8b704b77c`）规定训练 V/AV 缺脸改 A；评估 V 均匀七类并强制计错，AV 仅音频回退；E/H 画面专家均匀。内验证仍每 0.25 epoch 只评 AV。其余折数、2 epoch 上限、学习率、LoRA 和 60/20/20 形式比例不变。

Q fold 0 已自然结束：两轮共处理 19,652 个训练样本，完成 8 次固定 700 句 AV 内验证；第 5 次验证得到最佳 macro-F1 0.6966，超过 0.45 门限。总 GPU 时长按 fold 0 实测重估为 50.51 小时，低于 55 小时门限。E2 CPU 分支已完成；H 与完整主表待 Q OOF。

新增显存保护仅从 fold 1 起生效，fold 0 全程保持原样。后续启动器已通过前两项门禁，正在缓存最长 AV 输入候选；显存预检通过前不启动 fold 1。E2 全程限定 CPU。

## 4. 校准与工作点方法

E1 每个留出折用其余四折的 OOF 概率拟合非负 log 概率权重与 7 类偏置；同一拟合集拟合温度与校准偏置，再只用拟合集确定 OP-N 的中性偏置。OP-A 直接 argmax。最终 ET-test/SIMS-valid 参数使用全部五折 OOF 拟合；两者标签未参与拟合。冻结折文件 SHA-256 为 `85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce`。防泄漏测试修改留出折标签后，留出折参数及预测不变。

Benchmark v1 的字段 `neutral_false_positive_rate` 实际为 **真实中性句被预测为非中性 / 真实中性句**。为保持冻结口径，OP-N 在拟合集上把此比例调至最接近 0.15 的可实现值；留出折合并后的比率可偏离 0.15。ECE 用 15 个等宽分箱。完整逐类指标和混淆矩阵见 `results/E1_*_metrics.json`。

## 5. 臂 E

E1 使用 X02 冻结概率，未重新提取特征；每组均覆盖 CV 16,366 句、ET-test 1,929 句且 ID 唯一。下表仅展示关键数值；其他指标在 JSON。

| E1 组合 | CV OP-A F1 | CV OP-N F1 | CV OP-N 中性失配率 | ET-test OP-N F1 | SIMS-valid Pearson |
|---|---:|---:|---:|---:|---:|
| A3 | 0.552 | 0.540 | 0.155 | 0.546 | 0.657 |
| A3+V2 | 0.595 | 0.587 | 0.161 | 0.592 | 0.746 |
| A3+V2+V1 | 0.601 | 0.596 | 0.159 | 0.593 | 0.749 |
| A3+V2+T1r | 0.627 | 0.617 | 0.157 | 0.625 | 0.773 |
| A3+A2+V2+V1+T1r | 0.639 | 0.631 | 0.157 | 0.626 | 0.777 |

按预定 OP-N CV macro-F1，最后一组是 **E1 臂内最高**。X02 未提供 T1r 的 SIMS-valid 预测；reviewer 允许补做。SenseVoice 只处理 SIMS-valid 音频，成功 620/647，另外 27 条源音频缺失；冻结的 T1 最终模型已完成 r-token valid 推理，含 T1r 的两组已补齐此指标。

E2 的五折特征级融合、全 CV 最终模型、ET-test、SIMS-valid、逐折校准和缺失模态评估均已完成。全部 E2 训练及推理使用 CPU，未占用 Q 的 GPU。下表为已冻结的 E2 指标；完整 OP-A/OP-N 指标、逐类结果及服务器概率文件的路径与 SHA-256 见 [AVT](results/E2_AVT_metrics.json)、[AV](results/E2_AV_metrics.json)、[A](results/E2_A_metrics.json)、[V](results/E2_V_metrics.json)。SIMS-valid 的相关系数基于有源音频的 620/647 句。

| E2 条件 | CV OP-A F1 | CV OP-N F1 | ET-test OP-N F1 | SIMS-valid Pearson |
|---|---:|---:|---:|---:|
| A+V+T | 0.608 | 0.603 | 0.576 | 0.743 |
| A+V | 0.563 | 0.558 | 0.512 | 0.714 |
| A | 0.519 | 0.513 | 0.518 | 0.604 |
| V | 0.419 | 0.402 | 0.326 | 0.625 |

E2 的 V 条件中，CV 40 句及 ET-test 3 句缺合格人脸，按冻结规则输出均匀分布、计错且保留在分母。E2 A+V+T 的 ET-test 串行单句延迟 P50 为 5.424 秒、P95 为 16.118 秒（1,929 句；各阶段计时相加，不含模型加载）。原始 E2 汇总和 SHA-256：`$R3/E2/cv/results.json` 为 `de1eb0f74076d34f2ea4ec5a63e3ff69dd084a52f205f35f6db7c780a04cdea9`；`$R3/E2/eval/summary.json` 为 `ac73f78c2f68632375946fe96787c31eebc7aa79c84865fb19e3a144199d0329`；`$R3/E2/latency/summary.json` 为 `f11ce22be673e4cb67b0f99f82160a4ef235ebfad733b65f02b412b7c6e97b46`。

## 6. 臂 Q

X02 人脸裁剪已在原框每侧外扩 20%。X03 对冻结裁剪帧重新做人脸尺寸至少 64/448 像素、YuNet 置信度至少 0.8 的过滤，再按原片段时长等间距选 8 帧，编码为 336×336 无音轨 MP4。每帧输入视觉处理器的源图像像素数为 **112,896**；示例编码流经 `ffprobe` 验证为 8 帧、336×336、仅视频流。全量 19,250 句中 19,207 条成功，43 条无合格人脸帧，其中官方 train 33、val 7、test 3；阈值没有事后调整。

Q 冒烟使用原始 WAV 在处理子进程转为 16 kHz 单声道，A/V/AV 分别留下 processor 回执：A 仅音频，V 仅视频，AV 同时含二者；forward 再次断言。七类首 token ID 两两不同。模型从本地 Qwen2.5-Omni-7B 基座加载，NF4 双重量化、BF16；语言注意力 LoRA rank 16，视觉注意力 LoRA rank 8，音频塔无可训练参数。实测峰值 CUDA allocated：A 10,089 MiB、V 24,291 MiB、AV 25,816 MiB。这里只进行了重复单样本优化步，不产生可用于评估的 adapter。

64 句正式训练循环冒烟完成 8 次参数更新，用时 60.36 秒，即 0.943 秒/句，峰值 CUDA allocated 28,892 MiB。六次训练每轮共 62,201 句、最多两轮，按实测约 32.59 小时，另加 20% 训练耗时余量后为 39.11 小时；固定子集共 33,600 次 AV 验证预计 3.49 小时，三条件正式推理预计 4.54 小时，模型加载/保存预留 0.5 小时。总预检估算 **47.63 GPU 小时**，低于新门限 55 小时；来源及计算见 [预检预算 JSON](results/q_budget_fold0_preflight.json)。媒体 CPU 预处理未计 GPU 小时。

fold 0 正式训练共处理 19,652 个样本，2457 次优化步，GPU 实测 **25,862.63 秒（7.18 小时）**；8 次固定 700 句 AV 内验证均覆盖完整且无推理失败。最佳为第 5 次，macro-F1 **0.696626**，adapter SHA-256 `aa1e213b1eac1ad0f1b42207e7b503b99fdf3e82d849c8f39d1bcf34ecce8b50`。模态前向回执：A 3960、V 3922、AV 11770；缺脸训练回退 30 次。完整训练曲线和统计见 [fold 0 汇总](results/q_fold0_summary.json)。流水线根据实测将六次训练及推理总 GPU 时长重估为 **50.5124 小时**，低于 55 小时上限；门禁值记录在服务器 `$R3/Q/pipeline_progress.json`。该预算仍须在后续每折根据实际耗时核验。

fold 1 起的 Q 训练及推理进程设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。训练每次内验证另存独立 `last`：adapter、优化器、调度器、PyTorch/NumPy/Python RNG、累积梯度、数据位置及输出文件偏移；异常退出最多从 `last` 重试 2 次，记录 `incidents.jsonl`。每个累积段遇 CUDA OOM 后清缓存重试一次，再失败则把该段句子和输入 token 长度记入 `skipped_oom.jsonl`；累计超过 10 句即停止。每 50 个优化步记录 PyTorch 峰值、`nvidia-smi` 已用显存和步时；连续 2 步超过此前步时中位数 3 倍时写告警。fold 0 结束后、fold 1 开始前，对全部训练 manifest 中去重的 15,281 条 AV 句按实际 `input_ids` 长度排序，逐条前后向检查最长 20 条；PyTorch 分配峰值超过 **30,000,000,000 字节（十进制 30 GB）**或 OOM 即停止，不截断输入。目前正在准备候选缓存，预检结果待运行。

## 7. 臂 H

Q 正式 OOF 不存在，H 未运行。

## 8. 主对照表与配对差值

缺少 Q/H 正式预测；完整主表和配对差值未生成。E1/E2 的已有结果不充当最终架构对照。

## 9. 缺失模态鲁棒性与截断敏感性

E2 的 A、V、A+V 缺失模态评估已完成，数值见第 5 节。Q 仅完成模态冒烟，没有评估预测；跨系统截断敏感性仍待 Q/H 完成。X02 冻结质量标记已就绪，不参与参数选择。

## 10. 错误分析

E1 各系统的逐类指标和混淆矩阵由脚本写入 `results/E1_*_metrics.json`。跨臂互补性及说话人排名未计算。

## 11. 臂 P

Reviewer 取消臂 P。X02 P2 已用 bitsandbytes NF4 双重量化和 BF16 尝试 4bit 加载，PyTorch CUDA 峰值仍为 58,041.6 MiB。X02_REVIEW 中“未尝试 4bit”的说法与运行记录不符。

## 12. 耗时与资源

E1 为 CPU 后处理，E2 的训练、目标集评估和延迟测量在 Q 训练期间只用 CPU；E2 端到端延迟见第 5 节。Q 冒烟显存峰值见第 6 节。GPU 预算估算、原始三条件计时与参数数量见 [预算 JSON](results/q_budget_summary.json)；视频覆盖见 [媒体 JSON](results/media_summary.json)；E1 汇总见 [E1 JSON](results/e1_summary_v4.json)。服务器结果根目录为 `/home/administrator/emotion-data/emotioncore_runs/X03`（下文记为 `$R3`）；运行元数据与 SHA-256 保存在 `results/`，逐句概率留在服务器，不入库。旧项目目录 `/home/administrator/projects/emotion` 只读使用本地基座和 YuNet 权重。

## 13. 异常

- ET 有 43 条固定人脸过滤失败，已在 fold 0 前冻结回退规则。
- SIMS-valid SenseVoice 首次调用的 Python 环境缺少 FunASR，失败收据保留；改用 X02 原 FunASR 环境后成功 620/647，27 条源音频缺失。
- CH-SIMS test 封存集未推理；旧项目目录保持只读。

## 14. 给 reviewer 的问题

预算、缺脸回退、T1r valid 补做、中性误报口径、X02 归档和取消 P 均已获批。目前没有新的协议问题。
