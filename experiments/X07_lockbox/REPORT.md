# X07：冻结系统的 CH-SIMS v2 封存集验收

## 1. 摘要

L3 健全性门槛通过，L4 全部完成；最终性能验收归 reviewer。
SIMS-valid H3 媒体可用 Pearson 0.828855，预登记门槛为 0.78。
SIMS-test H3 Pearson 0.792257，视频整群 95% 区间 [0.756380, 0.828748]，Acc-2 0.818880。
按 X07 修订 1 应用标签隔离；v2 协议先提交，再重做 L2 → L3 → L4。
身份检查在 L2 与指标锁前完成，两次回执一致；每个已完成划分仅计算一次指标。
没有训练、拟合、选参或改动模型、指标、bootstrap、缺媒体规则及门槛。
v1 协议、STOP 与目录保留；v1 未读取 test 媒体，也没有计算任何指标。
GPU 累计 2246.034 秒 / 14,400 秒，包含初始化失败；X07 PR #9 保持草稿、未合入。

## 2. 第 0 步

PR [#8](https://github.com/Xpeng1999/emotioncore-zh/pull/8) 已由 reviewer 验收通过，附两项登记局限：计划端点至输出 P50 约 2.2 秒；内存前约 90 分钟增长 11% 后持平。

PR #8 于 2026-10-14 转 Ready。首次合并请求被自动审批将“X07 新 PR 不合入”误解为撤销 #8 授权而拒绝，未产生合并；用户随后明确澄清并更新执行指令。按此授权完成合并，合并提交为 `5e947e9bcc8c1e99eba02e225e37db6cbcb9a4e5`，已验收 head 为 `16d5244447d9c4f07ba4c582dfdd686df9d58c48`。

本轮 `x07-lockbox` 从合并后的 `main` 创建。仅 X07 新 PR 保持未合入，等待 reviewer 验收。

## 3. 完成情况与偏离

L3 健全性门槛通过，L4 全部完成；最终性能验收归 reviewer。v2 的 L2 已重新通过，valid 共 647 条，test 共 1,034 条，逐条输入与预测凭据见 [交付审计](results/v2/delivery_audit.json)。

依据 [X07_REVISION_1.md](protocol/X07_REVISION_1.md)，在代码提交 `25b260d` 应用历史提案并完成第 2 节要求：独立身份函数只解析 video_id / clip_id / mode；在 L2 对 valid/test 各检查 ID 集合及每记录一行；指标入口在领取 O_EXCL 锁前重复检查并与 L2 回执逐项比较。领取锁后，只解析对应划分的标签。另一划分的标签在 pandas 解析前按预检行号跳过。两次检查均核验冻结 meta.csv SHA。

v2 仅变更标签访问代码、代码 SHA、协议溯源和执行路径。模型、参数包、人脸/VAD 配置、所有指标定义、bootstrap、缺媒体规则及通过门槛与 v1 完全一致。每条片段仍为单句，不运行 VAD。原始 7 条原生帧解码失败按冻结规则保留，没有修改解码器、取帧方式或参数。

v1 在任何 GPU 推理及指标执行前主动停止；**v1 期间没有读取任何 test 媒体，也没有计算任何指标**。原协议、原目录、STOP 及历史记录保持原样；[v1 报告快照](REPORT_BEFORE_REVISION_1.md) 和 [v1 系统卡快照](../../docs/SYSTEM_CARD_BEFORE_REVISION_1.md) 另行保留，[STOP_RECOVERY.md](STOP_RECOVERY.md) 和原提案保留为当时的审阅快照。v1 的 30,585 个文件大小与 mtime 均未变化，STOP 和原协议 SHA 复核一致。

v2 首次初始化因启动环境未包含 timm 而失败，预测数为 0。按协议的基础设施恢复规则，将 X06b 已使用的 timm 0.9.16 复制到 v2 独立依赖目录；269 个文件 SHA 一致，CPU 模型加载通过，相关依赖版本匹配验收记录。冻结代码与配置未改；续跑保留原始中断记录及 34.094541 秒 GPU 用时，没有重跑任何已完成片段。见 [恢复凭据](results/v2/dependency_recovery.json) 与 [续跑入口](results/v2/infra_resume_1.py)。

## 4. 协议冻结与泄漏断言

- v1 原协议提交 `6d9d07c2e592c82fa97d4d31e5eae37658e83360`，SHA `8ca5430c3c009450d30daa7c4cdf04bb162819d70df21857225069c37eca09c2`；保留不改。
- v2 代码及测试证据提交 `52852c837959f7ff345531adb8af96301242ead5`；[v2 协议](../../configs/x07_lockbox_protocol_v2.json) 提交 `32479b378bc75596f23fa2db430254c720d66127`，SHA `fba5481ec16df7b5d8f1d403542cd7552c9fbf2d57013da96f1b1e1fff475e80`。
- v2 协议先提交并推送，后启动 L2 与任何推理；启动证明见 [execution_start.json](results/v2/execution_start.json)。
- v2 独立目录：`/mnt/h/emotionAI/emotioncore_runs/X07_v2`；原目录 `/mnt/h/emotionAI/emotioncore_runs/X07` 未复用或覆盖。
- valid 647、test 1,034；ID 清单及 SHA、44 个模型/相关文件 SHA 与 v1 相同。25,426,572,777 字节模型资产在 GPU 启动前及交付时验证；Q、专家头、参数包、人脸与 VAD 均未变。

| L2 检查 | 清单行数 | SIMS-test 重叠 |
| --- | ---: | ---: |
| Q final epoch 0 | 17,793 | 0 |
| Q final epoch 1 | 17,793 | 0 |
| D1_fmhx A3 | 22,486 | 0 |
| D1_fmhx V2 | 22,486 | 0 |
| D1_fmhx V1 | 22,486 | 0 |
| 参数包 fitting_ids | 9,799 | 0 |
| MER-train 去重后 | 9,386 | 0 |

三条疑似重复 `mer_sample_00000876`、`mer_sample_00002517`、`mer_samplenew3_00104997` 已排除且不在上述任何训练清单中。训练文件 SHA 与最终 Q run_meta、三个专家 summary、参数包 fitting_set 交叉核对。

新增身份检查：meta.csv 共 4,403 条记录、4,404 个物理行，valid/test 的精确 ID 集与协议一致，两项检查分别通过并写入 [L2 回执](results/v2/leakage.json)。每个已测划分领取指标锁前的检查与 L2 回执完全相同，锁时间晚于全部预测完成时间。

[CPU 测试](results/v2/preflight_tests.txt)：**113 项通过**。覆盖身份/物理行检查失败不领取指标锁、冻结 SHA 或 L2 回执不一致时拒绝、两个方向的标签隔离哨兵、协议不可变及提交时间、七项泄漏断言、三个疑似重复、推理前拒绝指标、第二次指标拒绝、L3 拦截、X01 分数及冻结系统一致性。

时序核对（北京时间，全部完成后才领取指标锁）：

| 划分 | 全部预测完成 | 指标锁领取 | 一次计算结束 |
| --- | --- | --- | --- |
| valid | 2026-10-14 13:51:47 | 2026-10-14 13:51:57 | 2026-10-14 13:52:01 |
| test | 2026-10-14 14:22:16 | 2026-10-14 14:22:31 | 2026-10-14 14:22:35 |

## 5. SIMS-valid 预演

本次使用与 L4 相同的代码及配置，全部 647 条预测完成后只计算一次指标。H3 媒体可用 Pearson **0.828855**，达到预登记门槛 0.78，L3 **通过**。没有按结果改动参数或门槛。

| 指标（视频整群 95% 区间） | H3 | H_A | H_V |
| --- | ---: | ---: | ---: |
| 媒体可用 Pearson | 0.828855 [0.801631, 0.855394] | 0.780306 [0.747176, 0.816494] | 0.692689 [0.645447, 0.734257] |
| 媒体可用 Spearman | 0.826335 [0.798334, 0.850206] | 0.779929 [0.745529, 0.814281] | 0.664364 [0.614381, 0.712037] |
| Acc-2（非零标签） | 0.839640 [0.795912, 0.881945] | 0.809009 [0.764806, 0.852784] | 0.774775 [0.724527, 0.818854] |
| 全体 Pearson（不可用为均匀） | 0.797768 [0.759189, 0.833428] | 0.749210 [0.706764, 0.792458] | 0.655924 [0.606038, 0.701798] |
| OP-R90 覆盖率 | 0.663060 [0.598934, 0.719220] | 0.846986 [0.805836, 0.884558] | 0.601236 [0.537174, 0.656832] |
| 被接受片段 Pearson | 0.855969 [0.820051, 0.889687] | 0.795779 [0.762404, 0.831915] | 0.745375 [0.696823, 0.790982] |
| 零标签被判为非中性比例 | 0.282609 [0.190464, 0.405405] | 0.260870 [0.175245, 0.364586] | 0.184783 [0.107124, 0.272727] |

H3：媒体可用 613/647，R90 接受 429/647；H_A：媒体可用 613/647，R90 接受 548/647；H_V：媒体可用 601/647，R90 接受 389/647。
非零标签 555，零标签 92，视频整群数 120。
所有系统、所有指标均有 2,000 个有效 bootstrap 重复。

完整一次性计算结果：[valid_metrics.json](results/v2/valid_metrics.json)。

## 6. 封存集结果（主指标与次要指标）

全部 1,034 条预测完成后，首次领取 test 指标锁并一次计算 H3、H_A、H_V 的所有指标；没有推理重跑或指标复算。首次 test 媒体准备晚于 L3 通过的指标回执。主结果：H3 Pearson **0.792257**，视频整群 95% 区间 **[0.756380, 0.828748]**，Acc-2 **0.818880**。

| 指标（视频整群 95% 区间） | H3 | H_A | H_V |
| --- | ---: | ---: | ---: |
| 媒体可用 Pearson | 0.792257 [0.756380, 0.828748] | 0.763117 [0.725884, 0.798771] | 0.652793 [0.602035, 0.704787] |
| 媒体可用 Spearman | 0.801584 [0.766294, 0.834392] | 0.771356 [0.735447, 0.803788] | 0.650229 [0.600692, 0.700926] |
| Acc-2（非零标签） | 0.818880 [0.783718, 0.850890] | 0.804610 [0.771370, 0.837596] | 0.740944 [0.700527, 0.780130] |
| 全体 Pearson（不可用为均匀） | 0.761319 [0.721594, 0.799369] | 0.731151 [0.691799, 0.768804] | 0.621335 [0.568282, 0.674813] |
| OP-R90 覆盖率 | 0.623791 [0.562364, 0.681449] | 0.874275 [0.838390, 0.903012] | 0.588008 [0.523328, 0.648432] |
| 被接受片段 Pearson | 0.810012 [0.767874, 0.852481] | 0.777256 [0.740026, 0.813086] | 0.703798 [0.643379, 0.768555] |
| 零标签被判为非中性比例 | 0.219512 [0.141593, 0.306122] | 0.203252 [0.122565, 0.293678] | 0.186992 [0.116062, 0.264966] |

H3：媒体可用 976/1034，R90 接受 645/1034；H_A：媒体可用 976/1034，R90 接受 904/1034；H_V：媒体可用 967/1034，R90 接受 608/1034。
非零标签 911，零标签 123，视频整群数 142。
所有系统、所有指标均有 2,000 个有效 bootstrap 重复。

| reviewer 预登记门槛 | 本次值 | 数值对照 |
| --- | ---: | --- |
| H3 Pearson ≥ 0.75 | 0.792257 | 达到 |
| 视频整群 95% 下界 ≥ 0.70 | 0.756380 | 达到 |
| H3 Acc-2 ≥ 0.80 | 0.818880 | 达到 |

最终验收由 reviewer 判断。完整结果：[test_metrics.json](results/v2/test_metrics.json)。

## 7. 与开发集、ET-test 的对照

| 集合与任务 | 片段数 | H3 倾向 Pearson（媒体可用） | 倾向 Acc-2 | OP-N 七类 macro-F1 | OP-R90 七类 macro-F1 / 覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 本轮 SIMS-valid 倾向 | 647 | 0.828855 | 0.839640 | 不适用 | 不适用 |
| 本轮 SIMS-test 倾向 | 1,034 | 0.792257 | 0.818880 | 不适用 | 不适用 |
| X06b ET-test 七类 | 1,929 | 不适用 | 不适用 | 0.645131 | 0.641794 / 88.491% |
| X06b FMH 七类（参与校准/选择） | 3,266 | 不适用 | 不适用 | 0.696641 | 0.668132 / 89.988% |

既有 [ET-test](../X06b_realtime/results/r3_analysis_et_test_retry1.json)、[FMH](../X06b_realtime/results/r3_analysis_fmh.json) 直接引用，不重新推理。其 R3 单句诊断未应用多脸门禁，拒识计入 F1 分母。CH-SIMS 没有对应七类标签；情绪倾向 Pearson 不能与七类 F1 等价比较。本轮只能说明相应语料上的情绪倾向表现，不能证明真实设备、真实用户或多人场景的效果。

## 8. 失败片段清单

输入路径、SHA 和逐条失败原因见 [交付审计](results/v2/delivery_audit.json)。媒体不可用时，各系统均匀概率、倾向分数 −3/7；媒体可用 Pearson 排除这些条目，全体 Pearson 保留，Acc-2 非零标签分母保留并计错。无可用人脸时 H3 回退 H_A，H_V 单独按不可用处理；不据此改模型或门槛。

媒体不可用条目：valid 共 34 条（missing 27 条、STOP native decode failed 7 条）；test 共 58 条（STOP native decode failed 18 条、missing 40 条）。另有 valid 12 条、test 9 条仅 H_V 因无可用人脸而不可用，H3 按冻结规则回退 H_A；逐条 ID 见 [次要系统无脸清单](results/v2/secondary_no_face.json)。

原有 7 条 native decode 失败继续按相同规则登记；没有修改解码或参数。

| 划分 | 片段 ID | 原因 |
| --- | --- | --- |
| valid | aqgy4_0001__00028 | missing |
| valid | aqgy4_0001__00033 | STOP native decode failed |
| valid | aqgy4_0016__00007 | STOP native decode failed |
| valid | aqgy4_0018__00020 | STOP native decode failed |
| valid | aqgy4_0023__00000 | missing |
| valid | aqgy5_0005__00033 | STOP native decode failed |
| valid | aqgy5_0017__00027 | STOP native decode failed |
| valid | aqgy5_0018__00007 | STOP native decode failed |
| valid | aqgy5_0022__00023 | STOP native decode failed |
| valid | aqgy5_0028__00037 | missing |
| valid | aqgy5_0028__00041 | missing |
| valid | aqgy5_0030__00013 | missing |
| valid | video_0002__0006 | missing |
| valid | video_0002__0007 | missing |
| valid | video_0002__0009 | missing |
| valid | video_0002__0010 | missing |
| valid | video_0002__0020 | missing |
| valid | video_0002__0023 | missing |
| valid | video_0002__0026 | missing |
| valid | video_0002__0028 | missing |
| valid | video_0002__0034 | missing |
| valid | video_0011__0031 | missing |
| valid | video_0021__0008 | missing |
| valid | video_0021__0010 | missing |
| valid | video_0046__0002 | missing |
| valid | video_0046__0026 | missing |
| valid | video_0046__0048 | missing |
| valid | video_0049__0001 | missing |
| valid | video_0049__0002 | missing |
| valid | video_0049__0003 | missing |
| valid | video_0049__0013 | missing |
| valid | video_0049__0014 | missing |
| valid | video_0049__0017 | missing |
| valid | video_0050__0004 | missing |
| test | aqgy4_0001__00023 | STOP native decode failed |
| test | aqgy4_0016__00017 | STOP native decode failed |
| test | aqgy4_0022__00003 | STOP native decode failed |
| test | aqgy5_0002__00011 | STOP native decode failed |
| test | aqgy5_0004__00019 | STOP native decode failed |
| test | aqgy5_0005__00005 | STOP native decode failed |
| test | aqgy5_0005__00014 | STOP native decode failed |
| test | aqgy5_0005__00040 | missing |
| test | aqgy5_0008__00001 | missing |
| test | aqgy5_0008__00010 | missing |
| test | aqgy5_0008__00020 | missing |
| test | aqgy5_0008__00021 | missing |
| test | aqgy5_0008__00023 | missing |
| test | aqgy5_0008__00025 | missing |
| test | aqgy5_0008__00033 | missing |
| test | aqgy5_0008__00036 | missing |
| test | aqgy5_0008__00037 | missing |
| test | aqgy5_0009__00012 | STOP native decode failed |
| test | aqgy5_0011__00020 | STOP native decode failed |
| test | aqgy5_0013__00029 | STOP native decode failed |
| test | aqgy5_0014__00011 | missing |
| test | aqgy5_0016__00040 | STOP native decode failed |
| test | aqgy5_0017__00016 | STOP native decode failed |
| test | aqgy5_0017__00018 | STOP native decode failed |
| test | aqgy5_0019__00000 | STOP native decode failed |
| test | aqgy5_0019__00017 | STOP native decode failed |
| test | aqgy5_0020__00015 | STOP native decode failed |
| test | aqgy5_0022__00012 | missing |
| test | aqgy5_0025__00001 | missing |
| test | aqgy5_0026__00012 | STOP native decode failed |
| test | aqgy5_0029__00019 | STOP native decode failed |
| test | aqgy5_0032__00001 | missing |
| test | test14__00000 | missing |
| test | test14__00001 | missing |
| test | video_0002__0003 | missing |
| test | video_0002__0004 | missing |
| test | video_0002__0008 | missing |
| test | video_0002__0017 | missing |
| test | video_0002__0025 | missing |
| test | video_0002__0029 | missing |
| test | video_0002__0032 | missing |
| test | video_0002__0033 | missing |
| test | video_0002__0037 | missing |
| test | video_0021__0001 | missing |
| test | video_0021__0003 | missing |
| test | video_0021__0004 | missing |
| test | video_0021__0005 | missing |
| test | video_0021__0006 | missing |
| test | video_0021__0011 | missing |
| test | video_0024__0037 | missing |
| test | video_0035__0001 | missing |
| test | video_0046__0034 | missing |
| test | video_0046__0054 | missing |
| test | video_0049__0020 | missing |
| test | video_0050__0002 | missing |
| test | video_0050__0008 | missing |
| test | video_0050__0010 | missing |
| test | video_0055__0034 | missing |

## 9. 耗时与资源

GPU 累计 **2246.034 秒（0.6239 小时）/ 4 小时**。按子进程墙钟计费，包含初始化、CPU 等待和失败；保留 12 秒终止余量。CPU 媒体准备不计入 GPU 用时。

| GPU 作业 | 状态 | 计费秒 |
| --- | --- | ---: |
| test / 2026-10-14T05:59:51.561179+00:00 | completed | 1349.622 |
| valid / 2026-10-14T05:32:40.288258+00:00 | failed | 34.095 |
| valid / 2026-10-14T05:37:28.134043+00:00 | completed | 862.318 |

执行机为 WSL2、RTX 5090 32 GB、128 CPU 线程、约 94 GB 内存。交付审计时根分区可用 112.93 GB，高于 80 GB 门槛；执行期间看门狗和逐条检查均启用。新增媒体、缓存、运行日志、预测与指标全部位于独立 v2 根目录。原始逐条预测保留服务器，仓库登记完整完成清单、路径及 SHA：[valid 预测索引](results/v2/valid_complete.json)、[test 预测索引](results/v2/test_complete.json)。

X06b 已验收局限继续有效：计划端点至输出 P50 约 2.2 秒、P95 约 2.8 秒；内存前 90 分钟增长约 11% 后持平。X07 单句批处理不能替代真实设备延迟或 6 小时以上长时运行验证。

valid：CPU 准备 260.905 秒，GPU 峰值分配 11.379 GiB；test：CPU 准备 440.113 秒，GPU 峰值分配 11.440 GiB。

## 10. 异常

历史 v1 标签隔离自检停止及原始 STOP 见 [stop_evidence.json](results/stop_evidence.json)，未实际执行有缺口的标签读取路径。修订 1 已授权并落实到独立 v2。

v2 首次 GPU 初始化因启动时缺少 timm 路径失败，未产生预测或标签读取。恢复过程使用已验收的原版本文件，保留中断日志、失败账本、版本和逐文件 SHA；未变更 venv 原件、模型、冻结代码或配置。

随后无进一步中断；L3 与 L4 按冻结顺序完成。

## 11. 给 reviewer 的问题

请 reviewer 按第 6 节预登记三项条件判断本轮是否通过；执行方只报告数值，不代替验收。

请同时审阅修订 1 的锁前身份检查、首次初始化故障与同版本依赖恢复凭据。缺媒体 Acc-2 计错、H3 无脸回退、H_V 使用既有 R90 阈值、整群 bootstrap 的退化处理均沿用 [冻结前口径](OPEN_QUESTIONS.md)。PR #9 保持 Draft，不合入。
