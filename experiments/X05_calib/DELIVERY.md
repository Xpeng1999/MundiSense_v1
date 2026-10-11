# X05 交付索引

执行范围为 `X05_PROMPT.md` 与 `X05_REVISION_1.md`。实验与自动报告于 2026-10-11 14:16（Asia/Shanghai）完成；随后完成执行方的独立指标复算与产物入库。PR #6 提交验收，不合入；是否通过由 reviewer 决定。

| 交付物 | 位置 |
| --- | --- |
| 固定十一节报告（第 9 节按协议留空） | [REPORT.md](REPORT.md) |
| 不可变方法文件 | [x05_method_freeze.json](../../configs/x05_method_freeze.json) |
| 部署参数包 | [calibration_bundle_v1.json](../../configs/calibration_bundle_v1.json) |
| 调用接口与 X06 输入要求 | [X06_HANDOFF.md](../../docs/X06_HANDOFF.md) |
| 全部汇总结果、bootstrap、来源和运行回执 | [results/](results/) |
| 可复核的服务器文件路径、SHA-256、仓库位置 | [delivery_manifest.json](results/delivery_manifest.json) |
| 最终测试 | [final_tests.json](results/final_tests.json) |
| 执行方独立复算与覆盖审计 | [delivery_audit_summary.json](results/delivery_audit_summary.json) |
| 历史停止与修订说明 | [STOP_REPORT_20261011.md](STOP_REPORT_20261011.md)、[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) |

参数包 SHA-256：`94a0805b9775676209306187a3880595b7e7592af70aaec8d50f3b2a7968979a`。

方法文件 SHA-256：`8e29dd19beaa3401fb2103542b5333d392d89b9c3e9f819153c313e8fcbcebbe`；冻结提交 `755d7045265d7b95d75fc7f6ffaac41838e24f37`，早于 Q_mix 分析，之后未修改。

## 验证回执

- A3 复现与原 X04a checkpoint 权重逐值相同；三个 D1_fmhx 头各使用 ET 13,100 + MER 9,386 句，FMH 交集为 0，无新选参、无验证集。
- Q_et、Q_mix 的 A/V/AV 每个文件均为全部 3,266 句；逐 ID、真值、说话人与模态回执检查通过；OOM 跳句为 0。
- 最终 pytest：35 项，33 通过、2 跳过、0 失败。两项跳过是既有 GPU 专用测试，最终测试阶段显式禁用 CUDA；实际部署参数包重放测试已执行。
- 执行方审计使用已有独立分类指标实现，从逐句预测重新计算四个系统的 FMH macro-F1、UAR、WA、中性误报与 R90/R80 拒识指标，与汇总一致。该审计不替代 reviewer 独立验收。
- 部署融合/校准拟合集为 9,799 句、9 名说话人；中性偏置和拒识阈值只拟合 FMH。ET-test、MER-test、SIMS-valid 只报告，CH-SIMS test 零推理。
- GPU 总计 2.891365538 / 5 小时。原 X04a 代码与三份冻结划分/规则未改。

## 证据读取

服务器根目录：`/mnt/h/emotionAI/emotioncore_runs/X05`。原始逐句概率、训练 ID 回执与模型权重保留在服务器，仓库只保存汇总及其路径/SHA。59 份取回的交付文件在 `delivery_manifest.json` 中逐项登记并校验。

`methods_*.json` 保留方法冻结时的结果。报告用 `diagnostics_summary.json` 中修正后的选择性曲线和守门触发率；这项修正只排除不可用预测被当作接受的情况，并对齐已有缺脸规则，没有重拟合或修改方法文件。报告和图表保留服务器生成的原始字节，可按 manifest 校验。

## 待 reviewer 判断

H_V 的 ET-test 所选标签 ECE 为 0.0621、top-label ECE 为 0.0610，超过 0.05，已在报告第 11 节单列。MER-test H_V 缺既有 Q-V 概率，未扩大推理范围补算。人脸质量五分位退化、FMH 三说话人和 700 句 checkpoint 选择限制均照实报告；无结果驱动的协议或阈值修改。
