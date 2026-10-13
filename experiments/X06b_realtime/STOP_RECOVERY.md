# X06b 旧 ET-test 汇总异常与恢复边界

状态：已获授权并恢复。原阶段于 2026-10-13 14:35:07（北京时间）写入 STOP；该历史回执仍保留。授权依据为 [X06B_RESUME.md](protocol/X06B_RESUME.md)，以下五条范围已由 reviewer 和用户确认。

## 原因与已完成部分

`r3_analysis_et_test` 的无脸句数统计直接读取旧预测的 `has_face`。X06a ET-test 的全部 1,929 条预测都没有该字段；FMH 格式包含该字段。概率、输入和句子 ID 完整且 SHA 已核对，属于 X06b 报表读取的格式兼容缺陷。

R2 已冻结 1/2。R3 FMH 3,266 句完成，OP-N 差 −0.005144，95% CI [−0.011781, +0.001656]。R3 ET-test 1,929 句推理完成，正式分析摘要未完成。GPU 共用 0.962119 小时，最后总预算预测 10.572962 / 14 小时。R4、AISHELL-4 补测、R5、曲线和最终审计未开始。

[原始 STOP、失败日志、失败 run_meta、ID/字段核对及账本](results/stop_diagnosis_20261013.json)；[阶段回执](results/stages.json)。失败阶段生成过 paired.jsonl，但无 ended_at 或 summary，不能当作正式完成结果。

## 已准备的修复

提交 `85c773d`：从旧阶段回执指定、SHA 校验通过的原输入读取 q_video，逐句按原 `Service.components` 规则恢复人脸标志（非空且 status == ok）。必须完全匹配唯一 ID；缺失输入字段或已有标志冲突时继续拒绝。输出额外记录输入来源与规则。原 ET-test 的无脸句数可恢复为 4；新记录为 0。

未修改 X06a、模型、参数包、人脸冻结配置、统计函数、阈值或标签。首次停止时，只在隔离 check_code 上验证：**62 passed in 2.88s**，[回执](results/unit_tests_et_schema_fix.json)。当时 eval_code 保持 `4c47c6a`，尚未部署修复（本次部署与恢复见下文）；旧 STOP 和失败产物均未改动。修复没有解除 no_stop 检查。

## 已批准的五条恢复范围

1. 保留原 STOP、失败日志、未结束 run_meta 和部分产物；以独立恢复授权回执关联原 STOP SHA，不删除或覆盖历史证据。
2. 在独立重试目录执行修复后的 ET-test CPU 汇总，复用已有 1,929 句新旧预测和原输入，不调用 GPU、不重跑 R2 或 R3 推理。
3. 为已失败历史尝试保留显式失败状态和外部退出证据，最终审计区分历史失败与重试成功；不补写虚假的原成功元数据。
4. 完成新的预算核验，从首个未执行阶段按原顺序继续 R4、R5、曲线和最终审计。累计 GPU 上限仍为 14 小时，所有冻结、数据范围和门槛不变。
5. PR #8 继续 Draft、未合入。本次确认已由 X06B_RESUME.md 和用户明确授予；无需调整科学门槛。

## 恢复执行记录

恢复代码 `277462b` 已在隔离副本通过 **70 项 CPU 测试（2.58 秒）**，并以相同干净提交部署到 eval_code。新建 [恢复回执](results/resume_authorization.json)，引用原 STOP SHA `c170b38c7ffd5b3f319e80e7ffee9d49143bf0cc40439e77f2ef023665a123a6` 和授权文件 SHA `cd79ec9461fc0b8b828e1faab310af24b0b99892fb36407deff4479b615a97b5`。原失败目录、STOP、日志、未结束 run_meta 和部分 paired.jsonl 未改动。

独立目录 `r3_analysis_et_test_retry1` 的 CPU 汇总已完成，复用 1,929 句既有预测，GPU=0；新的 paired.jsonl SHA 与历史部分配对文件完全相同，确认描述性人脸计数修复未改配对决策。ET-test OP-N 差 −0.010931，95% CI [−0.020634, −0.002120]，仅报告，不用于选择或 FMH 门槛。[重试回执](results/r3_analysis_et_test_retry1.json)。

[恢复后预算](results/budget_after_retry_r3_et_test.json)通过：累计 GPU 0.962119 小时，总预测 10.572962 / 14 小时。一次性 `emotioncore.x06b.resume` 已启动，当前进入 R4 对话时序流；后续按授权顺序 R4 → AISHELL-4 CPU 补测 → R5 → 曲线 → 最终审计。当前阶段仍未全部交付。

恢复放行严格绑定本次授权回执和原 STOP SHA。任何新停止写入独立 `stops/*.json`，不会覆盖原 STOP，并阻止后续阶段。最终审计仅将这一个 SHA 一致的未结束旧尝试记为 `historical_failed`，独立核对重试成功；其他未结束尝试仍拒绝。

授权文件同时允许今后纯报表读取格式问题在测试覆盖、保留原证据和新增恢复回执后自主恢复；涉及推理、指标、门槛、数据范围或预算的停止仍须等待 reviewer。此授权不把新 STOP 自动视为已解除。
