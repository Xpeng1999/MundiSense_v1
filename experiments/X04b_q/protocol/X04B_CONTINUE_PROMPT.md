# X04b 续跑指令（reviewer 修订）

先阅读 `X04B_REVIEW.md`。协议以 `X04B_PROMPT.md` 为准，只按下列修订改动，其余一律不变。

## 1. 续跑范围

- 训练 Q_mix 的**最终模型**，严格按 `X04B_PROMPT.md` 第 4 节第 3 条：
  - 训练数据为 X03 `final_train` 的 ET 句 + 已冻结的每 epoch MER 50% 子集（`final_epoch0/1.jsonl`）；
  - 用 fold 0 内验证说话人的 700 句选择 checkpoint；
  - 其余训练设置和保护机制不变。
- **推理：** ET-test、SIMS-valid 做 A/V/AV；MER-test 做 A/AV（6 类，另报 5 个共享类）。
- 徒劳门禁已完成它的作用，不再适用。fold 0、fold 1 不重训、不重推。

## 2. P1 的修订（ET 两折，6,533 句）

- **主口径改为"同折内按说话人留一"校准与融合**：对每名留出说话人，只用**同一模型**对同折其余 2 名说话人的 OOF 来拟合融合权重、温度、7 类偏置和 OP-N 中性偏置，再应用到该说话人。Q_mix 与 Q_et 用完全相同的方法；专家仍用 X04a D1。
- 系统：Q-AV、Q-A、H_ref、H3。差值用 6 名说话人整群 bootstrap，2,000 次，种子 20261007。
- 同时并列报告：原始 OP-A（不拟合）和原两折互拟合的结果，原数值不改。

## 3. P2（不变）

- 目标集的融合与校准参数：Q_mix 与 Q_et **都**用 fold 0+1 合并 OOF 拟合（Q_et 也只用这两折，不用 X03 的五折参数），然后应用到 SIMS-valid、ET-test、MER-test。
- **P2** = SIMS-valid 上 H_ref 与 Q-AV 的 Pearson 差（Q_mix − Q_et），按句 bootstrap 2,000 次，种子 20261007。同时报告 Spearman 和 Acc-2。
- ET-test（含逐说话人）与 MER-test 只报告，不参与决定。

## 4. 采用规则（reviewer 决定，agent 只呈现证据）

修订后 P1 中 H_ref 的区间下界 ≥ −0.01，**并且** P2 中 H_ref 的 Pearson 差区间下界 > 0。

## 5. 预算与停止

- GPU 总上限仍为 70 小时（已用 32.41 小时；最终模型加推理的预估约 19.5 小时）。启动前重估一次；超过上限先停下报告。
- 停止条件不变：泄漏或模态断言失败、覆盖不全、根分区可用空间低于 80GB、OOM 跳过超过 10 句。

## 6. 交付

- 新增 pytest：留一拟合不使用被预测说话人的标签，只使用同一模型的输出，且只使用同折的其他说话人。
- 更新 `experiments/X04b_q/REPORT.md`：
  - 第 3 节写明本修订；
  - 第 6 节用新口径的 P1 和 P2；原两折互拟合结果保留为并列列；
  - 第 8 节补上 ET-test 与 MER-test；
  - 第 5 节补上最终模型的训练曲线。
- 更新 `docs/X05_HANDOFF.md`，推送并更新 PR #5，不合入，等待 reviewer。
