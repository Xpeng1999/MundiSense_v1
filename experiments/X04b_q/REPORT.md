# X04b Q_mix 执行报告

## 1. 摘要

准备中；仅执行fold0、fold1和最终模型，MER每epoch50%。是否采用Q_mix由reviewer决定。

## 2. 第 0 步

X04a验收通过；PR #4原已Ready，2026-10-07T04:04:32Z合入main，merge e192c058e953a29a2a67b9b6ccde9260fcfeb8e9。x04b-q从该main创建。

## 3. 完成情况与偏离

清单冻结、预算与损失测试准备中；训练尚未启动。

## 4. 训练数据与 MER 损失实现

ET完全沿用X03清单、损失、提示词和缓存。MER每epoch4693句，六类分层固定种子20261007+epoch；仅首标签token做六类掩码CE，其余token全词表CE。全部从基座初始化。

## 5. 训练曲线、模态回执、各模型最佳 checkpoint

待执行。

## 6. 主比较 P1/P2

待执行；两折互拟合，种子20261007，2000次bootstrap。

## 7. 次要比较

待执行。

## 8. ET-test 与 MER-test

待执行，只报告；不参与采用规则。

## 9. 耗时与资源

GPU上限70小时；启动前预算未完成，不启动训练。根分区门限80GB，派生数据及adapter位于H盘。

## 10. 异常

暂无。

## 11. 给 reviewer 的问题

见OPEN_QUESTIONS.md；未覆盖口径采用保守做法。
