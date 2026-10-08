# X04b Q_mix 执行报告

## 1. 摘要

X04b状态：running；已完成模型['fold0', 'fold1']，活动阶段predict_fold1。仅执行预登记的fold0、fold1和最终模型；是否采用Q_mix由reviewer决定。

## 2. 第 0 步

X04a验收通过；PR #4原已Ready，2026-10-07T04:04:32Z合入main，merge e192c058e953a29a2a67b9b6ccde9260fcfeb8e9。x04b-q从该main创建。

## 3. 完成情况与偏离

状态：running。已完成模型['fold0', 'fold1']；活动阶段predict_fold1。不改变科学协议、不在ET-test/MER-test选择。

## 4. 训练数据与 MER 损失实现

每epoch ET原清单全部 + 4693条MER分层子集；只做fold0/fold1/final。训练前九个首token均不同。ET直接沿用X03 prepare_target和模型result.loss；MER仅在首标签token位置用六类CE，其他目标token全词表CE，同样按目标token数平均。起点为NF4基座，不载入X03 adapter；默认分配器、三处显式检查点、last续训/OOM/步时保护保留。

| 模型 | ET/epoch | MER/epoch | 训练总句次 | 内验证 |
| --- | --- | --- | --- | --- |
| fold0 | 9826 | 4693 | 29038 | 700 |
| fold1 | 9833 | 4693 | 29052 | 700 |
| final | 13100 | 4693 | 35586 | 700 |

清单/预算/缓存来源SHA见results；每epoch0/1种子20261007/20261008。分层舍入、形式分配与合并顺序详见OPEN_QUESTIONS。测试集均不进入训练和参数选择。

## 5. 训练曲线、模态回执、各模型最佳 checkpoint

| 模型 | 最佳内验证AV F1 | 最佳验证次数 | 检查点路径 | adapter SHA | OOM跳过句次 |
| --- | --- | --- | --- | --- | --- |
| fold0 | 0.7189 | 3 | /mnt/h/emotionAI/emotioncore_runs/X04b/Q/fold0/best_adapter_eval3 | 8d2020c851555ad22bc997e939f1f6d2818e5f6a22e84c40ab8087f030a5168c | 0 |
| fold1 | 0.6258 | 6 | /mnt/h/emotionAI/emotioncore_runs/X04b/Q/fold1/best_adapter_eval6 | 6da4e86fcfa4af6428590aaad6b626332a237ca579034571927aa6b2c42514ee | 0 |

每0.25epoch固定700句AV验证；全部验证曲线及模态统计见各summary。last与best独立保存，异常恢复及GPU耗时含在编排账本。

最长20条AV预检pass，峰值14,577,884,160字节，门限30,000,000,000字节；回执见results/precheck.json。

## 6. 主比较 P1/P2

待执行；两折互拟合，种子20261007，2000次bootstrap。

## 7. 次要比较

待执行。

## 8. ET-test 与 MER-test

待执行，只报告；不参与采用规则。

## 9. 耗时与资源

启动前总GPU投影 **64.3712小时**，上限70，门禁通过。计20%余量、0.5小时加载/检查点、1小时诊断；X03检查点开启实测曲线和X04a MER长度来源均冻结，精确计算两epoch的分层子集与形式。每完成模型用实耗上调剩余投影，根分区可用低于80GB停止。

| 模型 | 训练+8次验证预测GPUh | 全部预定推理GPUh |
| --- | --- | --- |
| fold0 | 15.5608 | 0.8854 |
| fold1 | 15.5659 | 0.8813 |
| final | 18.6963 | 0.8029 |

编排GPU累计账本31.024081小时（活动阶段结束/停止后入账；异常重试全计）。最新预算见results/q_budget_progress.json。

最新复核（2026-10-08T11:19:42.021005+00:00，fold1）：总GPU投影56.869621小时，上限70，门禁通过；包含剩余阶段20%余量与加载/诊断预留。

## 10. 异常

暂无异常退出。

## 11. 给 reviewer 的问题

见OPEN_QUESTIONS.md；未覆盖口径采用保守做法。
