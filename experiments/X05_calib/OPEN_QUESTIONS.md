# X05 待 reviewer 处理的问题

## Q1：专家 fold 0 已训练 FMH，原定回退来源不成立（阻塞）

X05_PROMPT.md 第 3 节规定：最终专家头若训练过 06/10/11，就使用 fold 0 专家头，理由是这三名说话人在 fold 0 只用于内验证。

实际 `emotioncore.x04.heads` 分为选参、outer refit 和 final 三阶段。06/10/11 在 fold 0 **选参阶段**仅用于内验证，但 **outer refit 阶段**已进入训练。A3、V2、V1 的 `fold0/training_ids.jsonl` 均包含全部 3,266 条 FMH，且各自目录中仅保存最终头 `final_head.pt`，没有选参阶段 checkpoint。最终头则使用全部 CV 数据。

这不是本轮产生的训练泄漏，而是拟议 X05 FMH 用途与已有专家训练范围不兼容。如果照原回退假设执行，就会把专家训练过的样本当作未训练说话人预测用于部署拟合与评估。

保守处理：触发“泄漏断言失败，立即停止并报告”；没有训练、推理、Q_et 方法选择或 Q_mix 分析；不改用未经批准的混合折专家预测，也不补训任何头。

需要 reviewer 明确：是否存在可验证且未训练 FMH 的现存 D1 checkpoint，或应怎样修订专家侧 FMH 构造与拟合规则。证据与复核方法见 `REPORT.md` 和 `results/expert_fmh_gate.json`。
