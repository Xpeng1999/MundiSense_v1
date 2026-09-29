# X03 → X04 交接：预算门禁后状态

X03 尚未完成架构对照，不应把 E1 臂内结果当作主干选择。停止依据及未完成项目见 [X03 报告](../experiments/X03_fusion/REPORT.md)。

## 可复现的入口

| 项目 | 入口 | 依赖 | 服务器产物 |
|---|---|---|---|
| E1 五折概率融合、校准、ET-test/SIMS-valid | `python -m emotioncore.x03.fusion` | `experiments/X02_unimodal/results/artifact_index.json` 所列冻结 OOF/测试概率；`configs/folds_v1.json` | `/home/administrator/emotion-data/emotioncore_runs/X03/E1` |
| 336×336 八帧人脸视频 | `python -m emotioncore.x03.media` | X02 `face/receipts` 与 `face/frames`、YuNet 权重、冻结 ET manifest | `/home/administrator/emotion-data/emotioncore_runs/X03/media` |
| Q NF4 LoRA 模态和耗时冒烟 | `python -m emotioncore.x03.q_smoke` | 本地 Qwen2.5-Omni-7B 基座、原始 WAV、X03 视频 | `/home/administrator/emotion-data/emotioncore_runs/X03/Q_smoke_budget` |

入口的完整参数、Git 提交、输入 SHA-256 和起止时间见各产物目录的 `run_meta.json`。E1 每个组合的 OOF、ET-test 和可用的 SIMS-valid 概率与参数保存在各自子目录；五模态 E1 OOF SHA-256 为 `a5cafc6f566fe86a90114fa2c82a6870798953f629bdd7488ede229b5447d71a`，ET-test SHA-256 为 `c1036fc4ee7322f6705b2fab52f033949b19a9aab2af138a80b1461878dc767d`。其他逐句概率的路径与 SHA-256 见仓库 `experiments/X03_fusion/results/E1_*_metrics.json`。

## 模型与特征

- E1 只依赖 X02 缓存概率，不加载大模型。含 T1r 两组没有 SIMS-valid 同条件概率。
- Q 冒烟基座为只读 `/home/administrator/projects/emotion/omni_models/Qwen2.5-Omni-7B`，使用 NF4 双重量化与 BF16、语言 LoRA rank 16、视觉 LoRA rank 8；**没有正式 adapter**，不可用冒烟模型作评估。
- 视频已生成 19,207/19,250 条，43 条固定人脸过滤失败。数据在 X03 运行根目录，不在 Git 仓库。
- 可选 P 未启动。X02 大模型与 CLIP-GmP 的 H 盘位置仍见 [X02 对接说明](X03_HANDOFF.md)。

## 继续前的门禁

原定协议的 Q 全量预计 45.95 GPU 小时，超过 45 小时上限；E2、Q、H 的正式实验没有开始。若 reviewer 调整预算，先冻结 43 条人脸缺失的处理规则及 T1r 的 SIMS-valid 处理，再按原折分继续；CH-SIMS test 仍封存。
