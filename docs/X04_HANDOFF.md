# X03 → X04 交接（执行中）

X03 的最终主对照表尚未生成；架构选择由 reviewer 在完整证据交付后决定。当前进度见 [X03 报告](../experiments/X03_fusion/REPORT.md)。服务器运行根目录为 `/home/administrator/emotion-data/emotioncore_runs/X03`（下称 `$R3`），冻结 X02 运行根目录为 `/home/administrator/emotion-data/emotioncore_runs/X02`（下称 `$R2`）。

## 冻结输入与规则

- 五个外折：`configs/folds_v1.json`，SHA-256 `85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce`。
- 每折 700 句内验证：[`configs/x03_inner_val_subsets.json`](../configs/x03_inner_val_subsets.json)，种子 20260929，SHA-256 `be2cd442346cfa978e22bea8ab2ae3776d52187b2581fe93e377fc4fb58d3cfd`。
- 缺脸规则：[`configs/x03_face_missing_policy.json`](../configs/x03_face_missing_policy.json)，SHA-256 `ce65415b5f7129b62ef5edc7eda5b20874ed88931f5a7cc295c276b8b704b77c`。ET 的 43 句缺脸：训练 V/AV 改 A，评估 V 均匀且强制计错，AV 仅音频回退；E/H 视觉专家均匀。
- 句级截断敏感性输入：`$R2/quality/utterance_quality.jsonl`，SHA-256 `80330bb718e31a8155793ab83e396bd4741a87347858f0342fd039cdb098bb32`。
- CH-SIMS test 是封存集，X03 只使用 SIMS-valid。

## 入口与依赖

| 系统/阶段 | 入口 | 依赖 | 服务器产物 |
|---|---|---|---|
| E1 五折概率融合、校准和目标集推理 | `python -m emotioncore.x03.fusion` | `$R3/T1r_sims_valid/artifact_index_x03.json`、X02 冻结概率、折文件和缺脸规则 | `$R3/E1_v4` |
| E2 特征融合 | `python -m emotioncore.x03.e2` 及 E2 的后续校准入口 | `$R2/packs/A3`、`$R2/packs/V2`、`$R3/E2/cls_*`、折文件 | `$R3/E2` |
| 八帧 336×336 人脸视频 | `python -m emotioncore.x03.media` | X02 face receipts/crops、YuNet、原始媒体 | `$R3/media` |
| Q 预处理与训练 | fold 0：`python -m emotioncore.x03.q_train`；fold 1 起：`python -m emotioncore.x03.q_train_guarded`；编排：`python -m emotioncore.x03.q_pipeline` | 本地 Qwen2.5-Omni-7B 基座、冻结 manifest、H 盘输入缓存 | `$R3/Q`；缓存 `/mnt/h/emotionAI/emotioncore_models/X03/q_inputs` |
| 最长 AV 输入显存预检 | `python -m emotioncore.x03.q_worst_case` | 六份冻结训练 manifest；所有 AV 输入的 processor 缓存 | `$R3/Q/worst_case_preflight` |
| Q 三条件预测与校准 | `python -m emotioncore.x03.q_predict`、`python -m emotioncore.x03.q_calibrate` | Q 五折及最终 adapter、A/V/AV 推理清单 | `$R3/Q/predict_*`、`$R3/Q_calibrated`（预测完成后） |
| Q 融合与 H 五组合 | `python -m emotioncore.x03.h_fusion` | Q 原始 OOF/目标集概率和 X02 专家概率 | `$R3/H`（待执行） |
| 主表、配对差值及错误分析 | `python -m emotioncore.x03.final_analysis` | E1/E2/Q/H 校准产物、X02 质量标记 | `$R3/analysis`（待执行） |

每个正式运行的完整命令、输入校验和 Git 提交保存在其 `run_meta.json`。仓库只纳入汇总指标和路径/SHA-256，不纳入逐句概率或模型权重。

## 模型和当前门禁

- E1 已完成五组合及 T1r SIMS-valid 补推理；其概率级融合只需已冻结的专家输出。E2 已在 CPU 完成五折、最终模型、ET-test、SIMS-valid、逐折校准和单句延迟测量。E2 的 AVT 原始 OOF、ET-test、SIMS-valid 概率分别位于 `$R3/E2/eval/AVT/{oof,et_test,sims_valid}.jsonl`；完整指标位于同目录 `metrics.json`，原始路径及 SHA-256 也记录在其 `artifacts` 字段。
- Q 从只读 `/home/administrator/projects/emotion/omni_models/Qwen2.5-Omni-7B` 基座开始，NF4 双重量化、BF16、语言 LoRA rank 16 和视觉 LoRA rank 8，音频塔冻结。Q fold 0 已自然结束，固定 AV 内验证最佳 macro-F1 为 0.696626，通过 0.45 门限；按实测重估总 GPU 时长 50.5124 小时，通过 55 小时门限。fold 0 adapter 位于 `$R3/Q/fold0/best_adapter`，SHA-256 `aa1e213b1eac1ad0f1b42207e7b503b99fdf3e82d849c8f39d1bcf34ecce8b50`。
- fold 0 不受新增显存保护影响。fold 1 起 Q 训练/推理须使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；每次内验证另存可恢复的 `last`，异常最多自动重试 2 次，OOM 跳过及监控留下逐句/逐步记录。这些保护已实现，但 fold 1 尚未启动。15,281 条去重 AV 训练候选已全部缓存，最长 20 条实际输入长度为 1,016–1,104 token。
- 接入新门禁的启动器已通过 fold 0 的健全性和时长门禁。最长 20 条 AV 前后向预检在首条两次遭遇 WSL/CUDA 驱动错误，未取得有效峰值；按门禁停止在 fold 1 前。详细路径和日志 SHA-256 见 [停止收据](../experiments/X03_fusion/results/q_pipeline_stop.json)。修复运行环境后仍须对相同冻结 20 条逐条完成前后向，每条 `torch.cuda.max_memory_allocated` ≤30,000,000,000 字节且无 OOM，才可启动 fold 1。不得把 fold 0 adapter 当成正式跨臂结果。
- 臂 P 依 reviewer 批复取消；X02 已尝试 4bit，峰值 58,041.6 MiB。

X03 完成后，本文件须补齐六个 Q adapter 的实际路径和 SHA-256、各候选系统最终推理入口及最终产物索引。
