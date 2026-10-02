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
- fold 0 训练时梯度检查点未实际生效，伴随 WSL 显存溢出；按 reviewer 批复保留其有效结果且不重训。fold 1 起 Q 训练/推理使用默认 CUDA 分配器，启动器清除 `PYTORCH_CUDA_ALLOC_CONF` 和 `PYTORCH_ALLOC_CONF`；模型在 `get_peft_model` 后以 `use_reentrant=False` 显式开启检查点，并断言 LLM、视觉塔、音频塔均为 True。每次内验证另存可恢复的 `last`，异常最多自动重试 2 次，OOM 跳过及监控留下逐句/逐步记录。
- 15,281 条去重 AV 训练候选已全部缓存，冻结最长 20 条长度为 1,016–1,104 token，清单 SHA-256 `feb1e44736402825dcdaafe9afee25f2c3b9b53edb6cfdf8b0124dbf8807cf10`。原始 WSL 故障见 [停止收据](../experiments/X03_fusion/results/q_pipeline_stop.json)；修复后的首条 [等价性回执](../experiments/X03_fusion/results/q_checkpoint_equivalence.json) 和同一 20 条 [显存预检回执](../experiments/X03_fusion/results/q_worst_case_recovery.json) 均通过，最大分配峰值 12,315,284,992 字节，低于 30,000,000,000 字节。
- 开启检查点后的 A/V/AV 单步外推曾超过原 55 小时门限，历史[停止收据](../experiments/X03_fusion/results/q_resume_budget_stop.json)保留。Reviewer 后续批准 **75 小时总上限**，含 fold 0 已用时、20% 剩余余量及诊断预留；每模型按实耗复核，超限须停在下一模型前。六个 Q 模型现已全部训练完成，五折 A/V/AV 外折概率各覆盖 16,366 句且 ID 唯一；最终模型目标集推理正在进行。最新完整预算投影为 **72.6248 小时**，见[逐模型回执](../experiments/X03_fusion/results/q_budget_progress.jsonl)。正式校准、H 与主表将在 Q 目标集预测齐备后生成，不作架构推荐。
- 臂 P 依 reviewer 批复取消；X02 已尝试 4bit，峰值 58,041.6 MiB。

## Q adapter 路径与 SHA-256

以下路径均来自完成训练的汇总回执；fold 0 保留原结果，fold 1–4 与最终模型显式开启梯度检查点。

| 模型 | adapter 路径 | SHA-256 |
|---|---|---|
| fold0 | `/home/administrator/emotion-data/emotioncore_runs/X03/Q/fold0/best_adapter` | `aa1e213b1eac1ad0f1b42207e7b503b99fdf3e82d849c8f39d1bcf34ecce8b50` |
| fold1 | `/home/administrator/emotion-data/emotioncore_runs/X03/Q/fold1/best_adapter_eval6` | `937832359fe53b5aff1de59bf4c1f5f2e6277a802e870686c13f8c3a78d0f5eb` |
| fold2 | `/home/administrator/emotion-data/emotioncore_runs/X03/Q/fold2/best_adapter_eval8` | `9f3823b162ee88eb0f557c6246c52cc538851fe07a205edb590c2487b3eb4eb6` |
| fold3 | `/home/administrator/emotion-data/emotioncore_runs/X03/Q/fold3/best_adapter_eval8` | `9287b9722f9cf0c571426e69c1dd607f62e28f0397e98675efaf4fac49913d3d` |
| fold4 | `/home/administrator/emotion-data/emotioncore_runs/X03/Q/fold4/best_adapter_eval6` | `5a23c9a273d08d980853e3991e87e2047701412a262fbde0d68301ad318d9e1a` |
| final | `/home/administrator/emotion-data/emotioncore_runs/X03/Q/final/best_adapter_eval8` | `427ebc148d447ba03dc747429323b440b10cbb8b4b8876247f48f71573d2d5a4` |

X03 完成后补齐各候选系统最终推理入口的完整命令及最终产物索引。
