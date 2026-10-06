# X03 → X04 交接

X03 的六个 Q 模型、E1/E2/H 和全部预登记分析已完成；架构选择由 reviewer 根据完整证据决定。当前进度见 [X03 报告](../experiments/X03_fusion/REPORT.md)。服务器运行根目录为 `/home/administrator/emotion-data/emotioncore_runs/X03`（下称 `$R3`），冻结 X02 运行根目录为 `/home/administrator/emotion-data/emotioncore_runs/X02`（下称 `$R2`）。

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
| Q 三条件预测与校准 | `python -m emotioncore.x03.q_predict`、`python -m emotioncore.x03.q_calibrate` | Q 五折及最终 adapter、A/V/AV 推理清单 | `$R3/Q/predict_*`、`$R3/Q_calibrated` |
| Q 融合与 H 五组合 | `python -m emotioncore.x03.h_fusion` | Q 原始 OOF/目标集概率和 X02 专家概率 | `$R3/H` |
| 主表、配对差值及错误分析 | `python -m emotioncore.x03.final_analysis` | E1/E2/Q/H 校准产物、X02 质量标记 | `$R3/analysis` |

每个正式运行的完整命令、输入校验和 Git 提交保存在其 `run_meta.json`。仓库只纳入汇总指标和路径/SHA-256，不纳入逐句概率或模型权重。

## 模型和当前门禁

- E1 已完成五组合及 T1r SIMS-valid 补推理；其概率级融合只需已冻结的专家输出。E2 已在 CPU 完成五折、最终模型、ET-test、SIMS-valid、逐折校准和单句延迟测量。E2 的 AVT 原始 OOF、ET-test、SIMS-valid 概率分别位于 `$R3/E2/eval/AVT/{oof,et_test,sims_valid}.jsonl`；完整指标位于同目录 `metrics.json`，原始路径及 SHA-256 也记录在其 `artifacts` 字段。
- Q 从只读 `/home/administrator/projects/emotion/omni_models/Qwen2.5-Omni-7B` 基座开始，NF4 双重量化、BF16、语言 LoRA rank 16 和视觉 LoRA rank 8，音频塔冻结。Q fold 0 已自然结束，固定 AV 内验证最佳 macro-F1 为 0.696626，通过 0.45 门限；按实测重估总 GPU 时长 50.5124 小时，通过 55 小时门限。fold 0 adapter 位于 `$R3/Q/fold0/best_adapter`，SHA-256 `aa1e213b1eac1ad0f1b42207e7b503b99fdf3e82d849c8f39d1bcf34ecce8b50`。
- fold 0 训练时梯度检查点未实际生效，伴随 WSL 显存溢出；按 reviewer 批复保留其有效结果且不重训。fold 1 起 Q 训练/推理使用默认 CUDA 分配器，启动器清除 `PYTORCH_CUDA_ALLOC_CONF` 和 `PYTORCH_ALLOC_CONF`；模型在 `get_peft_model` 后以 `use_reentrant=False` 显式开启检查点，并断言 LLM、视觉塔、音频塔均为 True。每次内验证另存可恢复的 `last`，异常最多自动重试 2 次，OOM 跳过及监控留下逐句/逐步记录。
- 15,281 条去重 AV 训练候选已全部缓存，冻结最长 20 条长度为 1,016–1,104 token，清单 SHA-256 `feb1e44736402825dcdaafe9afee25f2c3b9b53edb6cfdf8b0124dbf8807cf10`。原始 WSL 故障见 [停止收据](../experiments/X03_fusion/results/q_pipeline_stop.json)；修复后的首条 [等价性回执](../experiments/X03_fusion/results/q_checkpoint_equivalence.json) 和同一 20 条 [显存预检回执](../experiments/X03_fusion/results/q_worst_case_recovery.json) 均通过，最大分配峰值 12,315,284,992 字节，低于 30,000,000,000 字节。
- 开启检查点后的单步外推曾超过55小时，历史停止收据保留；随后获批75小时。六个Q模型、五折三条件OOF与最终三条件ET-test/SIMS-valid预测已全部完成。Q训练与预测进程计时68.0243 GPU小时，含1小时诊断及0.5小时加载预留69.5243小时；全部逐模型复核≤75小时。见[预算账本](../experiments/X03_fusion/results/q_budget_progress.jsonl)。
- 正式主表位于 `$R3/analysis_complete/main_table_complete.json`，路径与SHA见[最终产物索引](../experiments/X03_fusion/results/final_artifact_index.json)。H臂内最高组合是H_QA_QAV_A3_V2_V1，E臂内最高是E1_A3_A2_V2_V1_T1r；这两个名称仅表示预定CV规则的臂内结果。

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

## 最终系统与参数

| 候选 | raw概率/特征依赖 | 最终参数/神经权重 | 正式概率目录 |
|---|---|---|---|
| E1 五组合与E最佳 | A3_h2、A2_h2、V2_h2、V1_h1、T1_r（具体组合见metrics） | `$R3/E1_v4/<组合>/parameters_final.json` | `$R3/E1_v4/<组合>` |
| E2 AVT/AV/A/V | A3/V2帧特征；未微调T1基座CLS | `$R3/E2/final/final_model.pt`；校准参数在`$R3/E2/eval/<条件>/metrics.json` | `$R3/E2/eval/<条件>` |
| Q-A/Q-V/Q-AV | Q三条件raw输出，最终adapter见上表 | `$R3/Q_calibrated/Q_<条件>/metrics.json`的parameters_final | `$R3/Q_calibrated/Q_<条件>` |
| Q融合 | Q-A与Q-AV，共享Q模型、两次forward | `$R3/H/Q_fusion/metrics.json`的parameters_final | `$R3/H/Q_fusion` |
| H五组合与H最佳 | 固定Q/专家组合见metrics的components | `$R3/H/<组合>/metrics.json`的parameters_final | `$R3/H/<组合>` |
| A3参照/V2参照 | X02 A3_h2/V2_h2 | `$R3/E1_v4/E1_A3/parameters_final.json`；`$R3/V2_reference/metrics.json` | `$R3/E1_v4/E1_A3`；`$R3/V2_reference` |

每个目录的`{oof,et_test,sims_valid}.jsonl`留在服务器；仓库不纳入逐句概率或转写。完整主表与各系统metrics的artifacts字段记录路径/SHA，主表所需七系统的概率索引见final_artifact_index.json。E2原始推理命令见`$R3/E2/final/{et_test,sims_valid}_run_meta.json`；E2的最终head只在整个CV训练后用于目标集，不用于OOF CLS。

## 模型/特征依赖与环境

- A3：`$R2/models/A3`；A2：Qwen2.5-Omni基座的冻结audio_tower；V2：`$R2/models/V2`；V1：`/home/administrator/emotion-data/open_resources_v1/models/emotiefflib/enet_b0_8_best_afew.pt`。
- 专家最终分类头：`$R2/final/{A3_h2,A2_h2,V2_h2,V1_h1}/final_head.pt`。
- T1r：`$R2/models/T1`及`$R2/heads/T1_cv/final_model.pt`；SenseVoiceSmall：`/home/administrator/emotion-data/open_resources_v1/models/SenseVoiceSmall`。T1冻结token在`$R2/tokens/T1_r_{cv,test}.npz`；SIMS-valid另在`$R3/T1r_sims_valid`。
- E2 CV缓存：`$R2/packs/{A3,V2}`及`$R3/E2/cls_cv`；目标集pack/CLS在`$R3/E2/{target_et,target_sims,cls_test,cls_sims}`。
- 常规Python：`$R2/venv/bin/python`；V1需X02原视觉环境`/home/administrator/emotion-data/open_resources_v1/envs/visual/bin/python`（含timm）；SenseVoice需`/home/administrator/emotion-data/open_resources_v1/envs/audio/bin/python`。
- 所有模型复用本地固定权重。Q输入缓存在H盘；原项目目录只读。Q默认CUDA分配器、三塔显式检查点、`ulimit -n 65536`。E2始终CPU。

## 完整复现命令

在干净仓库运行，`PYTHONPATH=src`，使用独立输出目录。run_meta保存每次原始完整命令及依赖SHA。以下重放只使用已授权的ET-test/SIMS-valid与冻结OOF。

```bash
R2=/home/administrator/emotion-data/emotioncore_runs/X02
R3=/home/administrator/emotion-data/emotioncore_runs/X03
PY=$R2/venv/bin/python
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
ulimit -n 65536
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

# 最终Q的ET-test三条件raw推理
$PY -m emotioncore.x03.q_predict --run-name final_et_test \
  --manifest-meta $R3/Q/inference_manifests/run_meta.json \
  --adapter $R3/Q/final/best_adapter_eval8 \
  --adapter-sha256 427ebc148d447ba03dc747429323b440b10cbb8b4b8876247f48f71573d2d5a4 \
  --cache-root /mnt/h/emotionAI/emotioncore_models/X03/q_inputs \
  --output-dir $R3/replay/Q_et

# SIMS-valid同一adapter：--run-name final_sims_valid
# --manifest-meta $R3/Q/sims_valid_manifests/run_meta.json
# --output-dir $R3/replay/Q_sims

$PY -m emotioncore.x03.q_calibrate --prediction-root $R3/Q \
  --folds configs/folds_v1.json \
  --et-manifest experiments/X01_benchmark/results/et_utterances.jsonl \
  --output-dir $R3/replay/Q_calibrated

$PY -m emotioncore.x03.h_fusion --q-root $R3/Q \
  --artifact-index $R3/T1r_sims_valid/artifact_index_x03.json \
  --folds configs/folds_v1.json --face-policy configs/x03_face_missing_policy.json \
  --output-dir $R3/replay/H

$PY -m emotioncore.x03.fusion \
  --artifact-index $R3/T1r_sims_valid/artifact_index_x03.json \
  --face-policy configs/x03_face_missing_policy.json --output-dir $R3/replay/E1

$PY -m emotioncore.x03.final_analysis --e1-root $R3/E1_v4 \
  --e2-root $R3/E2/eval --q-cal-root $R3/Q_calibrated --h-root $R3/H \
  --v2-root $R3/V2_reference --quality $R2/quality/utterance_quality.jsonl \
  --folds configs/folds_v1.json \
  --et-manifest experiments/X01_benchmark/results/et_utterances.jsonl \
  --x01-metrics experiments/X01_benchmark/results/metrics.json \
  --output-dir $R3/replay/analysis
```

对冻结raw概率直接应用既有参数的API：E1/H用`fusion_logits → calibrated_logits → softmax`，OP-N额外加parameters_final.neutral_bias；Q/V2用`q_calibrate.apply_single`。这一步读取参数，不用目标标签拟合。新的原始媒体需按上述编码器/人脸/Q处理入口生成相同输入，不能把benchmark缓存时延当成实际在线服务时延。

## 交付核验

[固定14节报告](../experiments/X03_fusion/REPORT.md)、[完整主表](../experiments/X03_fusion/results/main_table_complete.json)、[差值](../experiments/X03_fusion/results/paired_speaker_bootstrap.json)、[质量敏感性](../experiments/X03_fusion/results/truncation_sensitivity.json)、[错误分析](../experiments/X03_fusion/results/error_analysis.json)、[核验回执](../experiments/X03_fusion/results/delivery_verification.json)。33项X03测试通过，覆盖逐折校准/融合防泄漏、OP-N拟合集、OOF覆盖、Q模态、恢复/OOM和预算保护。CH-SIMS test不推理。
