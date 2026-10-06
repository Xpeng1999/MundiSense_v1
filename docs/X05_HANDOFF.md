# X05 输入交接 — X04a证据

本轮结束，等待reviewer。数据配方和X04b均未决定。

| 系统/臂 | 报告指标 | 冻结参数/概率目录 |
| --- | --- | --- |
| D0/A3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D0/A3 |
| D0/V2 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D0/V2 |
| D0/V1 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D0/V1 |
| D0/H_ref | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D0/H_ref |
| D0/H3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D0/H3 |
| D0/E3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D0/E3 |
| D1/A3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1/A3 |
| D1/V2 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1/V2 |
| D1/V1 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1/V1 |
| D1/H_ref | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1/H_ref |
| D1/H3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1/H3 |
| D1/E3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1/E3 |
| D2/A3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D2/A3 |
| D2/V2 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D2/V2 |
| D2/V1 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D2/V1 |
| D2/H_ref | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D2/H_ref |
| D2/H3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D2/H3 |
| D2/E3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D2/E3 |
| D3/A3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D3/A3 |
| D3/V2 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D3/V2 |
| D3/V1 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D3/V1 |
| D3/H_ref | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D3/H_ref |
| D3/H3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D3/H3 |
| D3/E3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D3/E3 |
| Q/Q_A | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/Q/Q_A |
| Q/Q_AV | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/Q/Q_AV |
| D1_audioonly/A3 | experiments/X04a_data/results/main_table.json | /mnt/h/emotionAI/emotioncore_runs/X04a/analysis_recovery/D1_audioonly/A3 |

CPU目标推理：`python -m emotioncore.x04.targets --arm D1 --encoder A3 --job /mnt/h/emotionAI/emotioncore_runs/X04a/heads/D1/A3 --derived /mnt/h/emotionAI/emotioncore_runs/X04a --output <H盘目录>`；其他臂/编码器替换相应参数。融合应用参数来自上述 metrics.json，不在新目标标签上再拟合。

汇总分析入口：`python -m emotioncore.x04.analysis --derived /mnt/h/emotionAI/emotioncore_runs/X04a --output <H盘目录>`。该命令从冻结OOF重建参数，不运行Q训练。Q只读X03 adapter SHA427ebc14…d5a4。完整输入/概率/头路径和SHA见 results/source_artifact_index.json、selected_configs.json、main_table.json；仓库不含媒体、字幕、特征或逐句概率。CH-SIMS test零模型推理。
