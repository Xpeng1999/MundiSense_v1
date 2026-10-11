# X05 输入交接 — X04a证据

以下保留X04a交接记录；X04b最终证据与待验收状态见文末。

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

## X04b Q_mix

2026-10-10 reviewer授权续跑最终模型，旧徒劳停止统计量不再适用。当前状态：complete。等待reviewer，不自行决定采用。

仅复用fold0/1产物；P1同折内按说话人留一，P2和所有目标参数仍为两折合并OOF拟合。原两折互拟合与停止回执保留。新证据和源SHA见experiments/X04b_q/results/revised_*.json；最终adapter见final_summary.json；逐句概率和媒体只在服务器。

入口：emotioncore.x04b.continue_run / analysis --within-fold-loso。服务器续跑账本/home/administrator/emotion-data/emotioncore_runs/X04b/continuation，派生产物/mnt/h/emotionAI/emotioncore_runs/X04b。训练总上限70GPU小时，含已用32.407492。方法教训：不将另一模型单折校准迁移作为决策依据；未来同模型其他说话人或不少于四折合并OOF拟合。

最终adapter：/mnt/h/emotionAI/emotioncore_runs/X04b/Q/final/best_adapter_eval8；SHA256 67c5d689952fe78b736dc628c7ddf7d883203b00ad4d3361e22260fe3c0794f3。

P1 H_ref（同折留一，mix−et）：0.001611 [-0.007283, 0.007172]。

P2 H_ref Pearson差：0.018667，95%区间[0.0102186721067476, 0.02679478326476998]；共同有效620/647，排除共同无效27，2000次句bootstrap，seed20261007。

P2 Q_AV Pearson差：0.045237，95%区间[0.028754138170467, 0.061448208360358705]；共同有效620/647，排除共同无效27，2000次句bootstrap，seed20261007。

全部GPU实耗51.614157小时；ET-test1929、SIMS-valid647均完成A/V/AV，MER-test572完成A/AV。训练、目标、SHA及独立bootstrap核验见final_*_verification.json；采用决定等待reviewer，PR #5保持未合入。
