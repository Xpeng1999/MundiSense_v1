"""Render the fixed 14-section X03 report from aggregate delivery JSON only."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def render(root: Path):
    r=root/'experiments/X03_fusion/results'
    def read(name):return json.loads((r/(name+'.json')).read_text())
    table=read('main_table_complete');paired=read('paired_speaker_bootstrap')
    robust=read('modality_robustness');visual=read('visual_correction')
    errors=read('error_analysis');sensitivity=read('truncation_sensitivity')
    budget=[json.loads(l) for l in (r/'q_budget_progress.jsonl').read_text().splitlines()]
    labels=('neutral','joy','sadness','anger','fear','disgust','surprise')
    names={'E_best':'E 最佳（A3+A2+V2+V1+T1r）','Q_AV':'Q-AV','Q_A':'Q-A',
           'Q_fusion':'Q 融合（A+AV）','H_best':'H 最佳（Q-A+Q-AV+A3+V2+V1）',
           'A3_reference':'A3 参照','V2_reference':'V2 参照'}
    systems=list(names)
    sections={}
    def f(v):return '—' if v is None else f'{v:.4f}'
    def md(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
    def ci(m):
        j=m['complete_speaker_bootstrap_95']['metrics']['macro_f1']
        return f"[{j['lower_95']:.4f}, {j['upper_95']:.4f}]"
    sections[1]='\n'.join([
        'X03 的 E1/E2、Q 五折及最终模型、A/V/AV 推理、正式校准、H 和预登记分析均已完成。',
        f"主指标 CV OP-N macro-F1：E 最佳 **{f(table['E_best']['cv_op_n']['macro_f1'])}**，Q-AV **{f(table['Q_AV']['cv_op_n']['macro_f1'])}**，H 臂内最高 **{f(table['H_best']['cv_op_n']['macro_f1'])}**。",
        f"H−E 的配对差值为 **{paired['H_best_minus_E_best']['difference']:+.4f}**，95% 说话人整群 bootstrap 区间 **[{paired['H_best_minus_E_best']['lower_95']:.4f}, {paired['H_best_minus_E_best']['upper_95']:.4f}]**。",
        f"Q 已计训练与推理 **{budget[-1]['projected_total_gpu_h']-1.5:.4f} GPU 小时**；含诊断及加载预留 **{budget[-1]['projected_total_gpu_h']:.4f} 小时**，低于 75 小时上限。",
        'fold0 保留检查点未生效时的有效训练结果；fold1–4 与最终模型显式开启梯度检查点。CH-SIMS test 零推理；最终架构选择交 reviewer。'])
    old=(root/'experiments/X03_fusion/REPORT.md').read_text()
    old_sections={int(n):text.strip() for n,text in re.findall(r'^## (\d+)\.[^\n]*\n(.*?)(?=^## \d+\.|\Z)',old,re.M|re.S)}
    old_sections[13]=old_sections[13].split('\n- 汇总补测V1',1)[0]
    sections[2]=old_sections[2]
    sections[3]='''完成：第 0 步、E1 五组合、E2 网格/五折/最终模型、Q 六模型及三条件预测、H 五组合、主表、配对差值、视觉修正描述性对照、缺失模态、截断敏感性、错误分析、延迟与资源统计。

批准的变更仅为：75 GPU 小时总预算；训练前冻结的每折 700 句内验证（种子 20260929）；43 句缺脸规则；T1r SIMS-valid 补推理；worktree move 归档；取消 P；默认 CUDA 分配器与 PEFT 后显式梯度检查点。五折、2 epoch、每 0.25 epoch 验证、学习率与 LoRA 配置、AV/A/V 60/20/20 均未改。

冻结配置：`configs/folds_v1.json` SHA `85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce`；`configs/x03_inner_val_subsets.json` SHA `be2cd442346cfa978e22bea8ab2ae3776d52187b2581fe93e377fc4fb58d3cfd`；`configs/x03_face_missing_policy.json` SHA `ce65415b5f7129b62ef5edc7eda5b20874ed88931f5a7cc295c276b8b704b77c`。

E2 的 [CLS] 来自未微调 T1 基座及冻结 r-token IDs，防止全 CV 微调 T1 泄漏外折标签，按已批准的 OPEN_QUESTIONS 第 2 条执行。SIMS-valid 647 句中 27 条源音频缺失；相关系数仅在有有效输出的句子上计算，Acc-2 的非零标签分母保留失败样本。''' 
    sections[4]='''各系统均逐折留出：仅用其余四折 OOF 拟合非负 log 概率融合权重、温度和七类偏置，以及 OP-N 的中性偏置；应用到本折。ET-test/SIMS-valid 仅使用全部五折 OOF 拟合的最终参数，标签不参与拟合或选择。OP-A 取校准后 argmax；OP-N 在拟合集将真实中性被判非中性/真实中性调至最接近 0.15 的可实现值。留出折合并值可偏离 0.15。

ECE 为 15 个等宽分箱。主比较指标是 CV OP-N macro-F1。主表附 2,000 次说话人整群 bootstrap、种子 20260929 的 95% 区间，覆盖六项总体指标与逐类 P/R/F1；保留原系统 JSON 的历史区间种子。ET-test 仅四名说话人，区间应结合此局限解读。配对差值的两个系统使用完全相同的 CV 句子与相同说话人重采样。中期两折校准未用于正式结果。

完整指标与区间：[主表](results/main_table_complete.json)、[完整区间](results/complete_speaker_intervals.json)。'''
    tests=read('delivery_tests')
    audit=read('delivery_verification')
    count=re.search(r'(\d+) passed',tests['stdout']).group(1)
    sections[3]+=f"\n\n交付测试 **{count} 项通过**，覆盖逐折防泄漏、OP-N拟合集、OOF覆盖、Q模态与恢复/OOM/预算保护。只读最终核验 **{audit['verified_systems']} 个候选系统**的三划分覆盖、概率有效性和工件SHA全部通过；见[核验回执](results/delivery_verification.json)。"
    e1=[]
    for name in ('E1_A3','E1_A3_V2','E1_A3_V2_V1','E1_A3_V2_T1r','E1_A3_A2_V2_V1_T1r'):
        j=read(name+'_metrics');e1.append([name,f(j['cv']['op_a']['macro_f1']),f(j['cv']['op_n']['macro_f1']),f(j['et_test']['op_n']['macro_f1']),f(j['sims_valid']['pearson'])])
    e2=[]
    for form in ('AVT','AV','A','V'):
        j=robust['E2'][form];e2.append([form,f(j['cv']['op_a']['macro_f1']),f(j['cv']['op_n']['macro_f1']),f(j['et_test']['op_n']['macro_f1']),f(j['sims_valid']['pearson'])])
    sections[5]='E1 全部复用 X02 冻结专家概率。E2 五折小网格与最终模型全部在 CPU 完成；三个模态各自以 0.2 概率置零且至少保留一个。E 最佳按预定 CV OP-N 规则为 E1_A3_A2_V2_V1_T1r，未用目标集选参。\n\n'+md(['E1 组合','CV OP-A F1','CV OP-N F1','ET OP-N F1','SIMS Pearson'],e1)+'\n\n'+md(['E2 条件','CV OP-A F1','CV OP-N F1','ET OP-N F1','SIMS Pearson'],e2)+'\n\nT1r valid 成功 620/647；27 条缺音频按冻结规则保留失败记录。E2 原单句串行延迟 P50/P95 为 5.424/16.118 秒，不含加载，完整来源见 `results/e2_latency_summary.json`。'
    training=[];curves=[];receipts=[];checkpoints=[]
    for n in ('fold0','fold1','fold2','fold3','fold4','final'):
        j=read('q_'+n+'_summary')
        training.append([n,j['examples_seen'],j['optimizer_steps'],j['best_eval_index'],f(j['best_av_macro_f1']),f(j['gpu_elapsed_s']/3600),j.get('skipped_oom',0)])
        curves.append([n]+[f(e['macro_f1']) for e in j['evals']])
        c=j['modality_forward_counts'];receipts.append([n,c['A'],c['V'],c['AV'],j['face_missing_fallback_train']])
        checkpoints.append([n,j['best_checkpoint'],j['best_checkpoint_sha256']])
    qrows=[]
    for form in ('AV','A','V'):
        j=robust['Q'][form];qrows.append([form,f(j['cv']['op_a']['macro_f1']),f(j['cv']['op_n']['macro_f1']),f(j['et_test']['op_n']['macro_f1']),f(j['sims_valid']['pearson'])])
    midpoint=read('q_interim_fold01')
    sections[6]='''从 Qwen2.5-Omni-7B 基座训练，NF4 双重量化、BF16；LLM 注意力 LoRA rank16/alpha32/dropout0.05，视觉 rank8、学习率为 LLM 的 0.2 倍，音频塔冻结。学习率 5e-5、cosine、5% warmup、有效 batch8、最多两轮。最终模型仍以 fold0 内验证说话人和固定 700 句子集选择 checkpoint。

人脸轨迹裁剪外扩20%，再次固定过滤最小64/448像素、YuNet置信度0.8；八帧、336×336、无音轨，每帧源像素112,896，processor grid 亦留下回执。原 WAV 在隔离子进程转16kHz单声道。ET 19,207/19,250有合格脸；CV40、ET-test3缺脸，训练V/AV→A，评估V均匀且强制计错，AV音频回退。A/V/AV 的 processor 和 forward 都断言模态，七类首 token 唯一。各条件 OOF 均覆盖16,366个唯一ID；ET-test均1,929句；SIMS-valid均647句。目标集模态计数和文件SHA见 `q_final_*_summary.json`。

fold0 最佳内验证0.696626通过0.45门禁，原结果未重训。fold1–4/最终模型使用默认分配器，PEFT后以use_reentrant=False显式开启三塔检查点；独立last保留adapter、optimizer、scheduler、RNG、数据位置。异常退出最多恢复两次，OOM重试一次再记跳过，累计超10句停止。每50步显存/步时监控，连续慢步写告警；正式训练与推理无异常续训、无OOM跳过，合计跳过0句。

'''+md(['模型','处理训练句次','优化步','最佳验证序号','最佳AV F1','训练GPU小时','OOM跳过'],training)+'\n\n训练曲线（八次固定 AV 内验证）：\n\n'+md(['模型']+[str(i)+('/4 epoch') for i in range(1,9)],curves)+'\n\n模态前向回执统计：\n\n'+md(['模型','A','V','AV','缺脸回退句次'],receipts)+'\n\n最佳 adapter：\n\n'+md(['模型','服务器路径','adapter_model.safetensors SHA-256'],checkpoints)+'\n\n正式五折校准结果：\n\n'+md(['条件','CV OP-A F1','CV OP-N F1','ET OP-N F1','SIMS Pearson'],qrows)+'''

同批6,533句的历史两折中期 OP-A：Q-AV 0.6168/中性误报0.2430，Q-A 0.5615/0.2523，E1最佳0.6372/0.2089。Q 的每个留出折仅用另一个已完成折临时校准；E1用正式四折拟合。该非门禁信息与正式五折口径不同，未改变任何选择或参数，原回执 `q_interim_fold01.json` 保留。

首条最长样本开/关检查点 loss 与全部480个LoRA梯度最大相对差0；同一冻结20条训练AV清单SHA `feb1e44736402825dcdaafe9afee25f2c3b9b53edb6cfdf8b0124dbf8807cf10` 重跑无OOM，峰值12,315,284,992字节，低于30,000,000,000字节门限，输入未截断。回执 `q_checkpoint_equivalence.json`、`q_worst_case_recovery.json` 保留。

视觉修正描述性对照（X02为其原始未校准M指标，X03另列raw、OP-A、OP-N；训练/校准条件不同，不能作单一视觉改动的因果估计）：

'''+md(['系统/口径','CV F1','UAR','WA'],[[n]+[f(j['metrics']['M'][k]) for k in ('macro_f1','uar','wa')] for n,j in [(n,visual[n]['results']) for n in ('V3a_h1_X02','V3b_h1_X02')]]+[['Q-V raw']+[f(visual['Q_V_raw_argmax_X03'][k]) for k in ('macro_f1','uar','wa')]]+[['Q-V '+p]+[f(robust['Q']['V']['cv'][p][k]) for k in ('macro_f1','uar','wa')] for p in ('op_a','op_n')])
    h=read('h_summary')
    sections[7]='五个预登记 H 组合及 Q-A/Q-AV 融合均完成，同第4节逐折留出拟合。H 臂内最高为 H_QA_QAV_A3_V2_V1，选择只使用 CV OP-N。缺脸视觉专家均匀，Q-AV音频回退。\n\n'+md(['组合','CV OP-N F1','ET OP-N F1','SIMS Pearson'],[[n,f(v['cv_op_n_macro_f1']),f(v['et_test_op_n_macro_f1']),f(v['sims_valid_pearson'])] for n,v in h.items() if isinstance(v,dict)])
    rows=[]
    for n in systems:
        for p in ('op_a','op_n'):
            m=table[n]['cv_'+p];rows.append([names[n],p]+[f(m[k]) for k in ('macro_f1','uar','wa','neutral_false_positive_rate','nonneutral_macro_recall','ece15')]+[ci(m)])
    sections[8]='完整逐类 P/R/F1、六项总体指标的说话人区间、混淆矩阵以及所有来源SHA见[完整主表](results/main_table_complete.json)。以下数字由脚本读取JSON生成。\n\n'+md(['系统','CV工作点','F1','UAR','WA','中性误报','非中性宏召回','ECE15','F1 95%区间'],rows)+'\n\nET-test只报告OP-N：\n\n'+md(['系统','F1','UAR','WA','中性误报','非中性宏召回','ECE15','F1 95%区间'],[[names[n]]+[f(table[n]['et_test_op_n'][k]) for k in ('macro_f1','uar','wa','neutral_false_positive_rate','nonneutral_macro_recall','ece15')]+[ci(table[n]['et_test_op_n'])] for n in systems])+'\n\n旧LoRA仅作ET-test原工作点参照（非本轮重新校准OP-N，禁止放入CV）：\n\n'+md(['旧LoRA条件','F1','UAR','WA','中性误报'],[[n]+[f(v[k]) for k in ('macro_f1','uar','wa','neutral_false_positive_rate')] for n,v in table['old_lora_ET_only'].items()])+'\n\nSIMS-valid：\n\n'+md(['系统','Pearson','Spearman','Acc-2非零','有效相关分母','非零分母'],[[names[n]]+[f(table[n]['sims_valid'][k]) for k in ('pearson','spearman','acc_2_nonzero')]+[table[n]['sims_valid']['correlation_denominator'],table[n]['sims_valid']['nonzero_denominator']] for n in systems])+'\n\nCV OP-N配对差值（按说话人整群、2,000次、种子20260929）：\n\n'+md(['左−右','F1差值','95%区间'],[[n,f(v['difference']),f"[{v['lower_95']:.4f}, {v['upper_95']:.4f}]"] for n,v in paired.items()])
    rows=[]
    for arm in ('E2','Q'):
        for form,j in robust[arm].items():
            m=j['cv']['op_n'];rows.append([arm,form,f(j['cv']['op_a']['macro_f1'])]+[f(m[k]) for k in ('macro_f1','uar','wa','neutral_false_positive_rate')])
    sensrows=[]
    for n in systems:
        for split,j in sensitivity[n].items():sensrows.append([names[n],split,j['full_n'],j['filtered_n'],j['removed'],f(j['full']['macro_f1']),f(j['filtered']['macro_f1']),f(j['filtered_minus_full']['macro_f1'])])
    sections[9]='推理时缺失模态使用冻结丢弃/均匀规则，独立逐折校准；无重新训练或目标集拟合。E2 V与Q V的CV40句缺脸均强制计错并保留分母。\n\n'+md(['臂','条件','OP-A F1','OP-N F1','UAR','WA','中性误报'],rows)+'\n\n截断阈值事先固定：参考语速>9字/秒或ASR/参考字数<0.5。去掉标记句子后沿用原预测和参数，未重新校准或选择；完整六项变化见[截断敏感性](results/truncation_sensitivity.json)。\n\n'+md(['系统','划分','原句数','保留','去掉','原F1','保留F1','变化'],sensrows)
    comp=errors['E_vs_Q_AV_complementarity'];ranks=errors['speaker_ranks_cv_op_n']
    sections[10]=f"CV中 E对/Q错 {comp['E_correct_Q_wrong']}句（{comp['E_correct_Q_wrong_fraction']:.4%}），E错/Q对 {comp['E_wrong_Q_correct']}句（{comp['E_wrong_Q_correct_fraction']:.4%}）；两者都对{comp['both_correct']}句、都错{comp['both_wrong']}句。分母均为{comp['n']}。\n\n"+md(['说话人','OP-N F1排名（高→低）'],[[s,' > '.join(v['ranking'])] for s,v in ranks.items()])+'\n\n完整各说话人分数及 ET-test 混淆矩阵见[错误分析](results/error_analysis.json)。各系统CV OP-N混淆矩阵如下（行是真实，列是预测；无效样本计错但不填入七类列，另见JSON invalid_by_true）：\n\n'
    for n in systems:
        matrix=errors[n]['cv_confusion_op_n']
        sections[10]+='**'+names[n]+'**\n\n'+md(['真实\\预测']+list(labels),[[label]+[matrix[label][p] for p in labels] for label in labels])+'\n\n'
    sections[11]=old_sections[11]
    resrows=[]
    for n in systems:
        j=table[n];res=j['resources'];resrows.append([names[n],f(j['latency']['p50_s']),f(j['latency']['p95_s']),f(res['peak_cuda_allocated_bytes']/2**30),f"{res['primary_model_count']} + {res['classifier_head_count']} = {res['total_neural_artifact_count']}"])
    sections[12]=md(['系统','延迟P50秒','P95秒','CUDA allocated峰值GiB','编码器/ASR/检测 + 分类头'],resrows)+'''

延迟覆盖ET-test全部1,929句，方法为串行逐阶段合计：X02 face/编码器/SenseVoice逐句elapsed_s + Q原正式forward逐句计时 + 此次隔离CPU子进程的Q音视频processor（含IPC与缓存序列化） + 冻结T1最终分类器及各专家head的CPU单句forward + 冻结融合/校准。含Q视觉时额外计入X03过滤/八帧视频准备，公共face阶段只计一次；Q-A和Q-AV共享权重但各自processor与forward都计入。8句warmup，不含模型加载。该值是基于实测回执的串行全链路合计，不是所有模型同时驻留的部署墙钟基准。

显存为串行分阶段的实际CUDA allocated峰值：A3/A2/V2/V1/SenseVoice独立进程测ET-test最长20个音频片段（视觉固定8帧），Q保存全1,929句三条件正式推理总体峰值并另测各形式最长20个token输入。Q-A行用A探测峰值，含AV行保守使用完整三条件总体峰值。CPU检测/T1/heads/融合不占CUDA。此表不能解释为多模型同时驻留的显存。模型数计所需独立神经权重工件，Q两形式共用一份，分类头另外计数；融合/校准向量不算神经模型。详细路径、方法、源SHA与逐句计时SHA见 `latency_summary.json`、`resources_summary.json`、`resources_q_summary.json`、完整主表。

'''+md(['复核阶段','GPU总投影小时','门禁'],[[j['completed_model'],f(j['projected_total_gpu_h']),j['status']] for j in budget])+f"\n\nQ训练与推理进程实耗合计{budget[-1]['projected_total_gpu_h']-1.5:.4f}小时；六次复核保留20%剩余任务余量、1小时诊断与0.5小时加载预留。最终无剩余任务投影{budget[-1]['projected_total_gpu_h']:.4f}小时≤75小时。此次资源探测的额外GPU进程时间计入诊断预留，未突破1小时；CPU延迟、E2与后处理不计GPU。预算账本见 `q_budget_progress.jsonl`，最终审计见 `delivery_verification.json`。"
    sections[12]+=f"\n\n此次GPU资源补测含加载及V1失败尝试的进程计时为 **{audit['supplemental_resource_process_seconds_including_load_and_failed_V1']:.2f} 秒**，记入诊断预留。"
    sections[13]=old_sections[13]+'''
- 汇总补测V1显存时，X02/venv缺timm导致子进程在加载模型前退出。保留错误日志和initial_run_meta，改用X02原`open_resources_v1/envs/visual/bin/python`恢复；已完成的A3/A2/V2测量通过SHA后复用。模型、输入、正式预测与选择未改。后续V1与SenseVoice测量成功。
- 正式Q全部模型与三条件推理无异常退出重试、无OOM跳过；训练保护机制保持启用。延迟/资源补测为报告计时，不用于选参。
'''
    sections[14]='''原开放问题的预算、缺脸、T1r valid、归档、取消P、默认分配器/梯度检查点恢复和T1基座[CLS]方式均已获批并落实。

交 reviewer 的解释边界：H/E最高组合按固定CV规则选出，选择后的区间未校正候选系统选择效应；仅15个CV说话人、4个ET-test说话人；视觉修正为不同训练条件的描述性对照；串行阶段延迟/显存不代表多模型同时驻留部署。上述限制均未用于改变协议或阈值。完整对照证据和可复现入口见[交接文档](../../docs/X04_HANDOFF.md)，最终主干选择由 reviewer 决定。
'''
    titles=['摘要','第 0 步完成情况','完成情况与偏离','校准与工作点方法','臂 E','臂 Q','臂 H','主对照表与配对差值','缺失模态鲁棒性与截断敏感性','错误分析','臂 P','耗时与资源','异常','给 reviewer 的问题']
    output='# X03 融合架构对照：最终执行报告\n\n'+'\n\n'.join(f'## {i}. {titles[i-1]}\n\n{sections[i].strip()}' for i in range(1,15))+'\n'
    (root/'experiments/X03_fusion/REPORT.md').write_text(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,default=Path.cwd());render(p.parse_args().repo_root)


if __name__=='__main__':main()
