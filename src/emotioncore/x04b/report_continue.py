"""Report the reviewer revision while preserving original comparison JSON."""
import argparse,datetime as dt,json
from pathlib import Path
from emotioncore.x03.q_manifest import sha256,read_jsonl
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x04.report import export,replace_section,table,fmt

def run(args):
    git=git_provenance(Path.cwd());dest=args.repo/'experiments/X04b_q';r=dest/'results';body=(dest/'REPORT.md').read_text()
    state=export(args.root/'pipeline_progress.json',r/'continuation_progress.json')
    for f in ('run_meta.json','launch.json','budget_preflight.json','pipeline_stop.json'):
        if (args.root/f).exists():export(args.root/f,r/('continuation_'+f))
    write_json_atomic(r/'continuation_budget_progress.json',read_jsonl(args.root/'q_budget_progress.jsonl'))
    if (args.derived/'continue_tests.json').exists():export(args.derived/'continue_tests.json',r/'continue_tests.json')
    final=args.derived/'Q/final/summary.json'
    summaries={}
    for name in ('fold0','fold1','final'):
        p=args.derived/'Q'/name/'summary.json'
        if p.exists():
            summaries[name]=export(p,r/(name+'_summary.json'));export(p.parent/'run_meta.json',r/(name+'_run_meta.json'))
    body=replace_section(body,1,f"reviewer已批准续跑。当前状态{state['status']}，活动阶段{state.get('active_step')}；GPU已结束阶段累计{state['charged_gpu_hours']:.6f}小时。仅呈现证据，采用决定由reviewer作出。")
    body=replace_section(body,3,'2026-10-10 reviewer复核确认原停止执行正确，但撤销该跨模型校准统计量作为停止依据。仅续跑最终模型与目标推理，fold0/1训练和推理原样复用。P1改为同折内按说话人留一；原停止及两折互拟合数值保存在原JSON中。审查者已看到修订P1，采用决定实质依赖此前尚未观测的P2。修订原文见protocol/。\n\n状态：'+state['status']+'；'+str(state.get('reason','运行中，无新停止门禁。')))
    body=replace_section(body,5,table(['模型','最佳AV F1','最佳eval','adapter路径','SHA','OOM跳过'],[(k,fmt(v['best_av_macro_f1']),v['best_eval_index'],v['best_checkpoint'],v['best_checkpoint_sha256'],v['skipped_oom']) for k,v in summaries.items()])+'\n\n每0.25epoch固定700句；各summary保留完整曲线。默认分配器、三处检查点和last/OOM保护不变，最长20条AV预检已通过。')
    final_analysis=args.derived/'analysis_revised';p1=args.derived/'revised_p1'
    complete=(final_analysis/'run_meta.json').exists() and json.loads((final_analysis/'run_meta.json').read_text()).get('ended_at')
    data=final_analysis if complete else p1
    if (data/'run_meta.json').exists() and json.loads((data/'run_meta.json').read_text()).get('ended_at'):
        for f in ('run_meta.json','main_table.json','primary.json','secondary.json','raw_comparisons.json','source_artifact_index.json'):export(data/f,r/('revised_'+f))
        main=json.loads((data/'main_table.json').read_text());primary=json.loads((data/'primary.json').read_text());secondary=json.loads((data/'secondary.json').read_text());raw=json.loads((data/'raw_comparisons.json').read_text());old=json.loads((r/'interim_secondary.json').read_text());old.update(json.loads((r/'interim_primary.json').read_text()))
        def cell(v):return f"{v['difference']:.6f} [{v['left_minus_right_95'][0]:.6f}, {v['left_minus_right_95'][1]:.6f}]"
        pairs={**primary,**secondary}
        body=replace_section(body,6,table(['系统','原始OP-A不拟合','修订P1同折留一OP-N','原两折互拟合OP-N'],[(k,cell(raw[k]) if k in raw else '不适用（无预定义不拟合融合）',cell(pairs[k]),cell(old[k])) for k in ('Q_AV','Q_A','H_ref','H3')])+'\n\n同6533句/6说话人，bootstrap2000，seed20261007。每个留出说话人仅由同一Q版本、同一折其他两名说话人拟合；目标集两边均用两折合并OOF拟合，P2不变。原两折互拟合JSON逐字保留。\n\n'+(table(['P2 Pearson mix-et','差值','95%句区间'],[(k,fmt(v['difference']),v['difference_95']) for k,v in primary['P2'].items()]) if complete else 'P2待最终模型及目标推理完成。')+'\n\n采用规则由reviewer判断：修订P1 H_ref下界≥−0.01且P2 H_ref下界>0。审查者已见修订P1，修订不是盲态预登记。')
        body=replace_section(body,7,table(['系统','同折留一OP-A F1','同折留一OP-N F1','fear P/R/F1','disgust P/R/F1','surprise P/R/F1'],[(k,fmt(v['cv']['op_a']['macro_f1']),fmt(v['cv']['op_n']['macro_f1']),*[str([round(v['cv']['op_n']['per_class'][c][m],6) for m in ('precision','recall','f1')]) for c in ('fear','disgust','surprise')]) for k,v in main.items()])+'\n\n全部配对区间及逐类值见revised_secondary.json和revised_main_table.json。')
        if complete:
            body=replace_section(body,8,table(['系统','ET-test OP-N F1','中性误报','SIMS Pearson','Spearman','Acc-2','MER6 F1','MER5 F1'],[(k,fmt(v['et_test']['macro_f1']),fmt(v['et_test']['neutral_false_positive_rate']),fmt(v['sims_valid']['pearson']),fmt(v['sims_valid']['spearman']),fmt(v['sims_valid']['acc_2_nonzero']),fmt(v['mer_test']['op_n']['native6']['macro_f1']),fmt(v['mer_test']['op_n']['shared5']['macro_f1'])) for k,v in main.items()])+'\n\n'+table(['系统','ET说话人','F1','中性误报'],[(k,s,fmt(vv['macro_f1']),fmt(vv['neutral_false_positive_rate'])) for k,v in main.items() for s,vv in v['et_test']['per_speaker'].items()])+'\n\n全部目标指标见revised_main_table.json。目标标签零拟合；ET-test/MER-test只报告。')
        else:body=replace_section(body,8,'reviewer已授权最终模型目标推理：ET-test、SIMS-valid A/V/AV；MER-test A/AV六类和共享五类。尚未完成，不提前报告数值；目标参数仅由两折合并OOF拟合。')
    b=state['budget_projection'];body=replace_section(body,9,f"70 GPU小时总上限；承接原阶段32.407492小时。最新已结束阶段累计{state['charged_gpu_hours']:.6f}小时；总投影{b['projected_total_gpu_h']:.6f}小时（{b['at']}），含剩余20%余量及加载/诊断预留。活动阶段结束/停止后另计入账；预算日志见continuation_budget_progress.json。")
    body=replace_section(body,10,'原徒劳停止回执保留，reviewer已撤销其继续执行限制。续跑异常见continuation回执。'+str(state.get('reason','当前无新增停止。')))
    body=replace_section(body,11,'等待reviewer评估修订P1及P2，不由agent作采用决定。歧义见OPEN_QUESTIONS.md；审查者已见修订P1这一事实已披露。')
    (dest/'REPORT.md').write_text(body)
    h=args.repo/'docs/X05_HANDOFF.md';text=h.read_text().split('\n## X04b Q_mix')[0];text+='\n## X04b Q_mix\n\n2026-10-10 reviewer授权续跑最终模型，旧徒劳停止统计量不再适用。当前状态：'+state['status']+'。等待reviewer，不自行决定采用。\n\n仅复用fold0/1产物；P1同折内按说话人留一，P2和所有目标参数仍为两折合并OOF拟合。原两折互拟合与停止回执保留。新证据和源SHA见experiments/X04b_q/results/revised_*.json；最终adapter见final_summary.json（完成后）；逐句概率和媒体只在服务器。\n\n入口：emotioncore.x04b.continue_run / analysis --within-fold-loso。服务器续跑账本/home/administrator/emotion-data/emotioncore_runs/X04b/continuation，派生产物/mnt/h/emotionAI/emotioncore_runs/X04b。训练总上限70GPU小时，含已用32.407492。方法教训：不将另一模型单折校准迁移作为决策依据；未来同模型其他说话人或不少于四折合并OOF拟合。\n';h.write_text(text)
    write_json_atomic(r/'continuation_report_meta.json',{'git':git,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'report_sha256':sha256(dest/'REPORT.md'),'handoff_sha256':sha256(h),'analysis_complete':bool(complete)})
    write_json_atomic(r/'aggregate_index.json',{p.name:sha256(p) for p in r.glob('*.json') if p.name!='aggregate_index.json'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);run(p.parse_args())
if __name__=='__main__':main()
