"""Publish X04b aggregates and fixed eleven report sections without raw output."""
import argparse,datetime as dt,json
from pathlib import Path
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03.q_manifest import sha256,read_jsonl
from emotioncore.x04.report import export,replace_section,table,fmt

def run(args):
    git=git_provenance(Path.cwd());dest=args.repo/'experiments/X04b_q';results=dest/'results';results.mkdir(parents=True,exist_ok=True);report=(dest/'REPORT.md').read_text()
    manifest=export(args.derived/'manifests/run_meta.json',results/'manifest_run_meta.json');budget=export(args.derived/'preflight/budget_preflight.json',results/'budget_preflight.json');export(args.derived/'preflight/run_meta.json',results/'budget_run_meta.json');export(args.derived/'first_token_receipt.json',results/'first_token_receipt.json')
    report=replace_section(report,4,'每epoch ET原清单全部 + 4693条MER分层子集；只做fold0/fold1/final。训练前九个首token均不同。ET直接沿用X03 prepare_target和模型result.loss；MER仅在首标签token位置用六类CE，其他目标token全词表CE，同样按目标token数平均。起点为NF4基座，不载入X03 adapter；默认分配器、三处显式检查点、last续训/OOM/步时保护保留。\n\n'+table(['模型','ET/epoch','MER/epoch','训练总句次','内验证'],[(n,s['epochs'][0]['ET_n'],4693,s['train_count'],700) for n,s in manifest['runs'].items()])+'\n\n清单/预算/缓存来源SHA见results；每epoch0/1种子20261007/20261008。分层舍入、形式分配与合并顺序详见OPEN_QUESTIONS。测试集均不进入训练和参数选择。')
    report=replace_section(report,9,f"启动前总GPU投影 **{budget['total_projection_gpu_hours']:.4f}小时**，上限70，门禁{'通过' if budget['passed'] else '停止'}。计20%余量、0.5小时加载/检查点、1小时诊断；X03检查点开启实测曲线和X04a MER长度来源均冻结，精确计算两epoch的分层子集与形式。每完成模型用实耗上调剩余投影，根分区可用低于80GB停止。\n\n"+table(['模型','训练+8次验证预测GPUh','全部预定推理GPUh'],[(n,fmt(v['training_validation_h']),fmt(v['prediction_h'])) for n,v in budget['runs'].items()]))
    if (args.derived/'reference_fit_smoke.json').exists():export(args.derived/'reference_fit_smoke.json',results/'reference_fit_smoke.json')
    if (args.derived/'tests.json').exists():export(args.derived/'tests.json',results/'tests.json')
    if (args.derived/'precheck/summary.json').exists():
        probe=export(args.derived/'precheck/summary.json',results/'precheck.json');export(args.derived/'precheck/run_meta.json',results/'precheck_run_meta.json')
        report=replace_section(report,5,f"最长20条AV预检{probe['gate']}；最大分配显存{probe['peak_allocated_bytes']:,}字节，门限30,000,000,000字节。清单SHA与逐条回执见results/precheck.json；预检零优化更新。")
    completions={}
    for n in ('fold0','fold1','final'):
        path=args.derived/'Q'/n/'summary.json'
        if not path.exists():continue
        summary=export(path,results/(n+'_summary.json'));meta=export(path.parent/'run_meta.json',results/(n+'_run_meta.json'));completions[n]=summary
    if completions:
        report=replace_section(report,5,table(['模型','最佳内验证AV F1','最佳验证次数','检查点路径','adapter SHA','OOM跳过句次'],[(n,fmt(s['best_av_macro_f1']),s['best_eval_index'],s['best_checkpoint'],s['best_checkpoint_sha256'],s['skipped_oom']) for n,s in completions.items()])+'\n\n每0.25epoch固定700句AV验证；全部验证曲线及模态统计见各summary。last与best独立保存，异常恢复及GPU耗时含在编排账本。')
        if (args.derived/'precheck/summary.json').exists():
            report=replace_section(report,5,report.split('## 5. ',1)[1].split('\n',1)[1].split('\n## 6.',1)[0].strip()+f"\n\n最长20条AV预检{probe['gate']}，峰值{probe['peak_allocated_bytes']:,}字节，门限30,000,000,000字节；回执见results/precheck.json。")
        for n in completions:
            path=args.derived/'Q'/n/'completion_verification.json'
            if path.exists():export(path,results/(n+'_completion_verification.json'))
    status=None
    if (args.root/'pipeline_progress.json').exists():
        status=export(args.root/'pipeline_progress.json',results/'pipeline_progress.json')
        if (args.root/'pipeline_stop.json').exists():export(args.root/'pipeline_stop.json',results/'pipeline_stop.json')
        if (args.root/'q_budget_progress.jsonl').exists():
            write_json_atomic(results/'q_budget_progress.json',read_jsonl(args.root/'q_budget_progress.jsonl'))
        if (args.root/'incidents.jsonl').exists():write_json_atomic(results/'incidents.json',read_jsonl(args.root/'incidents.jsonl'))
        if (args.root/'launch.json').exists():export(args.root/'launch.json',results/'launch.json')
        export(args.root/'run_meta.json',results/'pipeline_run_meta.json')
        report=replace_section(report,3,f"状态：{status['status']}。已完成模型{list(completions)}；活动阶段{status.get('active_step')}。不改变科学协议、不在ET-test/MER-test选择。"+(f"\n\n停止原因：{status['reason']}" if 'reason' in status else ''))
        report=replace_section(report,1,f"X04b状态：{status['status']}；已完成模型{list(completions)}，活动阶段{status.get('active_step')}。仅执行预登记的fold0、fold1和最终模型；是否采用Q_mix由reviewer决定。")
        resource=report.split('## 9. ',1)[1].split('\n',1)[1].split('\n## 10.',1)[0].strip()
        report=replace_section(report,9,resource+f"\n\n编排GPU累计账本{status['charged_gpu_hours']:.6f}小时（活动阶段结束/停止后入账；异常重试全计）。最新预算见results/q_budget_progress.json。")
        projection=status.get('budget_projection')
        if projection:
            resource=report.split('## 9. ',1)[1].split('\n',1)[1].split('\n## 10.',1)[0].strip()
            report=replace_section(report,9,resource+f"\n\n最新复核（{projection['at']}，{projection['completed_model']}）：总GPU投影{projection['projected_total_gpu_h']:.6f}小时，上限{projection['limit_h']}，门禁{'通过' if projection['passed'] else '停止'}；包含剩余阶段20%余量与加载/诊断预留。")
        incidents=read_jsonl(args.root/'incidents.jsonl') if (args.root/'incidents.jsonl').exists() else []
        report=replace_section(report,10,f"编排异常退出回执{len(incidents)}次，见results/incidents.json。" if incidents else '暂无异常退出。')
        if status.get('reason'):report=replace_section(report,10,f"停止时间：{status.get('stopped_at')}。停止原因：{status['reason']}。原停止及异常回执保留，未自行放宽门限。")
    interim=args.derived/'interim'
    if (interim/'futility.json').exists():
        gate=export(interim/'futility.json',results/'futility.json');export(interim/'run_meta.json',results/'interim_run_meta.json')
        for filename in ('main_table.json','primary.json','secondary.json','source_artifact_index.json'):
            export(interim/filename,results/('interim_'+filename))
        report=replace_section(report,6,f"两折互拟合中期证据已生成，全部指标与2000次说话人bootstrap见results/interim_primary.json及interim_main_table.json。Q-AV OP-N差值{gate['point_difference']:.6f}，徒劳门限−0.02，{'停止' if gate['stop'] else '通过'}。P2须最终模型完成后计算；未用目标标签调参。")
        report=replace_section(report,7,'两折Q-AV、Q-A、H3配对差值与区间见results/interim_secondary.json，逐类值见interim_main_table.json。')
        if gate['stop']:report=replace_section(report,1,'fold1后徒劳门禁触发，按预登记协议停止，未训练最终模型。是否采用Q_mix由reviewer决定。')
    analysis=args.derived/'analysis'
    if (analysis/'run_meta.json').exists() and json.loads((analysis/'run_meta.json').read_text()).get('ended_at'):
        main=export(analysis/'main_table.json',results/'main_table.json');primary=export(analysis/'primary.json',results/'primary.json');secondary=export(analysis/'secondary.json',results/'secondary.json');export(analysis/'source_artifact_index.json',results/'source_artifact_index.json');export(analysis/'run_meta.json',results/'analysis_run_meta.json')
        p1=primary['H_ref'];p2=primary['P2']
        report=replace_section(report,1,'X04b计算完成：Q_mix fold0、fold1和最终模型，从基座训练；两epoch各4693条MER。完成预登记两折互拟合及P1/P2、跨语料和逐类证据。是否采用Q_mix由reviewer决定，未自行选择。')
        report=replace_section(report,6,table(['主比较','差值','95%区间','重采样单位'],[('P1 H_ref mix-et',fmt(p1['difference']),p1['left_minus_right_95'],'6名说话人'),*[(f'P2 SIMS {k} Pearson mix-et',fmt(v['difference']),v['difference_95'],'句') for k,v in p2.items()]])+'\n\n2000次、seed20261007。Q_mix与Q_et都仅用fold0/1互为拟合集，专家全为冻结X04a D1；目标参数用6533句合并拟合。采用规则预登记：P1区间下界≥−0.01且P2 H_ref下界>0；本报告只给证据，采用决定留给reviewer。')
        report=replace_section(report,7,table(['次要两折对比','OP-N F1差','95%说话人区间'],[(k,fmt(v['difference']),v['left_minus_right_95']) for k,v in secondary.items()])+'\n\n'+table(['系统','CV OP-A F1','CV OP-N F1','fear P/R/F1','disgust P/R/F1','surprise P/R/F1'],[(k,fmt(v['cv']['op_a']['macro_f1']),fmt(v['cv']['op_n']['macro_f1']),*[str([round(v['cv']['op_n']['per_class'][c][t],4) for t in ('precision','recall','f1')]) for c in ('fear','disgust','surprise')]) for k,v in main.items()]))
        report=replace_section(report,8,table(['系统','ET-test OP-N F1','ET-test中性误报','SIMS Pearson','MER6 OP-N F1','MER5共享 OP-N F1'],[(k,fmt(v['et_test']['macro_f1']),fmt(v['et_test']['neutral_false_positive_rate']),fmt(v['sims_valid']['pearson']),fmt(v['mer_test']['op_n']['native6']['macro_f1']),fmt(v['mer_test']['op_n']['shared5']['macro_f1'])) for k,v in main.items()])+'\n\n'+table(['系统','ET-test说话人','macro-F1','中性误报'],[(k,s,fmt(m['macro_f1']),fmt(m['neutral_false_positive_rate'])) for k,v in main.items() for s,m in v['et_test']['per_speaker'].items()])+'\n\n目标集零拟合；SIMS-valid仅提供预登记跨语料采用证据，不调超参/温度/阈值。MER只报告，移除worried真值后442句共享类；不作硬映射。全部指标/参数/逐类值见results/main_table.json。')
        handoff=args.repo/'docs/X05_HANDOFF.md';body=handoff.read_text().split('\n## X04b Q_mix')[0];body+='\n## X04b Q_mix\n\n本轮仅证据，等待reviewer。专家使用X04a D1；Q_mix从基座训练fold0/fold1/final。\n\n'+table(['系统','冻结评分和参数目录'],[(k,str(analysis/k)) for k in main])+'\n\n推理入口沿用 emotioncore.x03.q_predict（ET/SIMS A/V/AV）和 emotioncore.x04b.predict_mer（MER A/AV六类）。adapter路径与SHA见experiments/X04b_q/results/final_summary.json；全部source/概率路径与SHA见source_artifact_index及main_table。两折最终融合参数只由6533句ET OOF拟合，不接触目标标签。\n';handoff.write_text(body)
    (dest/'REPORT.md').write_text(report);write_json_atomic(results/'report_run_meta.json',{'git':git,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'report_sha256':sha256(dest/'REPORT.md')})
    write_json_atomic(results/'aggregate_index.json',{p.name:sha256(p) for p in results.glob('*.json') if p.name!='aggregate_index.json'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);run(p.parse_args())
if __name__=='__main__':main()
