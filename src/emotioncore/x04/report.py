"""Publish only aggregate X04a receipts and generate the fixed report sections."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
from emotioncore.x03.q_manifest import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


def read(path): return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(str(v) for v in row)+" |" for row in rows])


def fmt(value): return "—" if value is None else f"{value:.4f}"


def replace_section(text, number, body):
    marker = f"## {number}. "
    start = text.index(marker)
    heading_end = text.index("\n", start)
    stop = text.find("\n## ", heading_end)
    if stop < 0: stop = len(text)
    return text[:heading_end]+"\n\n"+body.rstrip()+"\n"+text[stop:]


def assert_aggregate(value):
    """Catch accidental learned outputs/text export before any repository write."""
    if isinstance(value, dict):
        forbidden = {"probabilities", "probabilities8", "op_a_probabilities", "op_n_probabilities",
                     "chinese", "text", "normalized", "crop_paths", "preview_path", "frames"}
        if forbidden & value.keys(): raise ValueError("Private data detected in aggregate report")
        for child in value.values(): assert_aggregate(child)
    elif isinstance(value, list):
        for child in value: assert_aggregate(child)


def export(path, target):
    payload = read(path); assert_aggregate(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Copy exact bytes: the committed budget must match the actual preflight.
    target.write_bytes(path.read_bytes())
    return payload


def run(args):
    provenance=git_provenance(Path.cwd())
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    dest = args.repo/"experiments/X04a_data"; results = dest/"results"
    registration = export(args.registration/"data_registration.json", results/"data_registration.json")
    archive = export(args.registration/"archive_receipt.json", results/"archive_receipt.json")
    export(args.registration/"prepare_run_meta.json", results/"prepare_run_meta.json")
    budget = export(args.preflight/"budget_preflight.json", results/"budget_preflight.json")
    dedup = export(args.preflight/"dedup/dedup_report.json", results/"dedup_report.json")
    export(args.preflight/"dedup/run_meta.json", results/"dedup_run_meta.json")
    if not budget["passed"] or budget["total_projection_gpu_hours"] > 20:
        raise ValueError("Budget gate is closed; formal report cannot imply continuation")
    cohort = registration["cohorts"]
    mer_train = cohort["MER_train"]; mer_test = cohort["MER_test"]; crema = cohort["CREMA_train"]
    dedup_manifest = args.derived/"manifests/mer_train_dedup.jsonl"
    # Counts only; no subtitle access or manifest copy into repository.
    counts = {}
    for line in dedup_manifest.read_text().splitlines():
        row = json.loads(line); counts[row["label"]] = counts.get(row["label"], 0)+1
    face = None
    evidence = args.derived/"data_evidence/face_evidence.json"
    if evidence.exists(): face = export(evidence, results/"face_evidence.json")
    if face:
        export(args.derived/"data_evidence/run_meta.json",results/"face_evidence_run_meta.json")
        export(args.derived/"full/face/timing.json",results/"face_timing.json")
    licenses = {}
    for name, path in (("MER", Path("/mnt/h/emotion-data/MER2026/README.md")),
                       ("CREMA", Path("/home/administrator/emotion-data/emotionAI_dataset/CREMA-D/LICENSE.txt"))):
        licenses[name] = {"path": str(path), "sha256": sha256(path)}
    (results/"license_evidence.json").write_text(json.dumps(licenses, indent=2)+"\n")
    card_rows = [(name, c["count"], json.dumps(c["classes"], ensure_ascii=False),
                  c["duration_s"]["min"], c["duration_s"]["p50"], c["duration_s"]["p90"], c["duration_s"]["max"])
                 for name, c in (("MER-train 原始", mer_train), ("MER-test", mer_test))]
    card = "# MER2026 数据卡 — X04a\n\n"+table(["划分", "n", "项目标签分布", "最短秒", "P50秒", "P90秒", "最长秒"], card_rows)
    card += f"\n\n去重后的训练集：{dedup['mer_train_after']} 条，标签分布 {json.dumps(counts,ensure_ascii=False)}。"
    card += "\n\n只用 MER2026，不重复加入已包含的 MER2025。训练与测试 ID 不相交；MER-test 与 MER-train 不保证说话人独立。572 条测试不参与训练、早停或任何参数选择。"
    card += "\n\n视频分辨率及各分辨率计数见 results/data_registration.json（MER_train/MER_test.video_resolutions）。视频采样时长取画面流，避免附加 data 流延长容器时长；音频时长分布取 WAV 流或容器回退值。"
    card += "\n\nffprobe 注册：全部音视频探测成功；此为头部探测，不能替代全解码。"
    if face:
        card += "\n\n"+table(["划分", "处理状态", "无脸比例", "多脸比例", "目标/第二轨迹面积比P10/P50/P90", "帧读取失败数"],
            [(s, json.dumps(face["cohorts"][s]["status_counts"]), fmt(face["cohorts"][s]["no_face_fraction"]),
              fmt(face["cohorts"][s]["multi_face_fraction"]), json.dumps(face["cohorts"][s]["area_ratio"]),
              face["cohorts"][s]["decode_failures"]) for s in ("MER_train", "MER_test")])
        card += f"\n\n随机多脸联系表：{face['contact_sheets']['sample_count']} 条，种子20261006；服务器 {args.derived}/MER_contact_sheets/selection.json。图像不入库，路径/SHA见 results/face_evidence.json。"
    else:
        card += "\n\n全量解码、人脸统计与随机60条多脸联系表待全量冻结 X02 管线完成后补齐。"
    card += f"\n\n许可证据：服务器 {licenses['MER']['path']}，SHA `{licenses['MER']['sha256']}`。README 标为 CC-BY-NC-4.0，并明确限定学术研究/挑战参与；数据、标注和派生文件未经书面许可不得再分发、镜像或用于商业用途。本仓库仅公开汇总证据和产物路径/SHA。\n"
    (args.repo/"docs/DATA_CARD_mer2026.md").write_text(card)
    (args.repo/"docs/DATA_CARD_cremad.md").write_text("# CREMA-D 数据卡 — X04a\n\n"+
        f"{crema['count']} 条，{crema['speakers']} 名说话人；只用于训练，零评估。\n\n"+
        table(["项目标签", "n"], sorted(crema["classes"].items()))+
        "\n\n标签按官方文件名 ANG/DIS/FEA/HAP/NEU/SAD，HAP→joy、SAD→sadness。音视频头部探测全部成功；分辨率均480×360。\n\n"+
        f"许可证：本地 LICENSE.txt 声明数据库使用 ODbL，数据库个体内容使用 DBCL。证据路径 {licenses['CREMA']['path']}，SHA `{licenses['CREMA']['sha256']}`。\n")
    registry = args.repo/"docs/DATA_REGISTRY.md"
    marker = "\n## X04a 注册补充\n"
    original = registry.read_text().split(marker)[0]
    registry.write_text(original+marker+"\n"+table(["语料", "本轮角色", "n", "数据卡"], [
        ("MER2026 train", "只训练；保守文本去重后", dedup["mer_train_after"], "DATA_CARD_mer2026.md"),
        ("MER2026 test", "只报告，禁止选择", mer_test["count"], "DATA_CARD_mer2026.md"),
        ("CREMA-D", "只训练", crema["count"], "DATA_CARD_cremad.md")])+
        "\n\nMER 媒体及派生文件留在 H 盘；MER2025 不另加入。CH-SIMS 任意划分均不训练，test 零模型推理。Benchmark v1.1 保留原 ET/SIMS 定义全文不变。\n")
    report = (dest/"REPORT.md").read_text()
    summary = table(["队列", "n", "用途"], [("MER-train 原始",9395,"注册"),
        ("MER-train 去重后",dedup["mer_train_after"],"训练"), ("MER-test",572,"只报告"), ("CREMA-D",7442,"训练")])
    summary += "\n\n"+table(["MER-train 对比方向", "候选对", "排除ID数", "指纹确认ID数"],
        [(name, spec["candidate_pairs"],spec["excluded_mer_ids"],spec["confirmed_mer_ids"]) for name,spec in dedup["directions"].items()])
    test_fraction = dedup["directions"]["SIMS_test"]["candidate_mer_ids"]/dedup["mer_train_before"]
    summary += f"\n\n与 CH-SIMS test 的文本候选占 {test_fraction:.6%}，低于5%门限；共排除{len(dedup['excluded_ids'])}个不同 MER-train ID。仅读取SIMS四列，零模型推理；指纹工具可用，确认方法为完整原始序列相等，未确认不改变保守文本排除。缺失字幕{dedup['missing_subtitle_count']}条。详见数据卡及 results/dedup_report.json。"
    if face:
        summary += "\n\n人脸管线完整回执：\n\n"+table(
            ["队列","n","状态","帧读取失败数","多脸比例"],
            [(name,c['clips'],json.dumps(c['status_counts']),c['decode_failures'],fmt(c['multi_face_fraction']))
             for name,c in face['cohorts'].items()])
        summary += f"\n\n随机多脸联系表{face['contact_sheets']['sample_count']}条、种子{face['contact_sheets']['seed']}，图像只存服务器。帧读取失败不等于整句失败；无脸按冻结规则保留分母并计为错误。"
    report = replace_section(report, 4, summary)
    report = replace_section(report, 3, "第0步完成；注册、保守去重和相同200句的修复后预算预检完成。正式运行只在本预算及报告提交并部署干净工作区后启动。A3头按最新批复使用cuda:0串行训练，V1/V2头CPU并行，Q仅补MER-test A/AV；全量结果尚待执行。")
    resources = f"相同200句修复后预检通过：特征与Q推理GPU投影 **{budget['total_projection_gpu_hours']:.3f}小时**（上限20）；此项尚不包含新批复的A3头训练。其中特征{budget['extraction_projection_hours']:.3f}小时、Q MER-test {budget['Q_mer_test_projection_hours']:.3f}小时，计20%余量、10分钟加载预留及该次预检GPU实耗。原v2预检消耗另记入pipeline GPU账本。CPU人脸处理投影{budget['CPU_face_projection_hours']:.3f}小时。预算原回执SHA `{sha256(args.preflight/'budget_preflight.json')}`。\n\nA3头使用cuda:0，fp16 mmap来源、单个128MiB锁页批次缓冲、index_select和non_blocking传输后转float32；新取数须torch.equal通过。首个外折完整网格后重新估算全部A3臂与总墙钟并写wall_time_estimate.json；总GPU投影超过20小时即停，不开始下一模型。批次顺序、网格、种子、早停和α不变。V1/V2头CPU并行。\n\nCPU分类头旧估算3–5天仅为历史CPU计划，已由最新GPU设备批复替代，不能当作当前ETA。每小时自动推进至报告和PR交付后停止。全部新增派生文件在H盘；根分区门限80,000,000,000字节。"
    if args.pipeline_root:
        progress_path=args.pipeline_root/'pipeline_progress.json'
        if progress_path.exists():
            progress=export(progress_path,results/'execution_progress.json')
            if 'resumed_from' in progress:
                previous=progress
                while 'resumed_from' in previous:
                    spec=previous['resumed_from'];path=Path(spec['path'])
                    if sha256(path)!=spec['sha256']:raise ValueError('Pipeline history SHA changed')
                    previous=export(path,results/f"pipeline_stop_{read(path)['git']['commit'][:8]}.json")
                export(path,results/'pipeline_original_stop.json')
            from emotioncore.x04.pipeline import completed_head
            completions={}
            for arm in ('D0','D1','D2','D3','D1_audioonly'):
                for encoder in ('A3','V2','V1'):
                    job=args.derived/f'heads/{arm}/{encoder}'
                    meta=completed_head(job)
                    if not meta:continue
                    result=read(job/'results.json')
                    completions[f'{arm}/{encoder}']={
                        'run_meta':{'path':str(job/'run_meta.json'),'sha256':sha256(job/'run_meta.json')},
                        'results':{'path':str(job/'results.json'),'sha256':sha256(job/'results.json')},
                        'git':meta['git'],'started_at':meta['started_at'],'ended_at':meta['ended_at'],
                        'elapsed_s':meta['elapsed_s'],'device':meta['device'],
                        'oof_sha256':result['oof_sha256'],'checkpoint_sha256':result['checkpoint_sha256'],
                        'folds':[{'fold':s['fold'],'outer_count':s['outer_count'],'selected':s['selected']}
                                 for s in result['folds']],'final_config':result['final_config']}
            assert_aggregate(completions)
            write_json_atomic(results/'heads_complete.json',completions)
            rendered=[]
            for stage in progress['steps']:
                parts=stage['name'].split('_')
                key='/'.join(parts[1:]) if len(parts)==3 and parts[0]=='heads' else None
                finished=completions.get(key)
                status='完成（产物SHA核验）' if finished else (str(stage['returncode']) if 'returncode' in stage else '运行中')
                elapsed=finished['elapsed_s'] if finished else stage.get('elapsed_s')
                if elapsed is None and stage.get('reused_from'):duration='复用；原回执见索引'
                else:duration=fmt(elapsed/3600) if elapsed is not None else '待结束'
                rendered.append((stage['name'],status,duration,'CPU' if stage['cpu_only'] else 'GPU'))
            stages=table(["阶段","状态","实测墙钟小时","设备"],
                rendered)
            report=replace_section(report,3,
                f"第0步、注册/去重和200句预检完成；正式管线使用干净提交{progress['git']['commit']}，启动{progress['started_at']}，当前状态{progress['status']}。\n\n"+
                stages+"\n\nA3头cuda:0单进程、V1/V2头CPU并行，Q本轮零训练。未完成阶段的结果仍待执行。")
            resources += f"\n\n正式管线已结束阶段的GPU累计账本{progress.get('charged_gpu_hours',0):.6f}小时（包括预检与验证预留）；当前GPU阶段的用时在结束或停止时加入。阶段0表示成功返回。\n\n"+stages
            for encoder in ('A3','V2','V1'):
                path=args.derived/f'full/features/{encoder}/run_meta.json'
                if path.exists() and 'ended_at' in read(path):
                    payload=read(path);request_seconds=payload.pop('request_seconds',[])
                    payload['source_run_meta']={'path':str(path),'sha256':sha256(path)}
                    payload['request_count']=len(request_seconds)
                    assert_aggregate(payload)
                    write_json_atomic(results/f'{encoder}_features_complete.json',payload)
        wall_path=args.pipeline_root/'first_grid_budget/wall_time_estimate.json'
        if wall_path.exists():
            export(wall_path,results/'wall_time_estimate.json')
            report=replace_section(report,12,(args.pipeline_root/'first_grid_budget/REPORT.md').read_text().split('## 12. ',1)[1].split('\n',1)[1].split('\n## 13.',1)[0].strip())
            resources=report.split('## 12. ',1)[1].split('\n',1)[1].split('\n## 13.',1)[0].strip()
            if progress_path.exists():
                resources += f"\n\n正式管线已结束阶段GPU累计账本{progress.get('charged_gpu_hours',0):.6f}小时；当前GPU阶段结束或停止时加入。\n\n"+stages
    report = replace_section(report,12,resources)
    report=replace_section(report,5,"统一8维h2，语料掩码及α网格不变。A3头cuda:0单进程，V1/V2头CPU并行；源特征fp16，只锁页单批128MiB缓冲。批次顺序、seed20260928、网格、早停和最终选参与X02一致；GPU取数与X02精确相等回执见results/gpu_batch_equivalence.json。")
    early_repro=args.derived/'d0_reproduction/d0_reproducibility.json'
    if early_repro.exists() and not args.complete:
        repro=export(early_repro,results/'d0_reproducibility.json')
        export(args.derived/'d0_reproduction/run_meta.json',results/'d0_reproduction_run_meta.json')
        report=replace_section(report,6,table(['编码器','逐句argmax一致率','本轮原始F1','X02原始F1','差值'],
            [(k,fmt(v['argmax_agreement']),fmt(v['new_raw']['macro_f1']),fmt(v['old_raw']['macro_f1']),fmt(v['macro_f1_difference']))
             for k,v in repro.items()])+'\n\n使用原始头输出对比；正式评估另执行冻结缺脸规则。仅复算已有OOF，本步骤零训练、零模型推理，不据此改变选择或参数。')
    if args.complete:
        analysis_root = args.analysis_root or args.derived/"analysis"
        analysis_meta = export(analysis_root/"run_meta.json", results/"analysis_run_meta.json")
        if not analysis_meta.get('ended_at'):raise ValueError('Analysis has not completed')
        for name, digest in analysis_meta['outputs'].items():
            if sha256(analysis_root/name)!=digest:raise ValueError('Analysis output SHA changed')
        for name in ('analysis_recovery_launch.json', 'analysis_recovery_tests.json', 'final_artifact_validation.json'):
            source = args.pipeline_root.parent/name
            if source.exists():export(source, results/name)
        export(args.pipeline_root/'pipeline_stop.json',results/'pipeline_stop_970ff48.json')
        export(args.pipeline_root/'run_meta.json',results/'pipeline_run_meta.json')
        main = export(analysis_root/"main_table.json", results/"main_table.json")
        paired = export(analysis_root/"paired_differences.json", results/"paired_differences.json")
        repro = export(analysis_root/"d0_reproducibility.json", results/"d0_reproducibility.json")
        selected = export(analysis_root/"selected_configs.json", results/"selected_configs.json")
        export(analysis_root/"source_artifact_index.json", results/"source_artifact_index.json")
        qb = export(args.derived/"budget_estimate/x04b_budget.json", results/"x04b_budget.json")
        export(args.derived/"budget_estimate/run_meta.json", results/"x04b_budget_run_meta.json")
        export(args.derived/"Q_predictions/run_meta.json", results/"q_predictions_run_meta.json")
        q_media = export(args.derived/"Q_media/summary.json", results/"q_media_summary.json")
        report = replace_section(report,1,"X04a计算完成：D0–D3×A3/V2/V1五折及最终模型、冻结Q的MER-test A/AV推理和三套融合。\n主比较均使用本轮D0，V1为h2。全部跨语料只报告，Q零训练。\n不决定数据配方，提交reviewer验收；本轮X04a结束。")
        report = replace_section(report,3,"注册、去重、200句预算门禁、12组专家头五折/最终模型、Q MER-test A/AV、融合及各项分析完成。敏感性仅D1 A3；详见回执、OPEN_QUESTIONS和第13节。等待reviewer验收，未运行X04b。原编排器在CPU分析计数写入异常后停止；独立干净工作区完成CPU分析恢复，原停止回执保留，详见第13节。")
        report = replace_section(report,6,table(["编码器","逐句argmax一致率","本轮原始F1","X02原始F1","差值"],[(k,fmt(v['argmax_agreement']),fmt(v['new_raw']['macro_f1']),fmt(v['old_raw']['macro_f1']),fmt(v['macro_f1_difference'])) for k,v in repro.items()])+"\n\n使用原始头输出对比；正式评估另执行冻结缺脸规则。")
        def et_table(keys):
            return table(["系统/臂","CV OP-A F1","CV OP-N F1","CV OP-N中性误报","ET-test OP-N F1"],[(k,fmt(main[k]['cv']['op_a']['macro_f1']),fmt(main[k]['cv']['op_n']['macro_f1']),fmt(main[k]['cv']['op_n']['neutral_false_positive_rate']),fmt(main[k]['et_test']['macro_f1'])) for k in keys])
        alpha_table = table(['系统/臂','fold0 α','fold1 α','fold2 α','fold3 α','fold4 α','final α'],
            [(k,*[f['selected']['alpha'] for f in v['folds']],v['final_config']['alpha'])
             for k,v in selected.items()])
        report = replace_section(report,7,et_table([k for k in main if k.split('/')[-1] in ('A3','V2','V1')])+"\n\n"+alpha_table+"\n\n每折学习率/权重衰减/epoch全部见 results/selected_configs.json；D0的α=1只是统一配置字段，没有额外语料。")
        report = replace_section(report,8,et_table([k for k in main if k.split('/')[-1] in ('H_ref','H3','E3')])+"\n\n"+table(["配对比较","CV OP-N F1差值","说话人95%区间"],[(k,fmt(v['difference']),str([round(x,4) for x in v['left_minus_right_95']])) for k,v in paired.items()])+"\n\n2000次说话人整群配对重采样，种子20261006。融合、校准和OP-N均复用X03：每个外折拟合只使用其余四折OOF标签，目标集零拟合。D0的V1为h2，与X03的h1参考不同。\n\n去重敏感性只重训D1的A3：音频完整指纹未确认重复，敏感性臂使用9395条MER训练句，保守主臂使用9386条；网格、批序、种子、早停和α不变。敏感性与主臂差值同样报告配对说话人区间，不据此选择数据配方。")
        report = replace_section(report,9,table(["系统/臂","SIMS Pearson","SIMS Spearman","SIMS Acc2","MER6 OP-N F1","MER共享5 OP-N F1"],[(k,fmt(v['sims_valid']['pearson']),fmt(v['sims_valid']['spearman']),fmt(v['sims_valid']['acc_2_nonzero']),fmt(v['mer_test']['op_n']['native6']['macro_f1']),fmt(v['mer_test']['op_n']['shared5']['macro_f1'])) for k,v in main.items()])+"\n\n"+table(["D0或Q系统","worried真值数","预测ET类分布","失败数"],[(k,v['worried_support'],json.dumps(v['worried_to_ET_op_n'],ensure_ascii=False),v['worried_failed']) for k,v in main.items() if 'worried_to_ET_op_n' in v])+"\n\nMER五共享类删除worried真值并在五个共享原生类上归一化；worried去向使用全部七个ET类。D0/Q的第八类为零；MER融合第八类偏置保持零，其余参数从ET学习。未把worried硬映射。MER不保证说话人独立。")
        report = replace_section(report,10,table(["系统/臂","类别","CV OP-N P","R","F1","ET-test OP-N F1"],[(k,c,fmt(v['cv']['op_n']['per_class'][c]['precision']),fmt(v['cv']['op_n']['per_class'][c]['recall']),fmt(v['cv']['op_n']['per_class'][c]['f1']),fmt(v['et_test']['per_class'][c]['f1'])) for k,v in main.items() for c in ('fear','disgust','surprise')]))
        report = replace_section(report,11,table(["仅估算方案","GPU小时含余量和预留"],[(k,fmt(v['total_gpu_hours'])) for k,v in qb['plans'].items()])+"\n\n两折预指定fold0/1；各模型2epoch、8次700句AV验证，AV/A/V=60/20/20。基于X03检查点开启后的实测、MER输入长度和缺脸回退，保守上包络及区间外平方外推；20%余量、0.5小时加载及1小时诊断预留。外推数量/长度分布/缺脸数和局限详见 results/x04b_budget.json。本轮零Q训练，此估算不构成X04b执行授权。")
        missing = table(['Q媒体划分','总句数','无合格脸','比例'],
            [(k,v['n'],v['missing'],f"{v['missing']/v['n']:.4%}") for k,v in q_media['by_split'].items()])
        section11 = report.split('## 11. ',1)[1].split('\n',1)[1].split('\n## 12.',1)[0].strip()
        report = replace_section(report,11,section11+'\n\n'+missing+'\n\nQ的V/AV无合格脸时回退A；MER-test只推理A/AV。媒体沿用X03尺寸、置信度和8帧规则，无学习推理的准备回执见 results/q_media_summary.json。')
        first_started = read(results/'pipeline_original_stop.json')['started_at']
        total_wall = (dt.datetime.fromisoformat(analysis_meta['ended_at'])-dt.datetime.fromisoformat(first_started)).total_seconds()/3600
        cpu_wall = (dt.datetime.fromisoformat(analysis_meta['ended_at'])-dt.datetime.fromisoformat(analysis_meta['started_at'])).total_seconds()/3600
        resources += f"\n\n**最终实耗：累计GPU {progress['charged_gpu_hours']:.6f}小时，上限20小时；所有GPU阶段已完成，无待入账阶段。** 从原编排器启动{first_started}至CPU分析恢复结束{analysis_meta['ended_at']}，总墙钟{total_wall:.4f}小时（包含异常诊断及恢复间隔）。CPU分析恢复{cpu_wall:.4f}小时，代码{analysis_meta['git']['commit']}，未重复训练或推理。上述15.282小时GPU/13.13小时剩余墙钟为12:31UTC首折预算历史快照，不是当前ETA。原report_analysis阶段返回1，独立恢复已成功完成27个系统及8项配对比较；验收测试29通过、2项GPU测试在CPU环境跳过，此前GPU精确取数相等回执仍有效。"
        report = replace_section(report,12,resources)
        handoff = "# X05 输入交接 — X04a证据\n\n本轮结束，等待reviewer。数据配方和X04b均未决定。\n\n"
        handoff += table(["系统/臂","报告指标","冻结参数/概率目录"],[(k,"experiments/X04a_data/results/main_table.json",str(analysis_root/k)) for k in main])
        handoff += f"\n\nCPU目标推理：`python -m emotioncore.x04.targets --arm D1 --encoder A3 --job {args.derived}/heads/D1/A3 --derived {args.derived} --output <H盘目录>`；其他臂/编码器替换相应参数。融合应用参数来自上述 metrics.json，不在新目标标签上再拟合。\n\n汇总分析入口：`python -m emotioncore.x04.analysis --derived {args.derived} --output <H盘目录>`。该命令从冻结OOF重建参数，不运行Q训练。Q只读X03 adapter SHA427ebc14…d5a4。完整输入/概率/头路径和SHA见 results/source_artifact_index.json、selected_configs.json、main_table.json；仓库不含媒体、字幕、特征或逐句概率。CH-SIMS test零模型推理。\n"
        (args.repo/"docs/X05_HANDOFF.md").write_text(handoff)
    (dest/"REPORT.md").write_text(report)
    write_json_atomic(results/"report_run_meta.json",{"git":provenance,"started_at":started,"ended_at":dt.datetime.now(dt.timezone.utc).isoformat(),"complete":args.complete,"report_sha256":sha256(dest/"REPORT.md")})
    (results/"aggregate_index.json").write_text(json.dumps({p.name:{"path":str(p.relative_to(args.repo)),"sha256":sha256(p)} for p in sorted(results.glob('*.json')) if p.name!='aggregate_index.json'},indent=2)+"\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--registration",type=Path,required=True);p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--derived",type=Path,required=True);p.add_argument("--complete",action="store_true")
    p.add_argument("--pipeline-root",type=Path)
    p.add_argument("--analysis-root",type=Path)
    run(p.parse_args())


if __name__=="__main__":main()
