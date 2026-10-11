"""Build the fixed eleven-section report from verified aggregate receipts."""
from pathlib import Path
from emotioncore.x05.common import ROOT,read,spec,start,finish

NAMES=('H_ref','H3','H_A','H_V')


def table(headers,rows):
    def f(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(f(v) for v in r)+' |' for r in rows])


def interval(m,key):
    ci=m['bootstrap']['ci95'][key]
    return f"{m[key]:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def run():
    output=ROOT/'report';meta=start(output)
    methods=read(ROOT/'methods/summary.json');deploy=read(ROOT/'deployment/summary.json')
    freeze=read('configs/x05_method_freeze.json');bundle=read(ROOT/'deployment/calibration_bundle_v1.json')
    diag=read(ROOT/'diagnostics/summary.json');figures=read(ROOT/'figures/summary.json')
    rep=read(ROOT/'reproduce_A3/summary.json')
    heads={c:read(ROOT/f'heads/D1_fmhx/{c}/summary.json') for c in ('A3','V2','V1')}
    gpu={p.parent.name:read(p) for p in (ROOT/'gpu_jobs').glob('*/summary.json')}
    hours=sum(r['charged_gpu_s'] for r in gpu.values())/3600
    mc=methods['selected_metrics'];fmh=deploy['fmh_loso'];et=deploy['targets']['et_test']
    for name,curves in diag['curves'].items():mc[name].update(curves)
    issues=[]
    for n in NAMES:
        for cohort,m in [('FMH LOSO',fmh[n]['all']),('ET-test',et[n])]:
            for definition in ('selected_label_ece','top_label_ece'):
                if m[definition]>.05:issues.append(f'{n} / {cohort} / {definition} = {m[definition]:.4f} > 0.05。')
    a3diff=deploy['expert_comparison_et_test']['A3']['difference']
    if a3diff<-.02:issues.append(f'A3 单专家原始 argmax macro-F1 下降 {a3diff:.4f}，超过 0.02；按修订只报告，不改变决定。')
    text=['# X05：校准、拒识与缺失模态（修订 1）',
      '## 1. 摘要',
      '- 第 0 步已完成；执行修订 1，部署专家使用 D1_fmhx，Q 未训练。',
      f"- A3 复现权重逐值相同：{rep['weights_equal']}；三个专家均排除 FMH 的 06/10/11。",
      f"- 方法冻结提交 `{bundle['method_freeze_commit'][:7]}`；H_ref/H3/H_A/H_V 均采用温度校准。",
      f"- Q_et 五折 H_ref/H3 的所选标签 ECE 为 {mc['H_ref']['selected_label_ece']:.4f}/{mc['H3']['selected_label_ece']:.4f}。",
      f"- Q_mix FMH LOSO H_ref/H3 macro-F1 为 {fmh['H_ref']['all']['macro_f1']:.4f}/{fmh['H3']['all']['macro_f1']:.4f}。",
      f'- GPU 累计 {hours:.3f} / 5 小时；无 Q 训练，CH-SIMS test 零推理。',
      '- 参数包、调用接口及证据已产出；MER-test H_V 因没有既有 Q-V 概率而不可计算。',
      '- 是否验收通过由 reviewer 判断；本报告不作系统采用推荐。',
      '## 2. 第 0 步',
      'PR #5 检查时已为 Ready。按用户授权锁定验收 head `7991aa4d6e06dd99373159be4c4b19053ffb40cc`，于 **2026-10-11 11:09:14（Asia/Shanghai）** 合入 main，合并提交 `69eb5263724cf32b290d107d960e91249cf12b7b`。从快进后的干净 main 建立 `x05-calib`。X05 使用 PR #6，保持未合入。',
      '## 3. 完成情况与偏离',
      '初始预检证实原 D1 fold 0 重拟合已使用 FMH 全部 3,266 句，按原停止规则中止；历史证据见 `STOP_REPORT_20261011.md` 和 `results/expert_fmh_gate.json`。',
      '**按 X05_REVISION_1.md 恢复。reviewer 在本修订前未看到任何 X05 结果**，此前仅产生门禁证据。修订新增 A3 复现及三个 D1_fmhx 头的固定配置训练，不选参、不使用验证集；GPU 仍共用 5 小时上限。`emotioncore.x04` 源码未修改，新入口直接调用 `emotioncore.x04.heads.train`。',
      'C1–C6 使用 Q_et + 原 D1 五折 OOF；C7 的 Q_mix 两折 OOF 配原 D1 留出专家预测，FMH 与部署目标配 D1_fmhx。不能用训练过 OOF 说话人的 D1_fmhx 预测冒充专家 OOF。这个混合来源继承原协议，模型迁移误差单独报告。',
      '边界与缺项：未增加任何非 FMH 的 Q 推理。既有 MER-test 仅有 Q-A/Q-AV，故 H_V 的 MER-test 结果缺失；其余目标行完整。人脸 presence 的四个经验五分位边界均为 1.0，按预登记并列处理保留空组。两项均在第 11 节列出。',
      '## 4. FMH 推理与 D1_fmhx 回执',
      f"A3 复现：weights_equal={rep['weights_equal']}；新旧 checkpoint SHA 均为 `{rep['checkpoint']['sha256']}`，无需以 ET-test 一致率兜底。复现固定 {rep['training']['epochs_executed']} epoch，验证次数为 0。",
      table(['专家','ET / MER 句数','固定 epoch','FMH 训练交集','Checkpoint SHA-256'],[
         [c,f"{h['inputs']['ET_count']} / {h['inputs']['MER_count']}",h['training']['epochs_executed'],
          h['inputs']['fmh_overlap'],h['checkpoint']['sha256']] for c,h in heads.items()]),
      '三个头训练 ET 说话人均与两个最终 Q 的 12 名 ET 训练说话人及逐句 ID 一致。MER 使用 X04a D1 的去重后 9,386 条记录（不替换为 Q 的 MER 子集），沿用各自 final_config 和种子 20260928。训练前排除 FMH、ET-test、MER-test、CH-SIMS；逐句回执仅在服务器。',
      table(['Q 版本','A','V','AV'],[[v,*[read(ROOT/f'Q/{v}_fmh/summary.json')['forms'][f]['n'] for f in ('A','V','AV')]] for v in ('et','mix')]),
      '两个最终 Q 均完成全部 3,266 句的三模态推理，原缺脸 V 均匀分布 / AV 回退 A 规则不变。没有 OOM 跳句。FMH 全部用于部署拟合；另报排除 700 句 checkpoint 选择集后的 2,566 句。',
      '## 5. 方法选择（C1–C3）与 x05_method_freeze.json',
      f"冻结文件 SHA-256：`{bundle['method_freeze_sha256']}`；提交 `{bundle['method_freeze_commit']}`（2026-10-11 11:32:49 Asia/Shanghai），早于第二段全部 Q_mix 分析。第二段启动校验文件 SHA 及已提交内容，配置之后未修改。",
      '每个系统独立应用预登记选择规则：五折 OOF 留一折，融合及校准仅拟合另外四折；汇总逐句 NLL 最低者，差距 ≤ 0.002 时选择更简单者。融合继续沿用 X03 的非负对数概率权重与中心化偏置，再单独拟合校准。因融合阶段已有偏置，融合系统温度与温度加偏置候选的 NLL 可近乎一致。',
      table(['系统','温度 NLL','温度+偏置 NLL','逐类温度+偏置 NLL','冻结方法'],[
          [n,*[freeze['candidate_nll'][n][m] for m in freeze['candidate_order']],freeze['selected_methods'][n]] for n in freeze['selected_methods']]),
      '标签是 OP-N argmax；置信度是**施加 OP-N 中性偏置之前**的校准概率中该标签的一项。OP-R90/80 用拟合集的经验阈值，接受 confidence ≥ threshold，并列值一并接受。第一/第二概率差仅作对照；不按目标集结果重新选择。',
      '## 6. 校准与选择性预测结果（Q_et 五折；Q_mix FMH）',
      '区间为说话人整群 bootstrap 2,000 次，种子 20261011（NumPy default_rng）。FMH 仅三名说话人，区间为描述性。拒识保留在召回分母；另列被接受样本上的 F1。完整逐类 P/R/F1、ECE、Brier、AURC、曲线和区间见 results JSON。',
      '风险曲线不接受强制不可用样本，横轴分母仍为完整队列；AURC 为曲线面积除以最大可达覆盖率。Q_et 的此项报告计算修正单列 `results/diagnostics_summary.json`，原方法运行原样保留；没有重拟合、重选方法或更改冻结文件。',
      'Q_et 五折留一折：',
      table(['系统','macro-F1 [95% CI]','UAR','WA','NLL','Brier','OP-A ECE','所选标签 ECE','R90 覆盖率','R80 覆盖率'],[
          [n,interval(mc[n],'macro_f1'),mc[n]['uar'],mc[n]['wa'],mc[n]['nll'],mc[n]['brier'],
           mc[n]['top_label_ece'],mc[n]['selected_label_ece'],mc[n]['rejection']['90']['coverage'],mc[n]['rejection']['80']['coverage']] for n in NAMES]),
      'Q_mix + D1_fmhx，FMH 内按说话人留一（每次用另外两名 FMH 加两折 OOF 拟合融合/校准，仅另外两名 FMH 拟合工作点）：',
      table(['系统','口径','macro-F1 [95% CI]','NLL','所选标签 ECE','中性误报','R90 覆盖率','R90 接受 F1','R90 拒识入分母 F1'],[
          [n,k,interval(fmh[n][k],'macro_f1'),fmh[n][k]['nll'],fmh[n][k]['selected_label_ece'],fmh[n][k]['neutral_false_positive_rate'],
           fmh[n][k]['rejection']['90']['coverage'],fmh[n][k]['rejection']['90']['accepted_macro_f1'],
           fmh[n][k]['rejection']['90']['reject_in_recall_macro_f1']] for n in NAMES for k in ('all','without_700')]),
      '只用两折 OOF 拟合全部参数的 FMH 对照（描述迁移变化，不重新选择）：',
      table(['系统','两折基线 F1','FMH LOSO F1','两折基线中性误报','FMH LOSO 中性误报','两折基线 R90 覆盖率'],[
          [n,deploy['twofold_only_baseline'][n]['all']['macro_f1'],fmh[n]['all']['macro_f1'],
           deploy['twofold_only_baseline'][n]['all']['neutral_false_positive_rate'],fmh[n]['all']['neutral_false_positive_rate'],
           deploy['twofold_only_baseline'][n]['all']['rejection']['90']['coverage']] for n in NAMES]),
      '![Q_et reliability](figures/q_et_oof_reliability.png)',
      '![Q_mix FMH reliability](figures/q_mix_fmh_reliability.png)',
      '![Risk coverage](figures/q_et_oof_risk_coverage.png)',
      '![F1 coverage](figures/q_et_oof_f1_coverage.png)',
      '![Neutral sensitivity](figures/neutral_sensitivity.png)',
      '## 7. 缺失模态与守门函数',
      table(['回退系统','独立拟合 F1','对照 1 F1','对照 2 F1','较好对照减 0.01'],[
          ['H_A',mc['H_A']['macro_f1'],mc['Q_A']['macro_f1'],mc['A3']['macro_f1'],max(mc['Q_A']['macro_f1'],mc['A3']['macro_f1'])-.01],
          ['H_V',mc['H_V']['macro_f1'],mc['Q_V']['macro_f1'],mc['V2']['macro_f1'],max(mc['Q_V']['macro_f1'],mc['V2']['macro_f1'])-.01]]),
      table(['沿用全系统参数的均匀填充控制','macro-F1'],[[n,r['macro_f1']] for n,r in methods['missing_full_parameters'].items()]),
      '部署 H_A/H_V 使用各自独立拟合的全部参数。控制 (i) 将依赖缺失模态的整个分量填均匀分布。人脸 presence 五分位结果：',
      table(['五分位组','n','H_ref F1','H3 F1','H_A F1'],[[r['quintile'],r['n'],
          *[r.get('systems',{}).get(n,'空组') for n in ('H_ref','H3','H_A')]] for r in methods['quality_and_gate']['quintiles']]),
      '四个切点均为 1.0，无法得到五个有差异且均衡的质量层；没有据此增加质量回退规则。',
      table(['守门输出与原因','触发句数','比例'],[[k,v,v/diag['n']] for k,v in diag['routing_trigger_counts'].items()]),
      f"守门规则固定：多脸拒绝（multi_face），两种模态都无拒绝，无脸走 H_A，无语音走 H_V，其余走全系统；质量值不影响路由。触发率使用句级任一帧多脸和音频时长代理，不能当作在线逐帧误拒率或 VAD 准确率。报告阶段补充 X03 缺脸策略匹配，修正 {diag['availability_corrections']} 条可用性代理；未修改冻结方法。上述性能表为固定系统/路由的评估，守门触发率单列。",
      '## 8. 迁移误差',
      '部署参数全量拟合：两折 OOF 6 人 + FMH 3 人，共 9,799 句拟合融合/校准；仅 FMH 全部 3,266 句拟合中性偏置和拒识阈值。ET-test/MER-test/SIMS-valid 零拟合、零选择，参数包原样应用。',
      table(['系统','ET-test F1','NLL','OP-A ECE','所选标签 ECE','中性误报','偏离 0.15','R90 覆盖率','R80 覆盖率'],[
          [n,et[n]['macro_f1'],et[n]['nll'],et[n]['top_label_ece'],et[n]['selected_label_ece'],et[n]['neutral_false_positive_rate'],
           et[n]['neutral_deviation_from_015'],et[n]['rejection']['90']['coverage'],et[n]['rejection']['80']['coverage']] for n in NAMES]),
      'ET-test 逐说话人和被接受样本 macro-F1 全部列于 `results/deployment_summary.json`。',
      table(['系统','MER 共享类 n','NLL','所选标签 ECE','中性误报','R90 覆盖率','接受 macro-F1'],[
          [n,*[deploy['targets']['mer_test'][n][k] for k in ('shared_n','nll','selected_label_ece','neutral_false_positive_rate')],
           deploy['targets']['mer_test'][n]['rejection']['90']['coverage'],deploy['targets']['mer_test'][n]['rejection']['90']['accepted_macro_f1']] for n in ('H_ref','H3','H_A')]),
      'MER 排除 worried 真值，仅对 neutral/joy/sadness/anger/surprise 五类重新归一化。无可靠说话人标识，不编造说话人 bootstrap。H_V 缺既有 Q-V 概率，不计算。',
      table(['系统','SIMS 零标签误判非中性','Pearson 可用媒体','Pearson 全部647','R90 覆盖率','R90 接受 Pearson'],[
          [n,*[deploy['targets']['sims_valid'][n][k] for k in ('neutral_non_neutral_rate','pearson_all_available','pearson_all_with_missing_uniform_scores')],
           deploy['targets']['sims_valid'][n]['rejection']['90']['coverage'],deploy['targets']['sims_valid'][n]['rejection']['90']['pearson_accepted']] for n in NAMES]),
      'SIMS 的 27 条原有缺媒体记录保留在覆盖率分母；媒体可用 Pearson 单独列出。Pearson 使用校准概率 joy 减 sadness/anger/fear/disgust 概率和。',
      '修订 1 的只报告专家对照（ET-test，单专家原始 argmax）：',
      table(['专家','原 D1 macro-F1','D1_fmhx macro-F1','差'],[[c,deploy['expert_comparison_et_test'][c]['D1']['raw_argmax_macro_f1'],
          deploy['expert_comparison_et_test'][c]['D1_fmhx']['raw_argmax_macro_f1'],deploy['expert_comparison_et_test'][c]['difference']] for c in ('A3','V2','V1')]),
      f"H_ref(Q_mix)，两组专家共用同一冻结部署参数：原 D1 {deploy['expert_comparison_et_test']['H_ref']['D1']:.4f}，D1_fmhx {deploy['expert_comparison_et_test']['H_ref']['D1_fmhx']:.4f}。",
      '## 9.',
      '## 10. 耗时、资源与异常',
      table(['GPU 工作','保守计费小时','状态'],[[k,v['charged_gpu_s']/3600,v['status']] for k,v in gpu.items()]),
      f'GPU 合计 **{hours:.3f} / 5 小时**，计费包含子进程加载与准备时间；V2/V1 训练及所有专家目标推理在 CPU。根分区启动时约 112.97 GB 可用，守护进程持续检查 ≥ 80 GB。所有新服务器产物位于 `/mnt/h/emotionAI/emotioncore_runs/X05`，旧目录只读。',
      '每次正式运行从独立干净 Git checkout 启动并记录 run_meta；原始逐句概率与训练回执仅在服务器，仓库保存汇总、路径、SHA。GPU 子进程超时、异常或断言失败会中止后续 GPU 作业；本轮没有跳过 OOM 句。初期本机 compileall 因系统字节码缓存目录权限失败，随后改用不写缓存的 AST 检查及服务器 pytest。',
      '测试命令和结果见 `results/final_tests.json`；图像来源和 SHA 见 `results/figures_summary.json`。图由 Matplotlib 对固定聚合值绘制。',
      '## 11. 给 reviewer 的问题',
      '\n'.join(f'- {s}' for s in issues) if issues else '- FMH LOSO 与 ET-test 的所有部署系统 ECE 均未超过 0.05；A3 原始 argmax 下降未超过 0.02。',
      '- 请按原门槛检查完整表格；三名 FMH 说话人的区间仅作描述。FMH 未参与最终模型梯度训练，但 700 句曾用于 Q checkpoint 选择，原 D1 配置亦继承先前选参；剔除 700 句口径不等于全流程完全盲测。',
      '- MER-test H_V 的 Q-V 概率不存在；受本轮 FMH-only Q 推理范围限制，该行不可计算。',
      '- 既有人脸 presence 几乎全为 1，五分位层退化；没有选择新的质量阈值。是否在后续使用更有区分度的质量指标/回退规则由 reviewer 决定。',
      '- 两折 OOF 使用原 D1 留出专家；部署/FMH 使用 D1_fmhx。该模型来源差异已完整列出，未以部署头的训练内预测冒充 OOF。',
      '- 本 PR 不合入；等待 reviewer 验收。']
    (output/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    finish(output,meta,{'report':spec(output/'REPORT.md'),'gpu_hours':hours,'reviewer_flags':issues,
                        'bundle':spec(ROOT/'deployment/calibration_bundle_v1.json')})


if __name__=='__main__':run()
