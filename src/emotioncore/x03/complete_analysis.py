"""Complete X03 comparison intervals, robustness and resource accounting.

No fitting or selection changes: all predictions and selected systems are frozen.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
import numpy as np
from emotioncore.eval.metrics import LABELS, classification
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.final_analysis import prediction_rows, score
from emotioncore.x03.q_manifest import read_jsonl, sha256


def statistics(confusion, bins):
    support=confusion.sum(-1)
    tp=np.diagonal(confusion[...,:7],axis1=-2,axis2=-1)
    predicted=confusion[...,:7].sum(-2)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=precision+recall>0)
    total=support.sum(-1)
    scalar={'macro_f1':f1.mean(-1),'uar':recall.mean(-1),'wa':tp.sum(-1)/total,
            'neutral_false_positive_rate':np.divide(confusion[...,0,1:7].sum(-1),support[...,0],out=np.zeros_like(total),where=support[...,0]>0),
            'nonneutral_macro_recall':recall[...,1:].mean(-1),
            'ece15':np.abs(bins[...,1]-bins[...,2]).sum(-1)/total}
    return scalar,{'precision':precision,'recall':recall,'f1':f1}


def speaker_intervals(rows, repetitions=2000, seed=20260929):
    speakers=sorted({r['speaker_id'] for r in rows})
    index={s:i for i,s in enumerate(speakers)}
    confusion=np.zeros((len(speakers),7,8))
    bins=np.zeros((len(speakers),15,3))
    for row in rows:
        g=index[row['speaker_id']];truth=LABELS.index(row['truth'])
        pred=LABELS.index(row['pred']) if row['pred'] in LABELS else 7
        confusion[g,truth,pred]+=1
        confidence=max(row['probabilities']);b=min(int(confidence*15),14)
        bins[g,b]+=[1,int(truth==pred),confidence]
    full,_=statistics(confusion.sum(0),bins.sum(0))
    expected=score(rows)
    if any(not np.isclose(full[k],expected[k],atol=1e-12) for k in full):
        raise ValueError('Bootstrap sufficient statistics disagree with frozen metrics')
    rng=random.Random(seed)
    weights=np.zeros((repetitions,len(speakers)))
    for rep in range(repetitions):
        for s in rng.choices(speakers,k=len(speakers)):
            weights[rep,index[s]]+=1
    scalars,classes=statistics(np.einsum('rg,gij->rij',weights,confusion),np.einsum('rg,gij->rij',weights,bins))
    def ci(values):
        lo,hi=np.percentile(values,[2.5,97.5],axis=0)
        return {'lower_95':float(lo),'upper_95':float(hi)}
    return {'unit':'speaker_id','seed':seed,'replicates':repetitions,'speaker_count':len(speakers),
            'metrics':{k:ci(v) for k,v in scalars.items()},
            'per_class':{label:{k:ci(v[:,i]) for k,v in classes.items()} for i,label in enumerate(LABELS)}}


def resource_counts(components):
    encoders=[];heads=[]
    if any(c.startswith('V') or c=='Q_AV' for c in components):encoders.append('YuNet')
    if any(c.startswith('Q_') for c in components):encoders.append('Qwen2.5-Omni-7B+final_adapter (shared across A/AV)')
    for n in ('A3','A2','V2','V1'):
        if any(c.startswith(n+'_') for c in components):encoders.append(n);heads.append(next(c for c in components if c.startswith(n+'_')))
    if 'T1_r' in components:encoders+=['SenseVoiceSmall','T1 fine-tuned classifier']
    return {'encoder_asr_detector_models':encoders,'classifier_heads':heads,
            'primary_model_count':len(encoders),'classifier_head_count':len(heads),
            'total_neural_artifact_count':len(encoders)+len(heads),
            'definition':'distinct neural weight artifacts needed over the serial pipeline; Q A/AV share weights; fusion/calibration arrays are not neural models'}


def run(args):
    git=git_provenance(Path.cwd())
    r2,r3=args.x02_root,args.x03_root
    table=json.loads((r3/'analysis/main_table.json').read_text())
    latency=json.loads((r3/'latency/summary.json').read_text())
    resources=json.loads((r3/'resources/summary.json').read_text())
    q_resources=json.loads((r3/'resources_q/summary.json').read_text())
    manifest=read_jsonl(args.manifest)
    cv_ids={r['utt_id'] for r in manifest if r['official_split'] in ('train','val') and r['speaker_id'] in json.loads(Path('configs/folds_v1.json').read_text())['cv_pool_speakers']}
    test_ids={r['utt_id'] for r in manifest if r['official_split']=='test'}
    sources={};intervals={}
    for name,bundle in table.items():
        if name=='old_lora_ET_only':continue
        root=Path(bundle['selected_source']);metric=json.loads((root/'metrics.json').read_text())
        components=metric.get('components',[name])
        if name=='V2_reference':components=['V2_h2']
        bundle['latency']=latency['systems'][name]
        count=resource_counts(components)
        gpu_stages=[s for s in ('Q','A3','A2','V2','V1','SenseVoice') if (s=='Q' and any(c.startswith('Q_') for c in components)) or (s=='SenseVoice' and 'T1_r' in components) or any(c.startswith(s+'_') for c in components)]
        count['peak_cuda_allocated_bytes']=max(resources['stages'][s]['peak_cuda_allocated_bytes'] for s in gpu_stages)
        if name=='Q_A':
            count['peak_cuda_allocated_bytes']=q_resources['conditions']['A']['peak_cuda_allocated_bytes']
            count['memory_method']='longest 20 frozen ET-test A inputs, same final adapter and default CUDA allocator'
        count['memory_method']=resources['method'];count['memory_limitation']=resources['limitation']
        count['q_condition_probes']={f:q_resources['conditions'][f]['peak_cuda_allocated_bytes'] for f in ('A','V','AV')} if 'Q' in gpu_stages else None
        bundle['resources']=count
        intervals[name]={}
        for split,ids,points in [('cv',cv_ids,('op_a','op_n')),('et_test',test_ids,('op_n',))]:
            for point in points:
                rows=prediction_rows(root/('oof.jsonl' if split=='cv' else 'et_test.jsonl'),ids,point,force_face_error=name=='V2_reference')
                value=speaker_intervals(rows)
                intervals[name][split+'_'+point]=value
                bundle[split+'_'+point]['complete_speaker_bootstrap_95']=value
        sources[name]={'root':str(root),'metrics_sha256':sha256(root/'metrics.json'),
                       'probabilities':{s:{'path':str(root/(s+'.jsonl')),'sha256':sha256(root/(s+'.jsonl'))} for s in ('oof','et_test','sims_valid')}}
    robustness={}
    for arm,root,forms in [('E2',r3/'E2/eval',('AVT','AV','A','V')),('Q',r3/'Q_calibrated',('AV','A','V'))]:
        robustness[arm]={}
        for form in forms:
            metric=json.loads((root/(('Q_'+form) if arm=='Q' else form)/'metrics.json').read_text())
            robustness[arm][form]={'cv':metric['cv'],'et_test':metric['et_test'],'sims_valid':metric['sims_valid']}
    visual={'purpose':'descriptive comparison; different X02/X03 calibration and training conditions, not a causal estimate',
            'Q_V_X03':robustness['Q']['V']}
    for n in ('V3a','V3b'):
        p=r2/f'heads/{n}_h1_cv/results.json'
        visual[n+'_h1_X02']={'source':str(p),'sha256':sha256(p),'results':json.loads(p.read_text())}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for filename,value in [('main_table_complete.json',table),('complete_speaker_intervals.json',intervals),
                           ('modality_robustness.json',robustness),('visual_correction.json',visual),('final_artifact_index.json',sources)]:
        write_json_atomic(args.output_dir/filename,value)
    write_json_atomic(args.output_dir/'run_meta.json',{'git':git,'command':sys.argv,
                      'ended_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'source_main_sha256':sha256(r3/'analysis/main_table.json'),
                      'latency_sha256':sha256(r3/'latency/summary.json'),
                      'resources_sha256':sha256(r3/'resources/summary.json'),
                      'q_resources_sha256':sha256(r3/'resources_q/summary.json'),
                      'outputs':{p.name:sha256(p) for p in args.output_dir.glob('*.json') if p.name!='run_meta.json'}})


def main():
    p=argparse.ArgumentParser()
    for flag in ('manifest','x02-root','x03-root','output-dir'):p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
