"""Paired two-fold crossfit for both Q sources; no target-label fitting."""
from __future__ import annotations
import argparse,datetime as dt,json,random,sys
from pathlib import Path
import numpy as np
from emotioncore.eval.metrics import LABELS,classification,sims_metrics
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03 import fusion,q_calibrate
from emotioncore.x03.h_fusion import q_rows
from emotioncore.x03.q_manifest import read_jsonl,sha256
from emotioncore.x04.analysis import et_metrics,scored,extend_mer,native_metrics,paired_speakers
from emotioncore.x04.prepare import require_h,storage_gate,write_rows
SEED=20261007
SYSTEMS={'Q_AV':('Q_AV',),'Q_A':('Q_A',),'H_ref':('Q_A','Q_AV','A3','V2','V1'),'H3':('Q_AV','A3','V2')}

def fit_system(logp,y,invalid,single):
    if single:
        c,b=q_calibrate.fit_single(logp[:,0],y,invalid)
        return {'calibration':c.tolist(),'neutral_bias':b}
    theta=fusion.fit_fusion(logp,y);logits=fusion.fusion_logits(logp,theta)
    c=fusion.fit_calibration(logits,y);b=fusion.fit_neutral_bias(fusion.calibrated_logits(logits,c),y)
    return {'fusion':theta.tolist(),'calibration':c.tolist(),'neutral_bias':b}

def apply_system(logp,invalid,params,single):
    if single:return q_calibrate.apply_single(logp[:,0],invalid,np.array(params['calibration']),params['neutral_bias'])
    logits=fusion.calibrated_logits(fusion.fusion_logits(logp,np.array(params['fusion'])),np.array(params['calibration']))
    a=fusion.softmax(logits);logits[:,0]+=params['neutral_bias'];n=fusion.softmax(logits)
    a[invalid]=n[invalid]=1/7
    return a,n

def twofold(logp,y,folds,invalid,single):
    if set(folds.tolist())!={0,1}:raise ValueError('Expected only preregistered folds0/1')
    a=np.empty((len(y),7));n=np.empty_like(a);params=[]
    for held in (0,1):
        fit=folds!=held;mask=~fit
        p=fit_system(logp[fit],y[fit],invalid[fit],single)
        a[mask],n[mask]=apply_system(logp[mask],invalid[mask],p,single)
        params.append({'held_fold':held,'fit_fold':1-held,'fit_count':int(fit.sum()),'held_count':int(mask.sum()),**p})
    return a,n,params

def read_checked(spec,artifacts):
    path=Path(spec['path'])
    if sha256(path)!=spec['sha256']:raise ValueError('Frozen source SHA changed')
    artifacts[str(path)]=spec;return read_jsonl(path)

def source(args,version,component,split,artifacts):
    if component.startswith('Q_'):
        form=component[2:]
        if version=='Q_et':
            if split=='mer_test':
                spec=json.loads((args.r4/'Q_predictions/artifacts.json').read_text())[form]
                return read_checked(spec,artifacts)
            rows,specs=q_rows(args.r3,form,split)
            artifacts.update({s['path']:s for s in specs});return rows
        if split=='oof':
            rows=[]
            for fold in (0,1):
                root=args.root/f'predict_fold{fold}';spec=json.loads((root/'summary.json').read_text())[form];rows+=read_checked(spec,artifacts)
            return rows
        root=args.root/('predict_mer_test' if split=='mer_test' else 'predict_final_'+split)
        spec=json.loads((root/'summary.json').read_text())[form]
        return read_checked(spec,artifacts)
    job=args.r4/f'heads/D1/{component}'
    if split=='oof':
        info=json.loads((job/'results.json').read_text());return read_checked({'path':str(job/'oof.jsonl'),'sha256':info['oof_sha256']},artifacts)
    spec=json.loads((job/'targets/artifacts.json').read_text())[split];return read_checked(spec,artifacts)

def stack(args,version,components,split,artifacts):
    maps=[]
    for comp in components:
        rows=source(args,version,comp,split,artifacts)
        if split=='oof':rows=[r for r in rows if r['fold'] in (0,1)]
        mapping={r['utt_id']:r for r in rows}
        if len(mapping)!=len(rows):raise ValueError('Duplicate source ID')
        maps.append(mapping)
    ids=sorted(maps[0]);n={'oof':6533,'et_test':1929,'sims_valid':647,'mer_test':572}[split]
    if len(ids)!=n or any(set(m)!=set(ids) for m in maps):raise ValueError('STOP OOF/target coverage')
    records=[maps[0][uid] for uid in ids];raw=[];invalid=[]
    missing=set(json.loads(Path('configs/x03_face_missing_policy.json').read_text())['face_missing_utt_ids'])
    for comp,mapping in zip(components,maps):
        rows=[mapping[uid] for uid in ids]
        keys=['label'] if split=='sims_valid' else ['truth_M']
        if split=='oof':keys+=['speaker_id','fold']
        if any(any(a[k]!=b[k] for k in keys) for a,b in zip(records,rows)):raise ValueError('Source truth/fold/speaker mismatch')
        p=np.array([r.get('probabilities8',r['probabilities']+[0.]) for r in rows],dtype=float)
        bad=np.array([r.get('effective_form')=='uniform' or r.get('status') in ('failed','missing_media','no_face','face_missing') for r in rows])
        if comp.startswith('V'):
            bad|=np.array([uid in missing for uid in ids]);p[bad]=1/8
        if not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():raise ValueError('Invalid probability')
        p/=p.sum(1,keepdims=True);raw.append(p);invalid.append(bad)
    bad=np.logical_and.reduce(invalid)
    if any(c.startswith('Q_') for c in components):bad|=np.array([r.get('status')=='missing_media' for r in records])
    return records,np.stack(raw,axis=1),bad

def paired_pearson(left,right):
    if [r['utt_id'] for r in left]!=[r['utt_id'] for r in right]:raise ValueError('SIMS paired ID mismatch')
    def values(rows):return np.array([np.nan if r['forced_error'] else r['op_n'][1]-sum(r['op_n'][2:6]) for r in rows])
    x=values(left);z=values(right);y=np.array([r['label'] for r in left]);valid=np.isfinite(x)&np.isfinite(z)&np.isfinite(y)
    x,z,y=x[valid],z[valid],y[valid];rng=np.random.default_rng(SEED);delta=[]
    if len(x)<3:raise ValueError('Insufficient paired SIMS outputs')
    for _ in range(2000):
        ix=rng.integers(len(x),size=len(x));delta.append(float(np.corrcoef(x[ix],y[ix])[0,1]-np.corrcoef(z[ix],y[ix])[0,1]))
    if not np.isfinite(delta).all():raise ValueError('Undefined SIMS bootstrap correlation')
    return {'n':len(x),'excluded_joint_failures':int((~valid).sum()),'unit':'utterance','seed':SEED,'replicates':2000,'difference':float(np.corrcoef(x,y)[0,1]-np.corrcoef(z,y)[0,1]),'difference_95':np.percentile(delta,[2.5,97.5]).tolist(),'bootstrap_differences':delta}

def run(args):
    git=git_provenance(Path.cwd());storage_gate();require_h(args.output);args.output.mkdir(parents=True,exist_ok=True)
    meta={'git':git,'command':sys.argv,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'interim':args.interim,'target_labels_used_for_fitting':False,'expert_recipe':'frozen X04a D1','folds':[0,1],'seed':SEED}
    write_json_atomic(args.output/'run_meta.json',meta);table={};artifacts={};paired={};sims={}
    et=read_jsonl(Path('experiments/X01_benchmark/results/et_utterances.jsonl'));foldspec=json.loads(Path('configs/folds_v1.json').read_text());hold={s:f['fold'] for f in foldspec['folds'][:2] for s in f['outer_holdout_speakers']};expected={r['utt_id']:r for r in et if r['speaker_id'] in hold}
    if len(expected)!=6533 or len(hold)!=6:raise ValueError('Frozen twofold cohort differs')
    for version in ('Q_et','Q_mix'):
        for name,components in SYSTEMS.items():
            records,raw,bad=stack(args,version,components,'oof',artifacts)
            if {r['utt_id'] for r in records}!=set(expected) or any(r['speaker_id']!=expected[r['utt_id']]['speaker_id'] or r['truth_M']!=expected[r['utt_id']]['labels']['M'] or r['fold']!=hold[r['speaker_id']] for r in records):raise ValueError('OOF label/speaker/fold leakage')
            y=np.array([LABELS.index(r['truth_M']) for r in records]);folds=np.array([r['fold'] for r in records]);single=len(components)==1
            p=raw[...,:7]/raw[...,:7].sum(2,keepdims=True);lp=np.log(np.clip(p,1e-12,1));a,n,params=twofold(lp,y,folds,bad,single);final=fit_system(lp,y,bad,single)
            key=version+'/'+name;output=args.output/key;output.mkdir(parents=True,exist_ok=True)
            item={'version':version,'system':name,'components':components,'cv':{'op_a':et_metrics(a,records,bad),'op_n':et_metrics(n,records,bad)},'parameters_oof':params,'parameters_final':final,'artifacts':{}}
            paired[key]=scored(n,records,bad)
            def save(split,rows,pa,pn,invalid):
                out=[{**r,'op_a':aa.tolist(),'op_n':nn.tolist(),'forced_error':bool(b)} for r,aa,nn,b in zip(rows,pa,pn,invalid)]
                path=output/(split+'.jsonl');write_rows(path,out);item['artifacts'][split]={'path':str(path),'sha256':sha256(path)};return out
            save('oof',records,a,n,bad)
            if not args.interim:
                for split in ('et_test','sims_valid','mer_test'):
                    rows,target,invalid=stack(args,version,components,split,artifacts)
                    if split=='mer_test':
                        pa,pn=extend_mer(target,final,single=single,unsupported_worried=single and version=='Q_et')
                        item[split]={point:{'native6':native_metrics(prob,rows,invalid),'shared5':native_metrics(prob,rows,invalid,shared=True)} for point,prob in [('op_a',pa),('op_n',pn)]}
                    else:
                        p=target[...,:7]/target[...,:7].sum(2,keepdims=True);pa,pn=apply_system(np.log(np.clip(p,1e-12,1)),invalid,final,single)
                        if split=='et_test':
                            item[split]=et_metrics(pn,rows,invalid);item[split]['per_speaker']={s:classification([r for r in scored(pn,rows,invalid) if r['speaker_id']==s]) for s in sorted({r['speaker_id'] for r in rows})}
                        else:item[split]=sims_metrics([{'id':r['utt_id'],'label':r['label'],'s':None if b else float(prob[1]-prob[2:6].sum())} for r,prob,b in zip(rows,pn,invalid)])
                    out=save(split,rows,pa,pn,invalid)
                    if split=='sims_valid':sims[key]=out
            write_json_atomic(output/'metrics.json',item);table[key]=item
    primary={};secondary={}
    for name in SYSTEMS:
        result=paired_speakers(paired['Q_mix/'+name],paired['Q_et/'+name],seed=SEED)
        (primary if name=='H_ref' else secondary)[name]=result
    futility=secondary['Q_AV']['difference']<-.02
    write_json_atomic(args.output/'futility.json',{'metric':'twofold crossfit CV OP-N macro-F1 Q_AV mix-et','point_difference':secondary['Q_AV']['difference'],'threshold':-.02,'stop':futility,'no_target_labels':True})
    if not args.interim:
        primary['P2']={name:paired_pearson(sims['Q_mix/'+name],sims['Q_et/'+name]) for name in ('H_ref','Q_AV')}
    for filename,value in [('main_table.json',table),('primary.json',primary),('secondary.json',secondary),('source_artifact_index.json',artifacts)]:write_json_atomic(args.output/filename,value)
    meta['ended_at']=dt.datetime.now(dt.timezone.utc).isoformat();meta['outputs']={p.name:sha256(p) for p in args.output.glob('*.json') if p.name!='run_meta.json'};write_json_atomic(args.output/'run_meta.json',meta)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--interim',action='store_true');p.add_argument('--r3',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X03/Q'));p.add_argument('--r4',type=Path,default=Path('/mnt/h/emotionAI/emotioncore_runs/X04a'));run(p.parse_args())
if __name__=='__main__':main()
