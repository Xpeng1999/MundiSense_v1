"""C7/C8, inaccessible until the committed method freeze and G completion."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
from scipy.stats import pearsonr

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x04.analysis import source
from emotioncore.x04.prepare import write_rows
from emotioncore.x05.common import (ROOT,R3,R4,R4B,FMH,MANIFEST,read,read_jsonl,sha256,spec,start,finish)
from emotioncore.x05.data import assemble,system,COMPONENTS,load_q_et_oof
from emotioncore.x05.calibration import (SYSTEMS,fit_base,fit_operating,base_scores,predictions,
                                       fit_held_speaker)
from emotioncore.x05.metrics import measure,selective_curve,classification,confusion
from emotioncore.x05.bundle import verify_method_freeze

DEPLOY_SYSTEMS=('H_ref','H3','H_A','H_V')


def checked(path,expected,sources):
    if sha256(path)!=expected:raise ValueError(f'STOP source SHA mismatch: {path}')
    sources[str(path)]={'path':str(path),'sha256':expected}
    rows=read_jsonl(path)
    if len(rows)!=len({r['utt_id'] for r in rows}):raise ValueError('STOP repeated prediction IDs')
    return rows


def completion_gate():
    for version in ('et','mix'):
        root=ROOT/f'Q/{version}_fmh'
        summary=read(root/'summary.json')
        if set(summary['forms'])!={'A','V','AV'} or any(r['n']!=3266 for r in summary['forms'].values()):
            raise ValueError('STOP incomplete FMH Q inference')
    for component in ('A3','V2','V1'):
        summary=read(ROOT/f'heads/D1_fmhx/{component}/summary.json')
        if not summary['passed'] or summary['inputs']['fmh_overlap']:
            raise ValueError('STOP D1_fmhx training leakage')
    gpu=[read(p) for p in (ROOT/'gpu_jobs').glob('*/summary.json')]
    if len(gpu)!=4 or any(r['status']!='completed' for r in gpu) or sum(r['charged_gpu_s'] for r in gpu)>18000:
        raise ValueError('STOP GPU jobs incomplete or budget exceeded')


def load_cohort(cohort,version='mix'):
    sources={};maps={}
    frozen=read('configs/folds_v1.json')
    outer={s:f['fold'] for f in frozen['folds'][:2] for s in f['outer_holdout_speakers']}
    expected={r['utt_id']:{'utt_id':r['utt_id'],'speaker_id':r['speaker_id'],'truth_M':r['labels']['M'],
              'fold':outer.get(r['speaker_id']) if cohort=='oof' else None}
              for r in read_jsonl(MANIFEST) if r['speaker_id'] in (outer if cohort=='oof' else FMH)}
    if len(expected)!=(6533 if cohort=='oof' else 3266):
        raise ValueError('STOP calibration cohort coverage incomplete')
    for component in COMPONENTS:
        if component.startswith('Q_'):
            form=component[2:];rows=[]
            if cohort=='oof':
                root=R4B/'Q' if version=='mix' else R3/'Q'
                for fold in range(2):
                    p=root/f'predict_fold{fold}/{form}.jsonl';summary=read(p.parent/'summary.json')
                    rows.extend(checked(p,summary[form]['sha256'],sources))
            else:
                root=ROOT/f'Q/{version}_fmh';summary=read(root/'summary.json')
                rows=checked(root/f'{form}.jsonl',summary['forms'][form]['sha256'],sources)
        elif cohort=='oof':
            rows=source(SimpleNamespace(r3=R3,derived=R4),'D1',component,'oof',sources)
            rows=[r for r in rows if r['utt_id'] in expected]
        else:
            root=ROOT/f'heads/D1_fmhx/{component}/targets';summary=read(root/'summary.json')
            rows=checked(root/'fmh.jsonl',summary['artifacts']['fmh']['sha256'],sources)
        maps[component]={r['utt_id']:r for r in rows}
    return assemble(maps,expected,sources)


def describe(data,values,subset):
    ix=np.flatnonzero(subset)
    result=measure(data['y'][ix],values['p'][ix],values['pred'][ix],values['confidence'][ix],
        data['speakers'][ix],{k:np.broadcast_to(v,len(data['y']))[ix] for k,v in values['thresholds'].items()})
    result['selective']=selective_curve(data['y'][ix],values['pred'][ix],values['confidence'][ix])
    return result


def save_predictions(output,name,data,values):
    path=output/f'{name}.jsonl'
    write_rows(path,[{'utt_id':uid,'speaker_id':data['speakers'][i],'truth':int(data['y'][i]),
        'probabilities':values['p'][i].tolist(),'label':int(values['pred'][i]),
        'confidence':float(values['confidence'][i]),
        'thresholds':{k:float(np.broadcast_to(v,len(data['y']))[i]) for k,v in values['thresholds'].items()}}
        for i,uid in enumerate(data['ids'])])
    return spec(path)


def apply_values(logp,bad,parameters):
    p,pred,confidence,margin=predictions(base_scores(logp,parameters),parameters['neutral_bias'])
    pred[bad]=-1
    return dict(p=p,pred=pred,confidence=confidence,margin=margin,thresholds=parameters['thresholds'])


def target_maps(split,sources,original_experts=False):
    maps={}
    for component in COMPONENTS:
        if component.startswith('Q_'):
            form=component[2:]
            if split=='mer_test' and form=='V':continue
            directory=R4B/'Q'/('predict_mer_test' if split=='mer_test' else f'predict_final_{split}')
            summary=read(directory/'summary.json')
            rows=checked(directory/f'{form}.jsonl',summary[form]['sha256'],sources)
        elif original_experts:
            rows=source(SimpleNamespace(r3=R3,derived=R4),'D1',component,split,sources)
        else:
            directory=ROOT/f'heads/D1_fmhx/{component}/targets';summary=read(directory/'summary.json')
            rows=checked(directory/f'{split}.jsonl',summary['artifacts'][split]['sha256'],sources)
        maps[component]={r['utt_id']:r for r in rows}
    ids=sorted(maps['Q_A'])
    if len(ids)!={'et_test':1929,'mer_test':572,'sims_valid':647}[split] or any(set(v)!=set(ids) for v in maps.values()):
        raise ValueError('STOP target probability coverage differs')
    key='label' if split=='sims_valid' else 'truth_M'
    for component,mapping in maps.items():
        if any(mapping[uid][key]!=maps['Q_A'][uid][key] for uid in ids):
            raise ValueError('STOP target component labels disagree')
    return ids,maps


def target_inputs(ids,maps,components):
    columns=[];bad=[]
    for component in components:
        ordered=[maps[component][uid] for uid in ids]
        p=np.asarray([r['probabilities'] for r in ordered],dtype=float)
        if p.shape!=(len(ids),7) or not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():
            raise ValueError('STOP invalid target probabilities')
        p/=p.sum(1,keepdims=True)
        invalid=np.asarray([r.get('effective_form')=='uniform' or
            r.get('status') in ('failed','missing_media','no_face','face_missing') for r in ordered])
        if component in ('Q_V','V1','V2'):
            missing=set(read('configs/x03_face_missing_policy.json')['face_missing_utt_ids'])
            invalid |= np.asarray([uid in missing for uid in ids])
        p[invalid]=1/7;columns.append(p);bad.append(invalid)
    return np.log(np.clip(np.stack(columns,axis=1),1e-12,1)),np.logical_and.reduce(bad)


def sims_report(p,pred,confidence,thresholds,labels,bad):
    s=p[:,1]-p[:,2:6].sum(1)
    def corr(keep):
        return float(pearsonr(labels[keep],s[keep]).statistic) if keep.sum()>2 and np.std(s[keep])>0 else None
    neutral=labels==0
    result={'n':len(labels),'media_available_n':int((~bad).sum()),
        'neutral_n':int(neutral.sum()),
        'neutral_non_neutral_rate':float(np.mean((pred[neutral]>0)&(~bad[neutral]))) if neutral.any() else None,
        'pearson_all_available':corr(~bad),'pearson_all_with_missing_uniform_scores':corr(np.ones(len(labels),bool)),
        'pearson_score':'calibrated pre-OP-N p(joy)-p(sadness,anger,fear,disgust)',
        'rejection':{}}
    for key,threshold in thresholds.items():
        keep=(confidence>=threshold)&(~bad)
        result['rejection'][key]={'coverage':float(keep.mean()),'accepted_n':int(keep.sum()),'pearson_accepted':corr(keep)}
    return result


def target_reports(output,bundle,sources):
    results={};comparison={}
    for split in ('et_test','mer_test','sims_valid'):
        ids,maps=target_maps(split,sources);records=[maps['Q_A'][uid] for uid in ids];results[split]={}
        for name,params in bundle['systems'].items():
            if split=='mer_test' and name=='H_V':
                results[split][name]={'status':'unavailable','reason':'No existing Q-V MER-test predictions; additional Q inference is outside the authorized FMH-only GPU scope'}
                continue
            logp,bad=target_inputs(ids,maps,params['components'])
            values=apply_values(logp,bad,params);p=values['p'];pred=values['pred'];confidence=values['confidence']
            if split=='sims_valid':
                result=sims_report(p,pred,confidence,params['thresholds'],np.asarray([r['label'] for r in records]),bad)
            elif split=='mer_test':
                shared=[0,1,2,3,6];keep=np.asarray([r['truth_M']!='worried' for r in records])
                pp=p[keep][:,shared];pp/=pp.sum(1,keepdims=True)
                ss=np.log(np.clip(pp,1e-12,1));ss[:,0]+=params['neutral_bias']
                guess=ss.argmax(1);guess[bad[keep]]=-1
                conf=pp[np.arange(len(pp)),np.maximum(guess,0)]
                yy=np.asarray([shared.index(LABELS.index(r['truth_M'])) for r,k in zip(records,keep) if k])
                result=measure(yy,pp,guess,conf,np.asarray(['MER']*len(yy)),params['thresholds'],
                    bootstrap=False,label_names=[LABELS[c] for c in shared])
                result.update(total_n=len(records),shared_n=len(yy),removed_worried_n=int((~keep).sum()),
                    bootstrap_note='Reliable speaker IDs unavailable; no invented speaker bootstrap')
            else:
                yy=np.asarray([LABELS.index(r['truth_M']) for r in records]);speakers=np.asarray([r['speaker_id'] for r in records])
                result=measure(yy,p,pred,confidence,speakers,params['thresholds'])
                result['per_speaker']={s:measure(yy[speakers==s],p[speakers==s],pred[speakers==s],
                    confidence[speakers==s],speakers[speakers==s],params['thresholds'],bootstrap=False) for s in sorted(set(speakers))}
            if split!='sims_valid':result['neutral_deviation_from_015']=result['neutral_false_positive_rate']-.15
            results[split][name]=result
            path=output/f'{name}_{split}.jsonl'
            write_rows(path,[{'utt_id':uid,'probabilities':pp.tolist(),'label':int(g),'confidence':float(c),
                              'media_available':not bool(b)} for uid,pp,g,c,b in zip(ids,p,pred,confidence,bad)])
            result['predictions']=spec(path)
        if split=='et_test':
            _,original=target_maps(split,sources,original_experts=True)
            yy=np.asarray([LABELS.index(r['truth_M']) for r in records])
            for component in ('A3','V2','V1'):
                item={}
                for arm,mm in [('D1',original),('D1_fmhx',maps)]:
                    logp,bad=target_inputs(ids,mm,(component,));guess=logp[:,0].argmax(1);guess[bad]=-1
                    item[arm]={'raw_argmax_macro_f1':float(classification(confusion(yy,guess))['macro_f1'])}
                item['difference']=item['D1_fmhx']['raw_argmax_macro_f1']-item['D1']['raw_argmax_macro_f1']
                comparison[component]=item
            params=bundle['systems']['H_ref'];lp,bad=target_inputs(ids,original,params['components'])
            old=apply_values(lp,bad,params)
            comparison['H_ref']={'D1':float(classification(confusion(yy,old['pred']))['macro_f1']),
                'D1_fmhx':results[split]['H_ref']['macro_f1'],
                'definition':'same frozen deployment calibration/OP-N applied to both expert sources; Q_mix fixed'}
    return results,comparison


def run(freeze_sha):
    freeze_path=Path('configs/x05_method_freeze.json')
    freeze=verify_method_freeze(freeze_path,freeze_sha,require_committed=True)
    completion_gate()
    output=ROOT/'deployment';meta=start(output)
    meta.update(method_freeze=spec(freeze_path),freeze_commit=subprocess.check_output(
        ['git','log','-1','--format=%H','--',str(freeze_path)],text=True).strip())
    write_json_atomic(output/'run_meta.json',meta)
    oof=load_cohort('oof');fmh=load_cohort('fmh');sources={**oof['sources'],**fmh['sources']}
    selected_ids=set(read('configs/x03_inner_val_subsets.json')['folds'][0]['utt_ids'])
    if len(selected_ids)!=700 or not selected_ids.issubset(fmh['ids']):raise ValueError('STOP FMH selection subset mismatch')
    masks={'all':np.ones(3266,bool),'without_700':np.asarray([uid not in selected_ids for uid in fmh['ids']])}
    report={};parameters={};baseline={};prediction_index={}
    for name in DEPLOY_SYSTEMS:
        components=SYSTEMS[name];method=freeze['selected_methods'][name]
        ox,_=system(oof,components);fx,bad=system(fmh,components)
        n=len(fx);values={'p':np.zeros((n,7)),'pred':np.zeros(n,dtype=int),'confidence':np.zeros(n),
                        'thresholds':{key:np.zeros(n) for key in ('90','80')}};fold_params=[]
        for speaker in sorted(FMH):
            held,scores,params,fit=fit_held_speaker(ox,oof['y'],fx,lambda ix:fmh['y'][ix],fmh['speakers'],speaker,method)
            p,g,c,_=predictions(scores,params['neutral_bias']);g[bad[held]]=-1
            values['p'][held]=p;values['pred'][held]=g;values['confidence'][held]=c
            for key in ('90','80'):values['thresholds'][key][held]=params['thresholds'][key]
            fold_params.append({'held_speaker':speaker,'fmh_fit_speakers':sorted(set(fmh['speakers'][fit])),
                'fmh_fit_n':len(fit),'oof_fit_n':len(ox),**params})
        report[name]={key:describe(fmh,values,mask) for key,mask in masks.items()}
        report[name]['loso_parameters']=fold_params
        report[name]['per_speaker']={s:describe(fmh,values,fmh['speakers']==s) for s in sorted(FMH)}
        prediction_index[name]=save_predictions(output,name+'_fmh_loso',fmh,values)
        old=fit_base(ox,oof['y'],method);old.update(fit_operating(base_scores(ox,old),oof['y']))
        old_values=apply_values(fx,bad,old)
        baseline[name]={key:describe(fmh,old_values,mask) for key,mask in masks.items()}
        baseline[name]['parameters']=old
        save_predictions(output,name+'_fmh_twofold_baseline',fmh,old_values)
        final=fit_base(np.concatenate((ox,fx)),np.concatenate((oof['y'],fmh['y'])),method)
        final.update(fit_operating(base_scores(fx,final),fmh['y']))
        parameters[name]={'components':list(components),**final}
    fit_path=output/'fitting_ids.jsonl'
    write_rows(fit_path,[{'utt_id':uid,'speaker_id':data['speakers'][i],'cohort':cohort}
        for cohort,data in [('twofold_oof',oof),('fmh',fmh)] for i,uid in enumerate(data['ids'])])
    components={c:read(ROOT/f'heads/D1_fmhx/{c}/summary.json')['checkpoint'] for c in ('A3','V2','V1')}
    qmeta=read(ROOT/'Q/mix_fmh/run_meta.json')
    for c in ('Q_A','Q_V','Q_AV'):components[c]={'path':qmeta['adapter'],'sha256':qmeta['adapter_sha256'],
        'base_index':qmeta['base_index']}
    bundle={'version':1,'labels':list(LABELS),'systems':parameters,'components':components,
        'method_freeze_sha256':freeze_sha,'method_freeze_commit':meta['freeze_commit'],
        'fitting_set':spec(fit_path),'fit_counts':{'twofold_oof':6533,'fmh':3266,'speakers':9},
        'fusion_calibration_fit':'twofold OOF plus all FMH',
        'operating_fit':'all FMH only','experts':'D1_fmhx',
        'gate_rules':{'multiple_faces':'reject:multi_face','neither_modality':'reject:no_modalities',
                      'no_face':'H_A','no_speech':'H_V','both':'requested full system',
                      'face_quality_threshold':None},
        'default_full_system':'H_ref','default_rejection_point':'90'}
    write_json_atomic(output/'calibration_bundle_v1.json',bundle)
    # Report Q_et final-model transfer without using it for method selection.
    et_fmh=load_cohort('fmh','et');et_oof=load_q_et_oof();qet_reference={}
    sources.update(et_fmh['sources']);sources.update(et_oof['sources'])
    for name in DEPLOY_SYSTEMS:
        ox,_=system(et_oof,SYSTEMS[name]);fx,bad=system(et_fmh,SYSTEMS[name])
        params=fit_base(ox,et_oof['y'],freeze['selected_methods'][name])
        params.update(fit_operating(base_scores(ox,params),et_oof['y']))
        values=apply_values(fx,bad,params)
        qet_reference[name]={key:describe(et_fmh,values,mask) for key,mask in masks.items()}
    targets,comparison=target_reports(output,bundle,sources)
    write_json_atomic(output/'sources.json',sources)
    finish(output,meta,{'fmh_loso':report,'twofold_only_baseline':baseline,'q_et_fmh_reference':qet_reference,
        'targets':targets,'expert_comparison_et_test':comparison,'prediction_index':prediction_index,
        'bundle':spec(output/'calibration_bundle_v1.json'),'sources':spec(output/'sources.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze-sha256',required=True)
    run(p.parse_args().freeze_sha256)
