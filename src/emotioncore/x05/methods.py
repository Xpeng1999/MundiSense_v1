"""C1-C6: leave-one-fold-out method development using Q_et and original D1."""
import collections
import hashlib
import numpy as np
from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x04.prepare import write_rows
from emotioncore.x05.common import ROOT,R2,read,read_jsonl,spec,start,finish
from emotioncore.x05.data import load_q_et_oof,system,COMPONENTS
from emotioncore.x05.calibration import (SYSTEMS,METHODS,fit_held_fold,base_scores,
    predictions,select_method,nll)
from emotioncore.x05.metrics import measure,selective_curve,classification,confusion
from emotioncore.x05.gate import route


def evaluate(data,logp,invalid,method):
    n=len(data['y']); probs=np.zeros((n,7)); pred=np.zeros(n,dtype=int)
    confidence=np.zeros(n); margin=np.zeros(n)
    thresholds={k:np.zeros(n) for k in ('90','80')}; parameters=[]
    curve_pred={str(t):np.zeros(n,dtype=int) for t in (.05,.10,.20,.25)}
    for fold in range(5):
        held,scores,params,fitting=fit_held_fold(logp,lambda ix:data['y'][ix],data['folds'],fold,method)
        p,labels,c,m=predictions(scores,params['neutral_bias'])
        labels[invalid[held]]=-1
        probs[held]=p;pred[held]=labels;confidence[held]=c;margin[held]=m
        for key in thresholds: thresholds[key][held]=params['thresholds'][key]
        for key,bias in params['neutral_curve_biases'].items():
            _,labels,_,_=predictions(scores,bias);labels[invalid[held]]=-1;curve_pred[key][held]=labels
        parameters.append({'held_fold':fold,'fit_folds':sorted(set(data['folds'][fitting].tolist())),
            'fit_count':len(fitting),'held_count':len(held),
            'fit_ids_sha256':hashlib.sha256('\n'.join(data['ids'][i] for i in fitting).encode()).hexdigest(),
            **params})
    report=measure(data['y'],probs,pred,confidence,data['speakers'],thresholds)
    report['selective']=selective_curve(data['y'],pred,confidence)
    report['margin_control']=selective_curve(data['y'],pred,margin)
    report['neutral_curve']={k:{m:float(v) for m,v in classification(confusion(data['y'],pp)).items()
        if m in ('macro_f1','nonneutral_macro_recall','neutral_false_positive_rate')} for k,pp in curve_pred.items()}
    return report,parameters,dict(p=probs,pred=pred,confidence=confidence,margin=margin,thresholds=thresholds)


def quality_report(data,outputs):
    quality_path=R2/'quality/utterance_quality.jsonl'
    q={r['utt_id']:r for r in read_jsonl(quality_path)}
    receipts=[]; hashes=hashlib.sha256()
    for uid in data['ids']:
        if uid not in q:raise ValueError('STOP incomplete utterance quality coverage')
        p=R2/f'face/receipts/{uid}.json';r=read(p);receipts.append(r)
        hashes.update((uid+':'+spec(p)['sha256']+'\n').encode())
    # Presence is the recorded X02 face-quality scalar; keep ties together.
    score=np.asarray([float(r.get('target_presence_fraction',0)) for r in receipts])
    cuts=np.quantile(score,[.2,.4,.6,.8]); group=np.searchsorted(cuts,score,side='right')
    result={'quality_scalar':'X02 target_presence_fraction (missing = 0)',
        'cutpoints':cuts.tolist(),'tie_policy':'ties enter the higher quintile; empty bins reported',
        'utterance_quality':spec(quality_path),'face_receipts_sha256':hashes.hexdigest(),'quintiles':[]}
    for g in range(5):
        ix=np.flatnonzero(group==g); item={'quintile':g+1,'n':len(ix)}
        if len(ix):
            item['truncation_suspect_rate']=float(np.mean([q[data['ids'][i]]['truncation_suspect'] for i in ix]))
            item['systems']={name:float(classification(confusion(data['y'][ix],o['pred'][ix]))['macro_f1'])
                              for name,o in outputs.items() if name in ('H_ref','H3','H_A')}
        result['quintiles'].append(item)
    triggers=collections.Counter()
    for uid,r,s in zip(data['ids'],receipts,score):
        has_face=r['status']=='ok' and s>0
        # Available utterance statistics: any multi-face frame is a conservative proxy.
        count=2 if r.get('multi_face_frames',0)>0 else int(has_face)
        routing,reason=route(has_face,count,s,q[uid]['audio_duration_s']>0)
        triggers[f'{routing}:{reason}']+=1
    result['routing_trigger_counts']=dict(triggers)
    result['routing_proxy']='utterance any multi_face_frames>0; speech proxy is audio_duration_s>0, not VAD'
    return result


def run():
    output=ROOT/'methods';meta=start(output);data=load_q_et_oof()
    write_json_atomic(output/'sources.json',data['sources'])
    scores={}; selected={}; candidates={}; predictions_by_system={}; parameters_by_system={}
    for name,components in SYSTEMS.items():
        logp,invalid=system(data,components);candidates[name]={};values={}
        for method in METHODS:
            report,params,values[method]=evaluate(data,logp,invalid,method)
            candidates[name][method]=report
            write_json_atomic(output/f'{name}_{method}.json',{'metrics':report,'parameters':params})
            values[method]['parameters']=params
            write_json_atomic(output/'progress.json',{'system':name,'method':method,'completed':True})
        scores[name]={method:candidates[name][method]['nll'] for method in METHODS}
        selected[name]=select_method(scores[name]);chosen=values[selected[name]]
        predictions_by_system[name]=chosen;parameters_by_system[name]=chosen['parameters']
        write_rows(output/f'{name}_oof.jsonl',[{'utt_id':uid,'speaker_id':data['speakers'][i],
            'fold':int(data['folds'][i]),'truth':int(data['y'][i]),'probabilities':chosen['p'][i].tolist(),
            'label':int(chosen['pred'][i]),'confidence':float(chosen['confidence'][i]),
            'thresholds':{k:float(v[i]) for k,v in chosen['thresholds'].items()}} for i,uid in enumerate(data['ids'])])
    missing={}
    for full in ('H_ref','H3'):
        components=SYSTEMS[full]
        for fallback,keep in [('H_A',('Q_A','A3')),('H_V',('V1','V2'))]:
            p=data['p'][:,[COMPONENTS.index(c) for c in components]].copy()
            for j,c in enumerate(components):
                if c not in keep:p[:,j]=1/7
            logp=np.log(np.clip(p,1e-12,1));pred=np.zeros(len(p),dtype=int)
            prob=np.zeros((len(p),7));conf=np.zeros(len(p))
            for fold,params in enumerate(parameters_by_system[full]):
                ix=np.flatnonzero(data['folds']==fold)
                prob[ix],pred[ix],conf[ix],_=predictions(base_scores(logp[ix],params),params['neutral_bias'])
            missing[f'{full}->{fallback}']=measure(data['y'],prob,pred,conf,data['speakers'])
    quality=quality_report(data,predictions_by_system)
    selected_metrics={n:candidates[n][selected[n]] for n in SYSTEMS}
    freeze={'version':1,'selection_population':'Q_et plus original D1, all five ET OOF folds only',
        'selection':'per-system pooled leave-one-fold-out NLL; within 0.002 choose simpler a,b,c',
        'candidate_order':list(METHODS),'selected_methods':selected,'candidate_nll':scores,
        'confidence':'pre-OP-N calibrated probability of OP-N argmax label',
        'neutral_target':.15,'rejection_coverages':[.9,.8],
        'rejection_ties':'accept >= empirical threshold, all ties included',
        'ece_bins':15,'bootstrap':{'unit':'speaker_id','replicates':2000,'seed':20261011},
        'fusion':'unchanged X03 nonnegative log-probability fusion with centered biases, then separate calibration',
        'fallback':'independent full parameter fit for H_A and H_V',
        'missing_control':'full-system parameters; every component requiring the absent modality is uniform',
        'quality':{'scalar':'target_presence_fraction','quintiles':'empirical cutpoints, ties kept together'},
        'source_index':spec(output/'sources.json'),'code_commit':meta['git']['commit']}
    write_json_atomic(output/'x05_method_freeze.json',freeze)
    finish(output,meta,{'selected_metrics':selected_metrics,'missing_full_parameters':missing,
        'quality_and_gate':quality,'freeze':spec(output/'x05_method_freeze.json'),
        'oof_predictions':{n:spec(output/f'{n}_oof.jsonl') for n in SYSTEMS}})


if __name__=='__main__':run()
