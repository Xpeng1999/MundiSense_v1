"""One-shot post-completion metrics; test labels are first read here."""
import argparse
import numpy as np
from scipy.stats import rankdata
from emotioncore.x07.common import *

def sentiment(p):
    p=np.asarray(p,dtype=float)
    return p[...,1]-p[...,2:6].sum(axis=-1)

def correlation(x,y):
    if len(x)<2:
        return None
    a=x-x.mean(); b=y-y.mean(); denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denom) if denom>0 else None

def point(y,s,available,accepted,pred):
    nz=y!=0; zero=y==0
    ok=available & nz & (s!=0) & ((s>0)==(y>0))
    return {
        'pearson_available':correlation(s[available],y[available]),
        'spearman_available':correlation(rankdata(s[available]),rankdata(y[available])),
        'acc2':float(ok.sum()/nz.sum()) if nz.any() else None,
        'pearson_all_uniform':correlation(s,y),
        'op_r90_coverage':float(accepted.mean()),
        'pearson_accepted':correlation(s[accepted],y[accepted]),
        'neutral_non_neutral_rate':float(((pred[zero]>0)&available[zero]).mean()) if zero.any() else None,
    }

def compute(rows,labels,repetitions=2000,seed=20261014):
    ids=[r['id'] for r in rows]; y=np.array([labels[i] for i in ids],float)
    if not np.isfinite(y).all() or (np.abs(y)>1).any():
        raise ValueError('STOP invalid SIMS labels')
    groups=np.asarray([i.split('__')[0] for i in ids]); keys=sorted(set(groups))
    indices=[np.flatnonzero(groups==k) for k in keys]
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(keys),size=(repetitions,len(keys)))
    result={}
    for system in SYSTEMS:
        values=[r['systems'][system] for r in rows]
        p=np.asarray([v['probabilities'] for v in values],float)
        if p.shape!=(len(rows),7) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('STOP invalid calibrated probabilities')
        available=np.asarray([v['available'] for v in values],bool)
        accepted=np.asarray([v['accepted_r90'] for v in values],bool)&available
        pred=np.asarray([v['op_n'] for v in values],int)
        s=sentiment(p);s[~available]=-3/7
        estimates=point(y,s,available,accepted,pred)
        series={k:[] for k in estimates}
        for draw in draws:
            take=np.concatenate([indices[g] for g in draw])
            sample=point(y[take],s[take],available[take],accepted[take],pred[take])
            for k,v in sample.items():
                if v is not None and np.isfinite(v):
                    series[k].append(v)
        result[system]={**estimates,'n':len(rows),'media_available_n':int(available.sum()),
            'accepted_n':int(accepted.sum()),'nonzero_n':int((y!=0).sum()),'neutral_n':int((y==0).sum()),
            'video_count':len(keys),'ci95':{k:np.percentile(v,[2.5,97.5]).tolist() if v else None for k,v in series.items()},
            'bootstrap_valid_replicates':{k:len(v) for k,v in series.items()}}
    return result

def run(split):
    protocol,receipt=authority()
    stage_gate(split)
    directory=ROOT/split
    ids=protocol['inventory']['splits'][split]['ids']
    done,rows=load_completed(directory,ids)
    if done['protocol_sha256']!=sha(PROTOCOL):
        raise RuntimeError('STOP inference protocol mismatch')
    verify_precedes(receipt,done['first_inference_at'])
    claim_metrics(directory)
    # This is deliberately below the completion AND exclusive-once guards.
    import pandas as pd
    source=protocol['inventory']['meta_csv']; checked(source)
    df=pd.read_csv(source['path'],usecols=['video_id','clip_id','mode','label'],dtype={'video_id':str,'clip_id':str})
    selected=df[df['mode']==split]
    labels={f'{r.video_id}__{r.clip_id}':float(r.label) for r in selected.itertuples()}
    if set(labels)!=set(ids):
        raise RuntimeError('STOP metric population mismatch')
    begin=now()
    systems=compute(rows,labels,protocol['bootstrap']['repetitions'],protocol['bootstrap']['seed'])
    result={'split':split,'protocol_sha256':sha(PROTOCOL),'protocol_commit':receipt['commit'],
        'inference_completion':spec(directory/'complete.json'),'labels_source':source,
        'started_at':begin,'ended_at':now(),'systems':systems,'bootstrap':protocol['bootstrap']}
    if split=='valid':
        value=systems['H3']['pearson_available']
        result['L3_pass']=value is not None and value>=.78
    write_new(directory/'metrics.json',result)
    if split=='valid' and not result['L3_pass']:
        stop('L3','H3 media-available Pearson < 0.78',spec(directory/'metrics.json'))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['valid','test']);run(p.parse_args().split)
