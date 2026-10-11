"""Strict probability loading; method development has a Q_et-only entry point."""
from types import SimpleNamespace
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x04.analysis import source
from emotioncore.x05.common import R3, R4, read, read_jsonl, MANIFEST

COMPONENTS=('Q_A','Q_V','Q_AV','A3','V2','V1')


def assemble(maps, expected, sources):
    ids=sorted(expected)
    if any(set(mapping)!=set(ids) for mapping in maps.values()):
        raise ValueError('STOP component OOF/FMH coverage incomplete')
    records=[expected[i] for i in ids]; columns=[]; bad=[]
    for name in COMPONENTS:
        ordered=[maps[name][uid] for uid in ids]
        for r,e in zip(ordered,records):
            if (r['speaker_id'],r['truth_M'])!=(e['speaker_id'],e['truth_M']):
                raise ValueError('STOP label/speaker disagreement')
            if 'fold' in e and r.get('fold')!=e['fold']:
                raise ValueError('STOP OOF fold disagreement')
        p=np.asarray([r['probabilities'] for r in ordered],dtype=float)
        if p.shape!=(len(ids),7) or not np.isfinite(p).all() or (p<0).any() or (p.sum(1)<=0).any():
            raise ValueError('STOP invalid component probabilities')
        p/=p.sum(1,keepdims=True)
        invalid=np.asarray([r.get('effective_form')=='uniform' or
            r.get('status') in ('failed','missing_media','no_face','face_missing') for r in ordered])
        if name in ('Q_V','V1','V2'):
            missing=set(read('configs/x03_face_missing_policy.json')['face_missing_utt_ids'])
            invalid |= np.asarray([uid in missing for uid in ids])
        p[invalid]=1/7
        columns.append(p); bad.append(invalid)
    return {'ids':ids,'records':records,'p':np.stack(columns,axis=1),
            'bad':np.stack(bad,axis=1),'sources':sources,
            'y':np.asarray([LABELS.index(r['truth_M']) for r in records]),
            'speakers':np.asarray([r['speaker_id'] for r in records])}


def load_q_et_oof():
    """Only reads X03 Q_et plus original D1 OOF; Q_mix cannot be requested."""
    folds=read('configs/folds_v1.json')
    by_speaker={s:f['fold'] for f in folds['folds'] for s in f['outer_holdout_speakers']}
    expected={r['utt_id']:{'utt_id':r['utt_id'],'speaker_id':r['speaker_id'],
              'truth_M':r['labels']['M'],'fold':by_speaker[r['speaker_id']]}
              for r in read_jsonl(MANIFEST) if r['speaker_id'] in by_speaker}
    if len(expected)!=16366 or len(by_speaker)!=15:
        raise ValueError('STOP five-fold OOF coverage incomplete')
    maps={}; sources={}; args=SimpleNamespace(r3=R3,derived=R4)
    for component in COMPONENTS:
        rows=source(args,'D1',component,'oof',sources)
        maps[component]={r['utt_id']:r for r in rows}
        if len(rows)!=len(maps[component]):
            raise ValueError('STOP duplicate component IDs')
    result=assemble(maps,expected,sources)
    result['folds']=np.asarray([r['fold'] for r in result['records']])
    return result


def system(data,components):
    indices=[COMPONENTS.index(c) for c in components]
    return np.log(np.clip(data['p'][:,indices],1e-12,1)),data['bad'][:,indices].all(1)
