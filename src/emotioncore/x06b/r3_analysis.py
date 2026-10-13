"""Paired new-versus-X06a FMH gate; ET-test only reports."""
import argparse,numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x06b.common import *
from emotioncore.x06.s6b import frozen_decisions,macro_f1


def original_face_flags(predictions,inputs):
    """Recover the original Service flag from its hash-checked input manifest."""
    old={r['utt_id']:r for r in predictions};source={r['utt_id']:r for r in inputs}
    if len(old)!=len(predictions) or len(source)!=len(inputs) or set(old)!=set(source):
        raise RuntimeError('STOP face input identity mismatch')
    flags={}
    for uid,record in old.items():
        item=source[uid]
        if 'q_video' not in item or (item['q_video'] is not None and 'status' not in item['q_video']):
            raise RuntimeError('STOP face input metadata missing')
        face=item['q_video'] is not None and item['q_video']['status']=='ok'
        if 'has_face' in record and (type(record['has_face']) is not bool or record['has_face']!=face):
            raise RuntimeError('STOP face flag disagrees with original input')
        flags[uid]=face
    return flags


def paired(y,a,b,seed=20261013,iterations=2000):
    rng=np.random.default_rng(seed);draws={k:[] for k in ['op_n','r90','coverage']}
    for _ in range(iterations):
        ix=rng.integers(0,len(y),len(y))
        for k in ['op_n','r90']:draws[k].append(macro_f1(y[ix],b[k][ix])-macro_f1(y[ix],a[k][ix]))
        draws['coverage'].append(float(np.mean(b['r90'][ix]>=0)-np.mean(a['r90'][ix]>=0)))
    result={}
    for k in ['op_n','r90']:
        before,after=macro_f1(y,a[k]),macro_f1(y,b[k])
        result[k]={'old_macro_f1':before,'new_macro_f1':after,'delta':after-before,'ci95':np.percentile(draws[k],[2.5,97.5]).tolist()}
    before,after=float(np.mean(a['r90']>=0)),float(np.mean(b['r90']>=0))
    result['coverage']={'old':before,'new':after,'delta':after-before,'ci95':np.percentile(draws['coverage'],[2.5,97.5]).tolist()}
    return result


def run(cohort):
    no_stop();out,meta=start('r3_analysis_'+cohort)
    path=OLD/('fmh_stream_oracle' if cohort=='fmh' else 'revision2/et_remaining')/'summary.json'
    old_summary=read(path);old_source=old_summary['predictions'];new_source=read(ROOT/f'r3_{cohort}/summary.json')['predictions']
    old_records=rows(checked(old_source));old_inputs=old_summary['inputs']
    old_faces=original_face_flags(old_records,rows(checked(old_inputs)))
    old={r['utt_id']:r for r in old_records};new={r['utt_id']:r for r in rows(checked(new_source))}
    truth={e['utt_id']:e['label'] for s in (FMH if cohort=='fmh' else TEST) for e in source_timeline(s)['entries'] if e['kind']=='utterance'}
    if set(old)!=set(new) or set(new)!=set(truth):raise RuntimeError('STOP pairing identity mismatch')
    ids=sorted(truth);y=np.array([LABELS.index(truth[i]) for i in ids]);params=read('configs/calibration_bundle_v1.1.json')['systems']['H3']
    pa={c:np.array([old[i]['component_probs'][c] for i in ids]) for c in params['components']}
    pb={c:np.array([new[i]['component_probs'][c] for i in ids]) for c in params['components']}
    a,b=frozen_decisions(pa,params),frozen_decisions(pb,params);report=paired(y,a,b)
    status='STOP_r3_point' if cohort=='fmh' and report['op_n']['delta']<-.05 else 'completed'
    pairs=[{'utt_id':uid,'truth':int(y[j]),'old_op_n':int(a['op_n'][j]),'new_op_n':int(b['op_n'][j]),'old_r90':int(a['r90'][j]),'new_r90':int(b['r90'][j])} for j,uid in enumerate(ids)]
    (out/'paired.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    finish(out,meta,{'status':status,'cohort':cohort,'n':len(ids),**report,
        'component_argmax_agreement':{c:float(np.mean(pa[c].argmax(1)==pb[c].argmax(1))) for c in ['Q_AV','V2']},
        'old_no_face_n':sum(not face for face in old_faces.values()),'new_no_face_n':sum(not r['has_face'] for r in new.values()),
        'old_face_inputs':old_inputs,'old_face_rule':"q_video is not None and q_video.status == 'ok' (original Service)",
        'bootstrap':{'seed':20261013,'iterations':2000,'unit':'paired utterance','ci':'95% percentile'},
        'ci_acceptance':report['op_n']['ci95'][0]>=-.02 if cohort=='fmh' else None,
        'modality_and_multiface_gates':False,'paired':spec(out/'paired.jsonl'),'old':old_source,'new':new_source,'gpu_seconds':0})
    if status!='completed':stop('R3',status,spec(out/'summary.json'));raise RuntimeError(status)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['fmh','et_test']);run(p.parse_args().cohort)
