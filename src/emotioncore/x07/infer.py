"""One frozen resident service, one pass per clip, no labels or metrics."""
import argparse
import time
import numpy as np
from emotioncore.x07.common import *
from emotioncore.x06.service import Service
from emotioncore.x06.decision import apply
from emotioncore.x05.calibration import base_scores,predictions
from emotioncore.x04.prepare import storage_gate
from emotioncore.eval.metrics import LABELS

def unavailable(reason):
    return {'probabilities':[1/7]*7,'available':False,'accepted_r90':False,'op_n':-1,
            'confidence':0.,'reason':reason,'route':'unavailable'}

def decisions(bundle,components,has_face,scores):
    main=apply(bundle,components,'H3',has_face=has_face,speech=True,scores=scores,alpha=0)
    p=main['calibrated_probabilities']; label=LABELS.index(main['unrejected_label'])
    result={'H3':{'probabilities':p,'available':True,'accepted_r90':main['label']!='reject',
        'op_n':label,'confidence':main['confidence'],'reason':main['reason'],'route':main['route']}}
    for name in ('H_A','H_V'):
        if name=='H_V' and not has_face:
            result[name]=unavailable('no_face');continue
        params=bundle['systems'][name]
        logp=np.log(np.clip(np.asarray([components[c] for c in params['components']])[None],1e-12,1))
        p,label,confidence,_=predictions(base_scores(logp,params),params['neutral_bias'])
        ambiguous=name=='H_V' and len(scores)>1 and scores[1]>=.5*scores[0]
        result[name]={'probabilities':p[0].tolist(),'available':True,
            'accepted_r90':bool(confidence[0]>=params['thresholds']['90'] and not ambiguous),
            'op_n':int(label[0]),'confidence':float(confidence[0]),'route':name,
            'reason':'ambiguous_multi_face' if ambiguous else 'single_modality'}
    return result

def run(split,resume_reason=None):
    if os.environ.get('X07_GPU_GUARDED')!=split:
        raise RuntimeError('Use X07 watchdog')
    protocol,receipt=authority();stage_gate(split)
    directory=ROOT/split
    if (directory/'complete.json').exists() or (directory/'metrics.started.json').exists():
        raise RuntimeError('STOP cannot rerun completed inference or measured split')
    ids=protocol['inventory']['splits'][split]['ids']; prepared=read(directory/'inputs.json')
    if set(prepared['inputs'])!=set(ids) or prepared['protocol_sha256']!=sha(PROTOCOL):
        raise RuntimeError('STOP input coverage/protocol mismatch')
    start=now(); verify_precedes(receipt,start)
    metadata={'started_at':start,'protocol':receipt,'code_commit':protocol['code_commit'],
              'inference_git':git('rev-parse','HEAD'),'labels_read':False,'training':False}
    if (directory/'inference.started.json').exists():
        if not resume_reason:
            raise RuntimeError('STOP interrupted inference requires a recorded infrastructure reason')
        old=read(directory/'inference.started.json')
        if old['protocol']!=receipt:
            raise RuntimeError('STOP resume protocol differs')
        metadata['first_inference_at']=old['started_at']
        write_new(directory/'resumes'/f'{time.time_ns()}.json',{**metadata,'reason':resume_reason})
    else:
        write_new(directory/'inference.started.json',metadata)
        metadata['first_inference_at']=start
    service=Service()
    if service.q.training or any(p.requires_grad for p in service.q.parameters()):
        raise RuntimeError('STOP Q is not frozen for inference')
    service.torch.cuda.reset_peak_memory_stats()
    receipts={}; failures=[]
    for i,uid in enumerate(ids):
        storage_gate(); dest=directory/'predictions'/f'{uid}.json'
        item=read(checked(prepared['inputs'][uid]))
        if dest.exists():
            old=read(dest)
            if old['id']!=uid or old['protocol_sha256']!=sha(PROTOCOL) or old['input']!=prepared['inputs'][uid]:
                raise RuntimeError('STOP resume prediction integrity mismatch')
            receipts[uid]=spec(dest)
            continue
        begin=time.monotonic()
        if item['status']!='ok':
            systems={name:unavailable(item['failure']) for name in SYSTEMS}
            components={};timing={};has_face=False
        else:
            if sha(item['wav'])!=item['wav_sha256']:
                raise RuntimeError('STOP audio input changed')
            # H_V + compare_both requests the union of all three frozen systems.
            # Shared A3/V2/V1 and Q-A/Q-AV/Q-V execute at most once per complete clip.
            components,has_face,timing=service.components(item,'H_V',has_speech=True,compare_both=True)
            systems=decisions(service.bundle,components,has_face,item['face']['track_scores'])
        result={'id':uid,'protocol_sha256':sha(PROTOCOL),'input':prepared['inputs'][uid],
            'systems':systems,'components':components,'has_face':has_face,'timing':timing,
            'completed_at':now(),'elapsed_s':time.monotonic()-begin}
        write_new(dest,result);receipts[uid]=spec(dest)
        atomic(directory/'inference_progress.json',{'done':i+1,'total':len(ids),'at':now(),
            'peak_gpu_bytes':service.torch.cuda.max_memory_allocated()})
    write_new(directory/'complete.json',{'n':len(ids),'ids_sha256':ids_sha(ids),'predictions':receipts,
        'protocol_sha256':sha(PROTOCOL),'first_inference_at':metadata['first_inference_at'],
        'completed_at':now(),'peak_gpu_bytes':service.torch.cuda.max_memory_allocated(),'labels_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['valid','test']);p.add_argument('--resume-reason')
    a=p.parse_args();run(a.split,a.resume_reason)
