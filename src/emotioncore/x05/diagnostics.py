"""Reconcile report-only availability proxies with the frozen face policy."""
import collections
import numpy as np
from emotioncore.x05.common import ROOT,R2,MANIFEST,read,read_jsonl,start,finish,spec
from emotioncore.x05.gate import route
from emotioncore.x05.metrics import selective_curve


def run():
    output=ROOT/'diagnostics';meta=start(output)
    cv=set(read('configs/folds_v1.json')['cv_pool_speakers'])
    missing=set(read('configs/x03_face_missing_policy.json')['face_missing_utt_ids'])
    quality={r['utt_id']:r for r in read_jsonl(R2/'quality/utterance_quality.jsonl')}
    counts=collections.Counter();changes=0
    for row in read_jsonl(MANIFEST):
        if row['speaker_id'] not in cv:continue
        uid=row['utt_id'];r=read(R2/f'face/receipts/{uid}.json')
        score=float(r.get('target_presence_fraction',0))
        detected=r['status']=='ok' and score>0
        available=detected and uid not in missing
        count=2 if r.get('multi_face_frames',0)>0 else int(available)
        routing,reason=route(available,count,score,quality[uid]['audio_duration_s']>0)
        counts[f'{routing}:{reason}']+=1
        changes+=int(detected!=available)
    curves={}
    for name in read(ROOT/'methods/summary.json')['selected_metrics']:
        records=read_jsonl(ROOT/f'methods/{name}_oof.jsonl')
        y=np.asarray([r['truth'] for r in records]);pred=np.asarray([r['label'] for r in records])
        confidence=np.asarray([r['confidence'] for r in records]);p=np.asarray([r['probabilities'] for r in records])
        sorted_p=np.sort(p,axis=1)
        curves[name]={'selective':selective_curve(y,pred,confidence),
                      'margin_control':selective_curve(y,pred,sorted_p[:,-1]-sorted_p[:,-2])}
    finish(output,meta,{'n':sum(counts.values()),'routing_trigger_counts':dict(counts),'curves':curves,
        'face_policy':spec(__import__('pathlib').Path('configs/x03_face_missing_policy.json')),
        'availability_corrections':changes,
        'note':'Report-only corrections: face availability excludes frozen X03 missing IDs; risk curves never accept unavailable predictions. Original method run is preserved. No refitting, method selection, thresholds or frozen JSON changed.',
        'speech_proxy':'audio_duration_s>0; no VAD was run',
        'multi_face_proxy':'any sampled frame has >1 face; conservative utterance-level proxy, not live per-frame rejection rate'})


if __name__=='__main__':run()
