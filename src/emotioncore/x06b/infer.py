"""H3 only, with unchanged frozen service numerics."""
import argparse,os
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face
from emotioncore.x06.service import Service
from emotioncore.x06.decision import apply


def run(kind,cohort):
    task=kind+'_'+cohort
    if os.environ.get('X06B_GPU_GUARDED')!=task:raise RuntimeError('Use X06b GPU watchdog')
    frozen_face();out,meta=start(task)
    source=read(ROOT/f'inputs_{task}/summary.json')['inputs'];items=rows(checked(source))
    allowed=FMH if cohort=='fmh' else TEST
    if {x['speaker_id'] for x in items}!=set(allowed):raise RuntimeError('STOP population assertion')
    service=Service();state={}
    with (out/'predictions.jsonl').open('w') as stream:
        for i,item in enumerate(items):
            speech=not item['visual_only'];p,face,timing=service.components(item,'H3',has_speech=speech)
            result=apply(service.bundle,p,'H3',has_face=face,speech=speech,scores=item['face']['track_scores'],alpha=0,state=state,stream_id=item['speaker_id'])
            # X06a's ungated full-H3 diagnostic uses fixed no-face fallback
            # values. These aliases do not cause another model forward.
            if speech and not face:p.update(Q_AV=p['Q_A'],V2=[1/7]*7)
            row={k:item[k] for k in ['segment_id','utt_id','speaker_id','cohort','start_s','end_s','decision_s','source','visual_only']}
            row.update(component_probs=p,has_face=face,track_scores=item['face']['track_scores'],decision=result,timing=timing)
            stream.write(json.dumps(row)+'\n')
            if (i+1)%25==0:stream.flush();storage_gate();write_json_atomic(out/'progress.json',{'done':i+1,'total':len(items)})
    finish(out,meta,{'status':'completed','n':len(items),'inputs':source,'predictions':spec(out/'predictions.jsonl'),'system':'H3','parameters_refitted':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['r3','r4_correct','r4_vad']);p.add_argument('cohort',choices=['fmh','et_test']);a=p.parse_args();run(a.kind,a.cohort)
