"""Incremental selected-scale inputs, with all raw-frame lifetimes bounded."""
import argparse,math,time,hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06b.face import Tracker
from emotioncore.x06b.video import Video
from emotioncore.x06b.detect import pipeline
from emotioncore.x06.prepare_segments import q_media,frame_index


def frozen_face():
    import subprocess
    no_stop();authority();path=Path('configs/x06b_face_freeze.json')
    expected=checked(read(ROOT/'r2/summary.json')['freeze'])
    if path.read_bytes()!=expected.read_bytes():raise RuntimeError('STOP face freeze differs')
    blob=subprocess.check_output(['git','show','HEAD:'+str(path)])
    if blob!=path.read_bytes():raise RuntimeError('STOP face freeze not committed')
    return read(path)


def make_item(event,face,directory,signal,identity):
    import soundfile as sf
    duration=event['end_s']-event['start_s'];uid=event['segment_id']
    wav=directory/'segment.wav'; sf.write(wav,signal[round(event['start_s']*16000):round(event['end_s']*16000)],16000,subtype='FLOAT')
    q=q_media(face,directory,uid,identity,duration)
    item={**event,'utt_id':uid,'duration_s':duration,'wav':str(wav),'wav_sha256':sha256(wav),'face':face,'q_video':q}
    write_json_atomic(directory/'input.json',item);return item


def events_for(speaker,kind,timeline):
    if kind=='r4_vad':return rows(ROOT/f'dialogue_streams/{speaker}_events.jsonl')
    return [{'segment_id':e['utt_id'],'utt_id':e['utt_id'],'speaker_id':speaker,
             'cohort':'fmh' if speaker in FMH else 'et_test','source':'speech_segment','visual_only':False,
             'start_s':e['start_s'],'end_s':e['end_s'],'decision_s':e['end_s']}
            for e in timeline['entries'] if e['kind']=='utterance']


def run(kind,cohort):
    import soundfile as sf
    config=frozen_face();out,meta=start(f'inputs_{kind}_{cohort}')
    speakers=sorted(FMH if cohort=='fmh' else TEST);result=[]
    with ThreadPoolExecutor(max_workers=8) as media_pool:
        pending=deque()
        def flush(force=False):
            while pending and (force or len(pending)>=16):
                result.append(pending.popleft().result())
                write_json_atomic(out/'progress.json',{'done':len(result),'pending':len(pending)})
        for speaker in speakers:
            tpath=OLD/f'preflight/timeline_{speaker}.json' if kind=='r3' else ROOT/f'preflight/dialogue_{speaker}.json'
            apath=OLD/f'streams/{speaker}.wav' if kind=='r3' else ROOT/f'dialogue_streams/{speaker}.wav'
            timeline=read(tpath);signal,sr=sf.read(apath,dtype='float32');assert sr==16000 and signal.ndim==1
            events=events_for(speaker,kind,timeline);cursor=0;tracker=None;video=Video(timeline)
            def frames():
                for i in range(math.ceil(timeline['duration_s']*5)):yield i,video.frame(i/5)
            def finish_event(event,track):
                directory=out/event['segment_id'];directory.mkdir(parents=True,exist_ok=True)
                face=track.finish() if track else Tracker(directory/'face').finish()
                identity=hashlib.sha256((sha256(tpath)+json.dumps(event,sort_keys=True)).encode()).hexdigest()
                pending.append(media_pool.submit(make_item,event,face,directory,signal,identity));flush()
            try:
                for i,frame,boxes,_ in pipeline(frames(),config['selected_scale']):
                    while cursor<len(events) and i>=frame_index(events[cursor]['end_s']):
                        finish_event(events[cursor],tracker);tracker=None;cursor+=1
                    if cursor<len(events) and i>=frame_index(events[cursor]['start_s']):
                        event=events[cursor]
                        if tracker is None:tracker=Tracker(out/event['segment_id']/'face')
                        tracker.add(frame,i-frame_index(event['start_s']),boxes)
                while cursor<len(events):finish_event(events[cursor],tracker);tracker=None;cursor+=1
            finally:video.close()
            flush(True);storage_gate()
    path=out/'inputs.jsonl'
    path.write_text(''.join(json.dumps(x)+'\n' for x in result))
    finish(out,meta,{'kind':kind,'cohort':cohort,'n':len(result),'inputs':spec(path),'face_freeze':spec('configs/x06b_face_freeze.json'),'gpu_seconds':0})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['r3','r4_correct','r4_vad']);p.add_argument('cohort',choices=['fmh','et_test']);a=p.parse_args();run(a.kind,a.cohort)
