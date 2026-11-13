"""CPU-only single-utterance preparation with the frozen X06b frame path."""
import argparse
import concurrent.futures
import math
import subprocess
import time
from emotioncore.x07.common import *
from emotioncore.x04.prepare import probe,storage_gate
from emotioncore.x06b.video import Video
from emotioncore.x06b.face import Tracker
from emotioncore.x06b.detect import pipeline
from emotioncore.x06.prepare_segments import q_media

MEDIA=Path('/home/administrator/emotion-data/emotioncore_derived/chsims_v2/raw/Raw')

class MediaError(Exception):
    pass

def one(task):
    split,uid,protocol_sha,config=task
    directory=ROOT/split/'inputs'/uid
    directory.mkdir(parents=True,exist_ok=True)
    dest=directory/'input.json'
    if dest.exists():
        old=read(dest)
        if old['protocol_sha256']!=protocol_sha or old['id']!=uid:
            raise RuntimeError('STOP prepared input changed')
        return spec(dest)
    video_id,clip=uid.split('__');source=MEDIA/video_id/(clip+'.mp4')
    result={'id':uid,'protocol_sha256':protocol_sha,'media_path':str(source),'started_at':now()}
    begin=time.monotonic()
    try:
        info=probe(source)
        if info['status']!='ok':
            raise MediaError(info['status'])
        if not any(s.get('codec_type')=='audio' for s in info['streams']):
            raise MediaError('missing_audio_stream')
        duration=info['duration_s']; wav=directory/'segment.wav'
        command=['ffmpeg','-nostdin','-v','error','-xerror','-i',str(source),'-vn','-ac','1','-ar','16000','-c:a','pcm_f32le','-y',str(wav)]
        run=subprocess.run(command,capture_output=True,text=True,timeout=120)
        if run.returncode:
            raise MediaError('audio_decode: '+run.stderr[-600:])
        import soundfile as sf
        import numpy as np
        audio,sr=sf.read(wav,dtype='float32')
        if sr!=16000 or not len(audio) or not np.isfinite(audio).all():
            raise MediaError('invalid_audio')
        timeline={'duration_s':duration,'entries':[{'kind':'utterance','start_s':0.,'end_s':duration,'video':{'path':str(source)}}]}
        video=Video(timeline);tracker=Tracker(directory/'face')
        def frames():
            for i in range(math.ceil(duration*5)):
                try:
                    frame=video.frame(i/5)
                except RuntimeError as exc:
                    raise MediaError(str(exc)) from exc
                yield i,frame
        try:
            for i,frame,boxes,_ in pipeline(frames(),config['selected_scale'],config['workers'],config['max_pending_frames']):
                tracker.add(frame,i,boxes)
        finally:
            video.close()
        face=tracker.finish()
        if face['status']=='failed':
            raise RuntimeError('STOP face preparation unexpectedly failed')
        q=q_media(face,directory,uid,info['sha256'],duration)
        if (q is not None and q['status']!='ok'
                and 'No X02 crop passed the fixed face size/confidence filter' not in q.get('error','')):
            raise RuntimeError('STOP Q media preparation failed')
        result.update(status='ok',utt_id=uid,duration_s=duration,wav=str(wav),wav_sha256=sha(wav),
                      face=face,q_video=q,source_sha256=info['sha256'])
    except MediaError as exc:
        result.update(status='missing_or_decode_failed',failure=str(exc))
    result.update(ended_at=now(),elapsed_s=time.monotonic()-begin)
    write_new(dest,result)
    return spec(dest)

def run(split):
    protocol,receipt=authority();stage_gate(split)
    if (ROOT/split/'metrics.started.json').exists() or (ROOT/split/'complete.json').exists():
        raise RuntimeError('STOP completed stage cannot be prepared again')
    start=now();verify_precedes(receipt,start)
    ids=protocol['inventory']['splits'][split]['ids']
    config=read('configs/x06b_face_freeze.json')
    out=ROOT/split;out.mkdir(exist_ok=True)
    if (out/'inputs.json').exists():
        raise RuntimeError('STOP input preparation already completed')
    results={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        tasks=[(split,uid,sha(PROTOCOL),config) for uid in ids]
        for uid,item in zip(ids,pool.map(one,tasks)):
            results[uid]=item
            storage_gate()
            atomic(out/'prepare_progress.json',{'done':len(results),'total':len(ids),'at':now()})
    write_new(out/'inputs.json',{'inputs':results,'protocol_sha256':sha(PROTOCOL),'started_at':start,
        'ended_at':now(),'n':len(ids),'vad_used':False,'labels_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['valid','test']);run(p.parse_args().split)
