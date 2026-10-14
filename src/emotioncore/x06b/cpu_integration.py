"""FMH-only producer integration; no emotion model or GPU execution."""
import threading,numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face
from emotioncore.x06b.realtime import producer
from emotioncore.x06.online_vad import LiveEvents,posterior_chunks
from emotioncore.x06.vad import Posterior


def run():
    import soundfile as sf
    config=frozen_face();out,meta=start('cpu_integration');t=source_timeline('06')
    entries=t['entries'][:5];end=entries[-1]['end_s'];tiny={**t,'entries':entries,'duration_s':end,'utterances':3}
    write_json_atomic(out/'timeline.json',tiny)
    with sf.SoundFile(OLD/'streams/06.wav') as f:signal=f.read(round(end*16000),dtype='float32')
    sf.write(out/'audio.wav',signal,16000,subtype='FLOAT')
    sources={'06':{'timeline':spec(out/'timeline.json'),'audio':spec(out/'audio.wav')}}
    from emotioncore.x06.streams import Video as PreviousVideo
    from emotioncore.x06b.video import Video as NewVideo
    old_video,new_video=PreviousVideo(tiny),NewVideo(tiny);equal=total=0
    try:
        for i in range(__import__('math').ceil(end*5)):
            equal+=int(np.array_equal(old_video.frame(i/5),new_video.frame(i/5)));total+=1
    finally:old_video.close();new_video.close()
    class Messages:
        def __init__(self):self.values=[]
        def put(self,value):self.values.append(value)
    messages=Messages();ready=threading.Event();ready.set()
    producer(sources,config,out,messages,ready,realtime=False)
    fresh=[m['item'] for m in messages.values if m['kind']=='segment']
    model=Posterior('fsmn');machine=LiveEvents(.3,.4);expected=[]
    for chunk in posterior_chunks(model,signal):
        for p in chunk['probabilities']:expected.extend(machine.advance(p))
    expected.extend(machine.finish(end))
    keys=['start_s','end_s','decision_s','source','visual_only']
    assert [{k:r[k] for k in keys} for r in fresh]==[{k:r[k] for k in keys} for r in expected]
    assert fresh and all(Path(r['wav']).exists() for r in fresh)
    stream=[m for m in messages.values if m['kind']=='stream_finished'][0]
    assert stream['raw_frame_high_water']<=32
    (out/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fresh))
    finish(out,meta,{'status':'completed','cohort':'fmh','speaker':'06','utterances':3,'events':len(fresh),
        'old_new_decoder_equal_frames':equal,'decoder_compared_frames':total,'event_schedule_equal':True,'raw_frame_high_water':stream['raw_frame_high_water'],'duration_s':end,
        'inputs':spec(out/'inputs.jsonl'),'gpu_seconds':0,'emotion_inference':False})

if __name__=='__main__':run()
