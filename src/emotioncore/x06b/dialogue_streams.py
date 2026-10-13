"""Render only the target voice plus low-energy noise; frozen online VAD."""
import math,numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face
from emotioncore.x06.service import Service
from emotioncore.x06.vad import Posterior
from emotioncore.x06.online_vad import posterior_chunks,LiveEvents


def lowest_rms(audio):
    x=np.asarray(audio,dtype=np.float64);n=min(1600,len(x))
    if not n:return 0.
    energy=np.r_[0.,np.cumsum(x*x)]
    return float(np.sqrt(max(0.,float(np.min(energy[n:]-energy[:-n])))/n))


def render(timeline,path):
    import soundfile as sf
    noise=0.;levels=[]
    with sf.SoundFile(path,'w',samplerate=16000,channels=1,subtype='FLOAT') as out:
        cursor=0
        for e in timeline['entries']:
            assert round(e['start_s']*16000)==cursor
            count=round(e['end_s']*16000)-cursor
            if e['kind']=='utterance':
                source=MEDIA/e['audio']['path']
                if sha256(source)!=e['audio']['sha256']:raise RuntimeError('STOP source audio changed')
                audio=Service.audio(source);noise=lowest_rms(audio)
                signal=np.pad(audio[:count],(0,max(0,count-len(audio))))
            else:
                signal=np.random.default_rng(e['noise_seed']).normal(0,noise,count).astype(np.float32)
                levels.append({'start_s':e['start_s'],'rms':noise,'noise_seed':e['noise_seed']})
            out.write(signal);cursor+=count
    return levels


def run():
    import soundfile as sf
    frozen_face();out,meta=start('dialogue_streams');model=Posterior('fsmn');sources={};counts={c:{'speech':0,'visual':0} for c in ['fmh','et_test']}
    for s in sorted(FMH|TEST):
        source=ROOT/f'preflight/dialogue_{s}.json';timeline=read(source);audio=out/f'{s}.wav'
        levels=render(timeline,audio);signal,sr=sf.read(audio,dtype='float32');assert sr==16000
        machine=LiveEvents(.3,.4);events=[];posterior=[]
        for chunk in posterior_chunks(model,signal):
            posterior.append(chunk['probabilities'])
            for p in chunk['probabilities']:events.extend(machine.advance(p))
        events.extend(machine.finish(timeline['duration_s']));events.sort(key=lambda r:(r['start_s'],r['end_s']))
        for i,event in enumerate(events):event.update(segment_id=f'{s}_{i:05d}',speaker_id=s,cohort=timeline['cohort'])
        epath=out/f'{s}_events.jsonl';epath.write_text(''.join(json.dumps(e)+'\n' for e in events))
        np.save(out/f'{s}_posterior.npy',np.concatenate(posterior))
        voice=sum(not e['visual_only'] for e in events);visual=len(events)-voice
        counts[timeline['cohort']]['speech']+=voice;counts[timeline['cohort']]['visual']+=visual
        sources[s]={'timeline':spec(source),'audio':spec(audio),'events':spec(epath),'posterior':spec(out/f'{s}_posterior.npy'),'noise_levels':levels,'speech':voice,'visual':visual}
        storage_gate();write_json_atomic(out/'progress.json',{'completed_speakers':list(sources)})
    finish(out,meta,{'status':'completed','streams':sources,'counts':counts,'vad':{'model':'fsmn','threshold':.3,'min_silence':.4},'gpu_seconds':0})

if __name__=='__main__':run()
