"""Causal H3 replay with incremental crops and bounded frame retention."""
import math,os,time,queue,traceback,multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face,make_item
from emotioncore.x06b.face import Tracker
from emotioncore.x06b.video import Video
from emotioncore.x06b.detect import pipeline
from emotioncore.x06.online_vad import posterior_chunks,LiveEvents
from emotioncore.x06.vad import Posterior
from emotioncore.x06.prepare_segments import frame_index
from emotioncore.x06.service import Service
from emotioncore.x06.decision import apply
from emotioncore.x06.live import rss_tree,quantiles


class WindowBank:
    def __init__(self,root):self.root=root;self.frames=OrderedDict();self.tracks={};self.high_water=0
    def add(self,index,frame,boxes):
        self.frames[index]=(frame,boxes);self.high_water=max(self.high_water,len(self.frames))
        if len(self.frames)>32:raise RuntimeError('STOP frame buffer exceeds fixed 32-frame bound')
    def update(self,key,end):
        start=key[1];first,last=frame_index(start),frame_index(end)
        if key not in self.tracks:
            self.tracks[key]=(Tracker(self.root/f'{key[0]}_{first:08d}'),first)
        track,next_index=self.tracks[key]
        for index,(frame,boxes) in self.frames.items():
            if next_index<=index<last:
                if index!=next_index:raise RuntimeError('STOP incremental tracker missed an input frame')
                track.add(frame,index-first,boxes);next_index=index+1
        self.tracks[key]=(track,next_index)
    def finish(self,key,end):
        self.update(key,end);track,next_index=self.tracks.pop(key)
        if next_index!=frame_index(end):raise RuntimeError('STOP incomplete event video')
        return track.finish()
    def prune(self,keep,retain_from):
        for key in list(self.tracks):
            if key not in keep:
                track,_=self.tracks.pop(key);track.tracks.clear();track.paths.clear();track.closed=True
        for index in list(self.frames):
            if index/5<retain_from:del self.frames[index]


def producer(sources,config,out,messages,begin,realtime=True):
    import soundfile as sf
    try:
        vad=Posterior('fsmn');messages.put({'kind':'ready'});begin.wait()
        with ThreadPoolExecutor(max_workers=8) as detector_pool:
            for speaker,source in sources.items():
                timeline=read(checked(source['timeline']));signal,sr=sf.read(checked(source['audio']),dtype='float32')
                assert sr==16000 and signal.ndim==1
                video=Video(timeline);machine=LiveEvents(.3,.4);bank=WindowBank(out/'crops'/speaker)
                started=time.monotonic();next_frame=event_number=0;totals={'decode':0.,'detection_work':0.,'decode_detection_wall':0.,'vad':0.}
                def wait_until(when):
                    if realtime:time.sleep(max(0.,started+when-time.monotonic()))
                def feed(when):
                    nonlocal next_frame
                    pipeline_started=time.monotonic()
                    def frames():
                        nonlocal next_frame
                        while next_frame/5<min(when,timeline['duration_s']):
                            i=next_frame;tic=time.monotonic();frame=video.frame(i/5);totals['decode']+=time.monotonic()-tic
                            next_frame+=1;yield i,frame
                    for i,frame,boxes,seconds in pipeline(frames(),config['selected_scale'],executor=detector_pool):
                        bank.add(i,frame,boxes);totals['detection_work']+=seconds
                    totals['decode_detection_wall']+=time.monotonic()-pipeline_started
                def window_key(e):return ('visual' if e['visual_only'] else 'speech',e['start_s'])
                def dispatch(events,windows,available,vad_seconds):
                    nonlocal event_number
                    endpoint=time.monotonic();feed(available)
                    for key,end in windows.items():bank.update(key,min(end,available))
                    for event in events:
                        tic=time.monotonic();face=bank.finish(window_key(event),event['end_s'])
                        identifier=f'{speaker}_{event_number:05d}';event_number+=1
                        event={**event,'segment_id':identifier,'speaker_id':speaker,'cohort':'et_test'}
                        directory=out/'inputs'/identifier;directory.mkdir(parents=True,exist_ok=False)
                        identity=__import__('hashlib').sha256((source['timeline']['sha256']+json.dumps(event,sort_keys=True)).encode()).hexdigest()
                        item=make_item(event,face,directory,signal,identity)
                        item.update(endpoint_monotonic=endpoint,input_stream_started_monotonic=started,
                            input_schedule_lag_s=max(0.,endpoint-started-event['decision_s']),
                            timing_input={'vad':vad_seconds/max(1,len(events)),'incremental_tracking_crop':face['face_seconds'],
                                          'after_endpoint_preparation':time.monotonic()-tic})
                        write_json_atomic(directory/'input.json',item);messages.put({'kind':'segment','item':item})
                    active={('visual',machine.next_silence-2)}
                    if machine.voice.start is not None:active.add(('speech',machine.voice.start*.01))
                    retain=machine.voice.index*.01
                    if machine.voice.start is not None:retain=min(retain,(machine.voice.last+1)*.01)
                    bank.prune(active,retain)
                try:
                    for chunk in posterior_chunks(vad,signal,wait_until=wait_until):
                        totals['vad']+=chunk['vad_seconds'];events=[];windows={}
                        for p in chunk['probabilities']:
                            windows[('visual',machine.next_silence-2)]=machine.next_silence
                            ready=machine.advance(p);events.extend(ready)
                            for e in ready:windows[window_key(e)]=e['end_s']
                            if machine.voice.start is not None:windows[('speech',machine.voice.start*.01)]=(machine.voice.last+1)*.01
                            windows[('visual',machine.next_silence-2)]=machine.next_silence
                        dispatch(events,windows,chunk['available_s'],chunk['vad_seconds'])
                    final=machine.finish(timeline['duration_s'])
                    dispatch(final,{window_key(e):e['end_s'] for e in final},timeline['duration_s'],0.)
                finally:video.close()
                messages.put({'kind':'stream_finished','speaker_id':speaker,'events':event_number,'duration_s':timeline['duration_s'],
                              'wall_s':time.monotonic()-started,'totals':totals,'raw_frame_high_water':bank.high_water})
        messages.put({'kind':'done','producer_rss':rss_tree(os.getpid())})
    except BaseException:
        messages.put({'kind':'error','traceback':traceback.format_exc()});raise


def run():
    if os.environ.get('X06B_GPU_GUARDED')!='r5':raise RuntimeError('Use X06b watchdog')
    config=frozen_face();out,meta=start('r5');sources=read(OLD/'streams/summary.json')['streams'];sources={s:sources[s] for s in sorted(TEST)}
    context=mp.get_context('spawn');messages=context.Queue(maxsize=2);begin_event=context.Event()
    child=context.Process(target=producer,args=(sources,config,out,messages,begin_event));child.start()
    samples=[];events=[];streams={};state={};last_sample=0.;producer_rss=0
    try:
        service=Service();first=messages.get(timeout=300)
        if first['kind']!='ready':raise RuntimeError(first)
        service.torch.cuda.reset_peak_memory_stats();begin=time.monotonic();begin_event.set()
        with (out/'events.jsonl').open('w') as dest,(out/'memory.jsonl').open('w') as memory:
            while True:
                elapsed=time.monotonic()-begin
                if elapsed-last_sample>=1 or not samples:
                    sample={'elapsed_s':elapsed,'rss_bytes':rss_tree(os.getpid())};samples.append(sample);memory.write(json.dumps(sample)+'\n');memory.flush();last_sample=elapsed
                try:message=messages.get(timeout=.5)
                except queue.Empty:
                    if not child.is_alive():raise RuntimeError('STOP producer exited')
                    continue
                if message['kind']=='error':raise RuntimeError(message['traceback'])
                if message['kind']=='done':producer_rss=message['producer_rss'];break
                if message['kind']=='stream_finished':streams[message['speaker_id']]=message;continue
                item=message['item'];speech=not item['visual_only'];p,face,timing=service.components(item,'H3',has_speech=speech)
                tic=time.monotonic();decision=apply(service.bundle,p,'H3',has_face=face,speech=speech,scores=item['face']['track_scores'],alpha=0,state=state,stream_id=item['speaker_id']);timing['fusion']=time.monotonic()-tic
                record={k:item[k] for k in ['segment_id','speaker_id','start_s','end_s','decision_s','visual_only','source']}
                record.update(timestamp=now(),component_probs=p,has_face=face,track_scores=item['face']['track_scores'],**decision,
                              timing={**item['timing_input'],**timing},input_schedule_lag_s=item['input_schedule_lag_s'])
                dest.write(json.dumps(record)+'\n');dest.flush();written=time.monotonic()
                record.update(written_monotonic=written,stream_started_monotonic=item['input_stream_started_monotonic'],
                              latency_s=written-item['endpoint_monotonic'],scheduled_latency_s=written-item['input_stream_started_monotonic']-item['decision_s'])
                events.append(record)
                if len(events)%25==0:storage_gate();write_json_atomic(out/'progress.json',{'events':len(events),'last':item['segment_id'],'elapsed_s':written-begin})
        wall=time.monotonic()-begin;child.join(timeout=30)
        if child.exitcode!=0:raise RuntimeError('STOP producer failed')
        end_rss=rss_tree(os.getpid())+producer_rss;samples.append({'elapsed_s':wall,'rss_bytes':end_rss})
        (out/'memory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
        (out/'latency.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
        voice=[r for r in events if not r['visual_only']];duration=sum(read(s['timeline']['path'])['duration_s'] for s in sources.values())
        after10=[r for r in samples if r['elapsed_s']>=600];baseline=after10[0]['rss_bytes'];peak=max(r['rss_bytes'] for r in after10)
        lags=[r['input_schedule_lag_s'] for r in events];firsthalf=[];secondhalf=[];offset=0.;offsets={}
        for s in sources:offsets[s]=offset;offset+=read(sources[s]['timeline']['path'])['duration_s']
        for e in events:(firsthalf if offsets[e['speaker_id']]+e['decision_s']<duration/2 else secondhalf).append(e['input_schedule_lag_s'])
        truth_delays=[];uncovered=[]
        for s in sources:
            current=[e for e in voice if e['speaker_id']==s]
            for truth in source_timeline(s)['entries']:
                if truth['kind']!='utterance':continue
                matched=[e for e in current if e['start_s']<truth['end_s'] and e['end_s']>truth['start_s']]
                if not matched:uncovered.append(truth['utt_id']);continue
                # A truth with multiple overlapping outputs is complete only
                # when its last contributing event has been written.
                end=max(matched,key=lambda e:e['written_monotonic']);truth_delays.append(end['written_monotonic']-end['stream_started_monotonic']-truth['end_s'])
        scheduled=quantiles([r['scheduled_latency_s'] for r in voice]);lagq=quantiles(lags);beforeq=quantiles(firsthalf);afterq=quantiles(secondhalf)
        summary={'status':'completed','system':'H3','events':len(events),'speech_events':len(voice),'streams':streams,'wall_s':wall,'media_s':duration,'wall_over_media':wall/duration,
            'latency_from_vad':quantiles([r['latency_s'] for r in voice]),'latency_from_scheduled_vad':scheduled,'latency_from_truth_end':quantiles(truth_delays),
            'truth_delay_n':len(truth_delays),'uncovered_truth_ids':uncovered,'truth_delay_rule':'last overlapping event write; uncovered separately counted, not assigned zero',
            'input_lag':{**lagq,'max':max(lags)},'first_half_lag':beforeq,'second_half_lag':afterq,
            'rss_at_10min':baseline,'rss_10min_sample_s':after10[0]['elapsed_s'],'rss_end':end_rss,'rss_after_10min_peak':peak,
            'rss_end_growth_after_10min':end_rss/baseline-1,'rss_peak_growth_after_10min':peak/baseline-1,
            'gpu_peak_allocated_bytes':service.torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':service.torch.cuda.max_memory_reserved(),
            'events_file':spec(out/'events.jsonl'),'latency_file':spec(out/'latency.jsonl'),'memory_file':spec(out/'memory.jsonl'),
            'stage_seconds':{k:{**quantiles([r['timing'].get(k,0) for r in events]),'total':sum(r['timing'].get(k,0) for r in events)} for k in set().union(*(r['timing'] for r in events))},
            'acceptance':{'wall':wall/duration<=1.,'lag':lagq['p95']<=2 and afterq['p95']<=beforeq['p95'],
                'scheduled_latency':scheduled['p50']<=1.5 and scheduled['p95']<=2.5,'memory':end_rss/baseline-1<.1},'sources':sources}
        finish(out,meta,summary)
    finally:
        if child.is_alive():child.terminate();child.join(timeout=10)

if __name__=='__main__':run()
