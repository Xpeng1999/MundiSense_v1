"""CPU-only registered FMH throughput and target-box selection."""
import argparse, math, time
import numpy as np
from emotioncore.x02.face import add_detections,rank_tracks,iou
from emotioncore.x06b.common import *
from emotioncore.x06b.video import Video
from emotioncore.x06b.detect import pipeline


def select(candidates):
    valid=[c for c in candidates if c['throughput']<=.5]
    if not valid: return None
    return max(valid,key=lambda c:(c['iou_ge_07_fraction'],c['scale']))


def frames(timelines,stats):
    for speaker,timeline in timelines.items():
        video=Video(timeline); entry_index=0
        try:
            for index in range(math.ceil(timeline['duration_s']*5)):
                when=index/5
                while when>=timeline['entries'][entry_index]['end_s'] and entry_index+1<len(timeline['entries']):entry_index+=1
                tic=time.monotonic();frame=video.frame(when);stats['decode_seconds']+=time.monotonic()-tic
                yield {'speaker':speaker,'frame':index,'window':entry_index},frame
        finally:video.close()


def target_boxes(path):
    result={}; tracks=[];key=None; count=0
    def close():
        if key is None:return
        ranked=rank_tracks(tracks,count)
        result[key]=ranked[0]['boxes'] if ranked else {}
    for row in rows(path):
        new=(row['speaker'],row['window'])
        if new!=key: close();key=new;tracks=[];count=0
        add_detections(tracks,row['frame'],row['boxes']);count+=1
    close();return result


def agreement(reference,candidate):
    values=[]; absent=0
    for key in reference.keys()|candidate.keys():
        a,b=reference.get(key,{}),candidate.get(key,{})
        for frame in a.keys()|b.keys():values.append(iou(a[frame],b[frame]) if frame in a and frame in b else 0.)
    return {'iou_median':float(np.median(values)) if values else 0.,
            'iou_ge_07_fraction':float(np.mean(np.asarray(values)>=.7)) if values else 0.,
            'iou_frame_denominator':len(values)}


def measure(scale, name, timelines):
    out,meta=start(name);stat={'decode_seconds':0.}; begin=time.monotonic();detection=0.;count=0
    with (out/'frames.jsonl').open('w') as dest:
        for item,frame,boxes,seconds in pipeline(frames(timelines,stat),scale):
            detection+=seconds;count+=1
            dest.write(json.dumps({**item,'boxes':boxes})+'\n')
            if count%500==0:
                dest.flush();storage_gate();write_json_atomic(out/'progress.json',{'frames':count,'seconds':time.monotonic()-begin,'speaker':item['speaker']})
    elapsed=time.monotonic()-begin; duration=sum(t['duration_s'] for t in timelines.values())
    summary={'scale':scale,'cohort':'fmh','speakers':sorted(timelines),'elapsed_s':elapsed,'media_s':duration,
             'throughput':elapsed/duration,'frames':count,**stat,'detector_work_seconds':detection,
             'frames_file':spec(out/'frames.jsonl'),'gpu_seconds':0,'includes_output_receipt_io':True}
    finish(out,meta,summary);return summary


def run():
    no_stop(); authority(); out,meta=start('r2'); timelines={s:source_timeline(s) for s in sorted(FMH)}
    if set(timelines)!=set(FMH):raise RuntimeError('STOP face selection must use FMH only')
    summaries=[]
    for scale,name in [(0.5,'r2_half'),(1/3,'r2_third')]:
        summaries.append(measure(scale,name,timelines))
    if not any(c['throughput']<=.5 for c in summaries):
        finish(out,meta,{'status':'STOP_r2_throughput','candidates':summaries,'gpu_seconds':0})
        stop('R2','both candidates exceed throughput 0.5',spec(out/'summary.json'));return
    reference=measure(1.,'r2_reference',timelines)
    original=target_boxes(checked(reference['frames_file']))
    for c in summaries:
        c.update(agreement(original,target_boxes(checked(c['frames_file']))))
        c['both_target_absent_frames']=c['frames']-c['iou_frame_denominator']
    chosen=select(summaries)
    frozen={'selected_scale':chosen['scale'],'detector':'YuNet unchanged','workers':8,'max_pending_frames':16,
            'source_speakers':sorted(FMH),'candidates':summaries,'reference':reference,'registration':spec(REG)}
    write_json_atomic(out/'freeze.json',frozen)
    finish(out,meta,{'status':'completed','selection':chosen,'freeze':spec(out/'freeze.json'),'candidates':summaries,'reference':reference,'gpu_seconds':0})


if __name__=='__main__':run()
