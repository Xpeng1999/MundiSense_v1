"""Forecast after every completed stage; GPU wall time includes waits."""
import argparse
from emotioncore.x06b.common import *


def ledger():return [{'source':spec(p),**read(p)} for p in sorted((ROOT/'gpu_jobs').glob('*/summary.json'))]

def run(stage):
    no_stop();out,meta=start('budget_'+stage);initial=read(ROOT/'preflight/summary.json');log=ledger()
    if any(r['status']!='completed' for r in log):raise RuntimeError('STOP failed GPU job requires review')
    used=sum(r['charged_gpu_s'] for r in log);done={Path(r['source']['path']).parent.name for r in log}
    rate=initial['h3_seconds_per_segment'];visual=initial['visual_seconds_per_segment']
    measured=[]
    for r in log:
        task=Path(r['source']['path']).parent.name
        if task.startswith(('r3_','r4_correct_')):
            n=read(ROOT/task/'summary.json')['n'];measured.append(r['charged_gpu_s']/n)
    if measured:rate=measured[-1]
    future={}
    for kind in ['r3','r4_correct']:
        for cohort,n in [('fmh',3266),('et_test',1929)]:
            task=kind+'_'+cohort
            if task not in done:future[task]=n*rate
    cuts=ROOT/'dialogue_streams/summary.json'
    if cuts.exists():
        counts=read(cuts)['counts'];exact=True
    else:
        counts={cohort:{'speech':initial['speech_segments_estimate']*n/5195,'visual':initial['silence_windows_estimate']*n/5195} for cohort,n in [('fmh',3266),('et_test',1929)]};exact=False
    for cohort,c in counts.items():
        if 'r4_vad_'+cohort not in done:future['r4_vad_'+cohort]=c['speech']*rate+c['visual']*visual
    if 'r5' not in done:future['r5']=initial['stage_gpu_seconds']['R5']
    base=sum(future.values());reserve=.2*base+(300 if base else 0);total=(used+base+reserve)/3600
    result={'status':'passed' if total<=14 else 'STOP_budget_estimate','stage':stage,'used_gpu_hours':used/3600,
        'future_gpu_seconds':future,'reserve_seconds':reserve,'estimated_total_gpu_hours':total,
        'limit_gpu_hours':14,'ledger':[r['source'] for r in log],'r4_exact_counts':exact,'speech_seconds_per_event':rate,'gpu_seconds':0}
    finish(out,meta,result);write_json_atomic(ROOT/'budget_current.json',spec(out/'summary.json'))
    if total>14:stop(stage,'forecast exceeds 14 hours',spec(out/'summary.json'));raise RuntimeError('STOP budget')


def gate(task):
    no_stop();authority();p=read(ROOT/'budget_current.json');b=read(checked(p))
    if b['status']!='passed':raise RuntimeError('STOP no passing budget')
    log=ledger();expected=b.get('ledger',[])
    if expected!=[r['source'] for r in log]:raise RuntimeError('STOP reestimate budget after previous GPU job')
    if task.startswith('r4_') and not b.get('r4_exact_counts'):raise RuntimeError('STOP exact frozen VAD counts required before R4 GPU')
    if task!='r3_fmh':
        a=read(ROOT/'r3_analysis_fmh/summary.json')
        if a['status']!='completed':raise RuntimeError('STOP FMH R3 gate not passed')
    return sum(r['charged_gpu_s'] for r in log)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');run(p.parse_args().stage)
