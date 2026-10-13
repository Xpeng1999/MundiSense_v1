"""Final immutable-source and budget audit; acceptance stays with reviewer."""
import argparse,fcntl,os,subprocess
import psutil
from emotioncore.x06b.common import *
from emotioncore.x06b.budget import ledger
from emotioncore.x06b.prepare import frozen_face


def provenance_record(path,history=None):
    record=read(path)
    if history and spec(path)==history['run_meta']:
        return history
    if not record.get('ended_at') or record['git']['dirty']:
        raise RuntimeError('STOP incomplete provenance: '+str(path))
    checked(record['summary'])
    return {'run_meta':spec(path),'summary':record['summary']}


def run(stopped=False):
    out,meta=start('delivery_audit');authority()
    active=[]
    for p in psutil.process_iter(['pid','cmdline']):
        cmd=p.info['cmdline'] or []
        if p.pid!=os.getpid() and '-m' in cmd and cmd[cmd.index('-m')+1].startswith('emotioncore.x06b.'):
            active.append({'pid':p.pid,'cmd':cmd})
    if active:raise RuntimeError('STOP final audit requires all experiment processes to exit')
    with (ROOT/'gpu.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fixed=['src/emotioncore/x02','src/emotioncore/x03','src/emotioncore/x04','src/emotioncore/x05','src/emotioncore/x06',
           'configs/calibration_bundle_v1.1.json','configs/x06_vad_freeze.json','configs/x06_silence_freeze.json','configs/x06_smoothing_freeze.json']
    changed=subprocess.check_output(['git','diff','--name-only',read(REG)['base_commit'],'HEAD','--',*fixed],text=True)
    if changed.strip():raise RuntimeError('STOP frozen baseline implementation changed')
    evidence={};history=None;authorization=None
    if (ROOT/'resume_authorization/summary.json').exists():
        from emotioncore.x06b.recovery import historical_failure,analysis_task
        history=historical_failure();authorization=spec(ROOT/'resume_authorization/summary.json')
    for p in sorted(list(ROOT.glob('*/run_meta.json'))+list(ROOT.glob('gpu_jobs/*/run_meta.json'))):
        if p.parent==out:continue
        evidence[str(p.parent)]=provenance_record(p,history)
    jobs=ledger();used=sum(r['charged_gpu_s'] for r in jobs)
    if used>LIMIT:raise RuntimeError('STOP GPU budget exceeded')
    if not stopped:
        no_stop();frozen_face()
        for task in ['r3_fmh','r3_et_test','r4_correct_fmh','r4_correct_et_test','r4_vad_fmh','r4_vad_et_test','r5']:
            assert read(ROOT/'gpu_jobs'/task/'summary.json')['status']=='completed'
        for task in ['r3_analysis_fmh','r3_analysis_et_test','r4_analysis_fmh','r4_analysis_et_test','aishell_test_04','cpu_integration']:
            if history and task=='r3_analysis_et_test':task=analysis_task('et_test')
            assert read(ROOT/task/'summary.json')['status']=='completed'
        old=rows(checked(read(OLD/'realtime_H3/summary.json')['events_file']))
        new=rows(checked(read(ROOT/'r5/summary.json')['events_file']))
        keys=['segment_id','speaker_id','start_s','end_s','decision_s','source','visual_only']
        assert [{k:r[k] for k in keys} for r in old]==[{k:r[k] for k in keys} for r in new], 'R5 event schedule differs from X06a'
    finish(out,meta,{'status':'stopped' if stopped else 'complete_delivery','gpu_hours':used/3600,'gpu_limit_hours':14,
        'gpu_ledger':jobs,'frozen_paths_unchanged':fixed,'evidence':evidence,'active_processes':active,
        'r5_schedule_identical_to_x06a':not stopped,'stop_receipt':spec(ROOT/'STOP.json') if (ROOT/'STOP.json').exists() else None,
        'historical_failure':history,'recovery_authorization':authorization,
        'gpu_seconds':0,'acceptance_is_separate':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stopped',action='store_true');run(p.parse_args().stopped)
