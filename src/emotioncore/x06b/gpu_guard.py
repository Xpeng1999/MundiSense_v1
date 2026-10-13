"""Single GPU job, 14h cumulative wall watchdog and storage gate."""
import argparse,fcntl,os,signal,subprocess,sys,time
from emotioncore.x06b.common import *
from emotioncore.x06b.budget import gate
from emotioncore.x06b.prepare import frozen_face

TASKS={f'{kind}_{cohort}':('infer',[kind,cohort]) for kind in ['r3','r4_correct','r4_vad'] for cohort in ['fmh','et_test']}
TASKS['r5']=('realtime',[])

def run(task):
    used=gate(task);frozen_face();module,args=TASKS[task]
    with (ROOT/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out,meta=start('gpu_jobs/'+task);begin=time.monotonic();failure=None
        command=[sys.executable,'-u','-m','emotioncore.x06b.'+module,*args]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','X06B_GPU_GUARDED':task,'PYTHONDONTWRITEBYTECODE':'1',
             'HF_HOME':str(ROOT/'cache/hf'),'XDG_CACHE_HOME':str(ROOT/'cache'),'TORCH_HOME':str(ROOT/'cache/torch'),
             'TRITON_CACHE_DIR':str(ROOT/'cache/triton'),'MODELSCOPE_CACHE':str(ROOT/'cache/modelscope')}
        with (out/'child.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                while child.poll() is None:
                    elapsed=time.monotonic()-begin
                    if used+elapsed>=LIMIT:raise RuntimeError('GPU total reached 14 hours')
                    storage_gate();write_json_atomic(out/'progress.json',{'pid':child.pid,'elapsed_gpu_s':elapsed,'total_gpu_hours':(used+elapsed)/3600})
                    time.sleep(5)
            except BaseException as exc:
                failure=str(exc);os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        charged=time.monotonic()-begin;status='completed' if child.returncode==0 and failure is None else 'failed'
        finish(out,meta,{'status':status,'returncode':child.returncode,'failure':failure,'charged_gpu_s':charged,'command':command,'log':spec(out/'child.log')})
        if status!='completed':stop(task,failure or 'GPU child failure',spec(out/'summary.json'));raise RuntimeError('STOP GPU job failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=TASKS);run(p.parse_args().task)
