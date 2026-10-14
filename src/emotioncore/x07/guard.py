"""A cumulative four-hour subprocess wall-clock budget, including initialization."""
import argparse
import fcntl
import signal
import time
from emotioncore.x07.common import *
from emotioncore.x04.prepare import storage_gate

LIMIT=4*3600

def run(split,resume_reason=None):
    import sys
    protocol,_=authority(assets=True);stage_gate(split)
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=0.
        for path in sorted((ROOT/'gpu_jobs').glob('*/receipt.json')):
            job=read(path)
            if job['status']=='running':
                raise RuntimeError('STOP unclosed GPU receipt; account for its elapsed time before resume')
            used+=job['charged_seconds']
        if used>=LIMIT:
            stop('budget','GPU four-hour budget exhausted');raise RuntimeError('STOP GPU budget')
        out=ROOT/'gpu_jobs'/f'{split}_{time.time_ns()}'
        out.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-u','-m','emotioncore.x07.infer',split]
        if resume_reason:
            command+=['--resume-reason',resume_reason]
        environment={**os.environ,'X07_GPU_GUARDED':split,'CUDA_VISIBLE_DEVICES':'0',
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1',
            'HF_HOME':str(ROOT/'cache/hf'),'XDG_CACHE_HOME':str(ROOT/'cache'),
            'TORCH_HOME':str(ROOT/'cache/torch'),'TRITON_CACHE_DIR':str(ROOT/'cache/triton')}
        begin=time.monotonic();failure=None
        receipt={'status':'running','split':split,'started_at':now(),'charged_seconds':0.,
                 'prior_seconds':used,'limit_seconds':LIMIT,'protocol_sha256':sha(PROTOCOL),'command':command}
        atomic(out/'receipt.json',receipt)
        with (out/'child.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            try:
                while child.poll() is None:
                    elapsed=time.monotonic()-begin
                    if used+elapsed>=LIMIT:
                        stop('budget','GPU four-hour limit reached');raise RuntimeError('GPU budget reached')
                    try:
                        storage_gate()
                    except Exception as exc:
                        stop('storage',exc);raise
                    receipt.update(charged_seconds=elapsed,pid=child.pid,updated_at=now())
                    atomic(out/'receipt.json',receipt)
                    time.sleep(2)
            except BaseException as exc:
                failure=repr(exc)
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        receipt.update(status='completed' if child.returncode==0 and failure is None else 'failed',
            charged_seconds=time.monotonic()-begin,ended_at=now(),returncode=child.returncode,failure=failure,
            log=spec(out/'child.log'))
        atomic(out/'receipt.json',receipt)
        if receipt['status']!='completed':
            raise RuntimeError('GPU infrastructure failure; preserve completed predictions and record cause before resuming')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['valid','test']);p.add_argument('--resume-reason')
    a=p.parse_args();run(a.split,a.resume_reason)
