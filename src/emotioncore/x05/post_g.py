"""Wait for both guarded FMH jobs, then run the frozen CPU-only delivery stages."""
import os
from pathlib import Path
import subprocess
import sys
import time
from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x05.common import ROOT,read,start,finish,spec,storage_gate,now
from emotioncore.x05.bundle import verify_method_freeze

FREEZE_SHA='8e29dd19beaa3401fb2103542b5333d392d89b9c3e9f819153c313e8fcbcebbe'


def run():
    output=ROOT/'post_g';meta=start(output)
    freeze=Path('configs/x05_method_freeze.json')
    verify_method_freeze(freeze,FREEZE_SHA,require_committed=True)
    meta['method_freeze']=spec(freeze);steps=[]
    write_json_atomic(output/'run_meta.json',meta)
    write_json_atomic(output/'progress.json',{'phase':'waiting_for_G','at':now()})
    while True:
        storage_gate()
        receipts={p.parent.name:read(p) for p in (ROOT/'gpu_jobs').glob('*/summary.json')}
        if any(r['status']!='completed' for r in receipts.values()):
            finish(output,meta,{'status':'stopped','reason':'GPU job stopped','steps':steps})
            raise RuntimeError('STOP GPU job stopped; no second-stage analysis')
        if all(name in receipts for name in ('reproduce_a3','fmhx_a3','fmh_q_et','fmh_q_mix')):
            break
        time.sleep(20)
    stages=[['emotioncore.x05.deployment','--freeze-sha256',FREEZE_SHA],
            ['emotioncore.x05.figures'],['emotioncore.x05.verify','--name','final_tests'],
            ['emotioncore.x05.report']]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
         'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}
    for args in stages:
        command=[sys.executable,'-u','-m',*args]
        write_json_atomic(output/'progress.json',{'phase':args[0],'at':now(),'command':command})
        with (output/(args[0].split('.')[-1]+'.log')).open('w') as log:
            started=time.monotonic();result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        steps.append({'command':command,'returncode':result.returncode,'elapsed_s':time.monotonic()-started})
        if result.returncode:
            finish(output,meta,{'status':'stopped','reason':args[0],'steps':steps})
            raise RuntimeError(f'STOP {args[0]} failed')
    finish(output,meta,{'status':'completed','steps':steps})
    print('X05 CPU delivery stages completed')


if __name__=='__main__':run()
