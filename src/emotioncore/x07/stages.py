"""Strict L2 -> L3 -> L4 driver. No automatic retries or parameter changes."""
import sys
from emotioncore.x07.common import *

def run():
    authority()
    write_new(ROOT/'stages.started.json',{'at':now(),'protocol_sha256':sha(PROTOCOL)})
    for module,args in [('leakage',[]),('prepare',['valid']),('guard',['valid']),('metrics',['valid']),
                        ('prepare',['test']),('guard',['test']),('metrics',['test'])]:
        no_stop()
        command=[sys.executable,'-u','-m','emotioncore.x07.'+module,*args]
        name='_'.join([module,*args]);atomic(ROOT/'stage_progress.json',{'stage':name,'at':now()})
        with (ROOT/(name+'.log')).open('x') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            write_new(ROOT/'stages.interrupted.json',{'stage':name,'returncode':code,'at':now()})
            raise RuntimeError('Stage interrupted: '+name)
        no_stop()
    write_new(ROOT/'stages.completed.json',{'at':now(),'protocol_sha256':sha(PROTOCOL)})

if __name__=='__main__':
    run()
