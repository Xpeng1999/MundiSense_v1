"""Infrastructure continuation using unchanged registered stage entrypoints."""
import os,sys,subprocess
from emotioncore.x07.common import *
protocol,registration=authority()
recovery=read(ROOT/'dependency_recovery.json')
assert recovery['predictions_before_resume']==0 and recovery['cpu_model_load_passed']
write_new(ROOT/'resume_driver.started.json',{'at':now(),'protocol_sha256':sha(PROTOCOL),'recovery':spec(ROOT/'dependency_recovery.json'),'launcher':spec(ROOT/'infra_resume_1.py'),'environment_pythonpath':os.environ['PYTHONPATH']})
reason='Missing timm at initialization before any prediction; restore byte-identical accepted X06b timm 0.9.16; retain failed GPU charge; frozen code/config unchanged. See dependency_recovery.json.'
for module,args,name in [('guard',['valid','--resume-reason',reason],'guard_valid_resume1'),
        ('metrics',['valid'],'metrics_valid'),('prepare',['test'],'prepare_test'),
        ('guard',['test'],'guard_test'),('metrics',['test'],'metrics_test')]:
    no_stop()
    atomic(ROOT/'stage_progress.json',{'stage':name,'at':now(),'continuation':'infra_resume_1'})
    command=[sys.executable,'-u','-m','emotioncore.x07.'+module,*args]
    with (ROOT/(name+'.log')).open('x') as log:
        status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    if status:
        write_new(ROOT/'resume_driver.interrupted.json',{'stage':name,'returncode':status,'at':now()})
        raise RuntimeError('Continuation interrupted: '+name)
    no_stop()
write_new(ROOT/'stages.completed.json',{'at':now(),'protocol_sha256':sha(PROTOCOL),'continuation':'infra_resume_1','recovery':spec(ROOT/'dependency_recovery.json')})
