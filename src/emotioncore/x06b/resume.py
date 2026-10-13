"""One authorized continuation; all original attempts remain untouched."""
import argparse,os,subprocess,sys
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face
from emotioncore.x06b.stages import execute


def active_processes():
    import psutil
    active=[]
    for p in psutil.process_iter(['pid','cmdline']):
        cmd=p.info['cmdline'] or []
        if p.pid!=os.getpid() and '-m' in cmd and cmd[cmd.index('-m')+1].startswith('emotioncore.x06b.'):
            active.append({'pid':p.pid,'cmd':cmd})
    return active


def steps():
    yield 'r3_analysis','et_test','--retry-authorized'
    yield 'budget','after_retry_r3_et_test'
    yield 'dialogue_streams',
    yield 'budget','after_dialogue_streams'
    for kind in ['r4_correct','r4_vad']:
        for cohort in ['fmh','et_test']:
            yield 'prepare',kind,cohort
            yield 'budget','after_inputs_'+kind+'_'+cohort
            yield 'gpu_guard',kind+'_'+cohort
            yield 'budget','after_gpu_'+kind+'_'+cohort
    for cohort in ['fmh','et_test']:
        yield 'r4_analysis',cohort
        yield 'budget','after_analysis_r4_'+cohort
    yield 'aishell_test',
    yield 'budget','after_aishell_test'
    yield 'gpu_guard','r5'
    yield 'budget','after_r5'
    yield 'curves',
    yield 'budget','after_curves'


def run():
    from emotioncore.x06b.recovery import validated_authorization
    no_stop();frozen_face();validated_authorization();out,meta=start('stages_resume1')
    try:
        records,failed=execute(out,list(steps()))
    except BaseException as exc:
        finish(out,meta,{'status':'stopped','failed_stage':'resume_driver','error':str(exc),'gpu_seconds':0})
        stop('execution','resume driver exception',spec(out/'summary.json'));raise
    finish(out,meta,{'status':'stopped' if failed else 'completed','failed_stage':failed,
        'stages':records,'authorization':spec(ROOT/'resume_authorization/summary.json'),'gpu_seconds':0})
    if failed:
        stop('execution',failed,spec(out/'summary.json'));raise RuntimeError('STOP '+failed)
    from emotioncore.x06b.delivery import run as audit
    try:audit()
    except BaseException:
        stop('delivery','final audit failed',spec(out/'summary.json'));raise


def launch():
    frozen_face()
    if active_processes():raise RuntimeError('STOP another experiment process is active')
    if (ROOT/'stages_resume1').exists():raise RuntimeError('STOP continuation already started')
    out,meta=start('stages_resume1_launch')
    with (out/'driver.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','emotioncore.x06b.resume'],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
                 'HF_HOME':str(ROOT/'cache/hf'),'XDG_CACHE_HOME':str(ROOT/'cache'),
                 'TORCH_HOME':str(ROOT/'cache/torch'),'TRITON_CACHE_DIR':str(ROOT/'cache/triton'),
                 'MODELSCOPE_CACHE':str(ROOT/'cache/modelscope')})
    finish(out,meta,{'status':'dispatched','pid':child.pid,'authorization':spec(ROOT/'resume_authorization/summary.json'),'gpu_seconds':0})
    print(child.pid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true')
    launch() if p.parse_args().launch else run()
