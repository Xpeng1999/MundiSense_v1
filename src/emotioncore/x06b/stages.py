"""Sequential bounded stages; freeze, scientific and budget gates fail closed."""
import argparse,os,subprocess,sys
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face


def steps():
    yield 'cpu_integration',
    yield 'budget','after_r2'
    for cohort in ['fmh','et_test']:
        yield 'prepare','r3',cohort
        yield 'budget','after_inputs_r3_'+cohort
        yield 'gpu_guard','r3_'+cohort
        yield 'budget','after_gpu_r3_'+cohort
        yield 'r3_analysis',cohort
        yield 'budget','after_analysis_r3_'+cohort
    yield 'dialogue_streams',
    yield 'budget','after_dialogue_streams'
    yield 'aishell_test',
    yield 'budget','after_aishell_test'
    for kind in ['r4_correct','r4_vad']:
        for cohort in ['fmh','et_test']:
            yield 'prepare',kind,cohort
            yield 'budget','after_inputs_'+kind+'_'+cohort
            yield 'gpu_guard',kind+'_'+cohort
            yield 'budget','after_gpu_'+kind+'_'+cohort
    for cohort in ['fmh','et_test']:
        yield 'r4_analysis',cohort
        yield 'budget','after_analysis_r4_'+cohort
    yield 'gpu_guard','r5'
    yield 'budget','after_r5'


def execute(out,stage_plan):
    records=[]
    for index,step in enumerate(stage_plan):
        no_stop();module,*args=step;command=[sys.executable,'-u','-m','emotioncore.x06b.'+module,*args]
        label='_'.join(step);record={'stage':label,'command':command,'started_at':now()}
        write_json_atomic(out/'progress.json',{'current':record,'completed':records})
        with (out/f'{index:02d}_{label}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        record.update(ended_at=now(),returncode=result.returncode,log=spec(out/f'{index:02d}_{label}.log'));records.append(record)
        if result.returncode:return records,label
    return records,None


def run():
    frozen_face();out,meta=start('stages');records,failed=execute(out,list(steps()))
    finish(out,meta,{'status':'stopped' if failed else 'completed','failed_stage':failed,'stages':records,'gpu_seconds':0})
    if failed:
        if not (ROOT/'STOP.json').exists():stop('execution',failed,spec(out/'summary.json'))
        raise RuntimeError('STOP '+failed)
    from emotioncore.x06b.delivery import run as audit
    audit()


def launch():
    frozen_face();out,meta=start('stages_launch')
    with (out/'driver.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','emotioncore.x06b.stages'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ))
    finish(out,meta,{'status':'dispatched','pid':child.pid,'gpu_seconds':0});print(child.pid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');launch() if p.parse_args().launch else run()
