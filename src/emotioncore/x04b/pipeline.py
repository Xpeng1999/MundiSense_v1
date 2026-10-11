"""Sequential Q_mix launcher with immutable gates and charged retry wall time."""
import argparse,datetime as dt,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03.q_manifest import sha256
from emotioncore.x04.prepare import storage_gate
from emotioncore.x03.q_pipeline import assert_e2_cpu
LIMIT=70.
ET_CACHE=Path('/mnt/h/emotionAI/emotioncore_models/X03/q_inputs')

def read(p):return json.loads(p.read_text())
def append(p,value):
    with p.open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')

def projection(args,progress,completed):
    budget=read(args.derived/'preflight/budget_preflight.json');done={s['name'] for s in progress['steps'] if s['returncode']==0};ratios=[1.]
    for n in ('fold0','fold1','final'):
        if 'train_'+n in done:
            actual=sum(s['elapsed_s'] for s in progress['steps'] if s['name']=='train_'+n)/3600
            ratios.append(actual/budget['runs'][n]['training_validation_h'])
    ratio=max(ratios);remaining_train=sum(v['training_validation_h'] for n,v in budget['runs'].items() if 'train_'+n not in done)*ratio
    remaining_pred=sum(v['prediction_h'] for n,v in budget['runs'].items() if ('predict_'+n if n!='final' else 'predict_final_mer') not in done)*ratio
    # Prediction reserve for final remains whole until all three target calls finish.
    actual=progress['charged_gpu_hours'];diagnostic=max(0.,1.-sum(s['elapsed_s'] for s in progress['steps'] if s['name']=='precheck')/3600)
    total=actual+(remaining_train+remaining_pred)*1.2+.5+diagnostic
    receipt={'completed_model':completed,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'charged_gpu_hours':actual,'observed_ratio_vs_initial':ratio,'remaining_training_h':remaining_train,'remaining_prediction_h':remaining_pred,'reserve_fraction':.2,'diagnostic_reserve_h':diagnostic,'load_reserve_h':.5,'projected_total_gpu_h':total,'limit_h':70,'passed':total<=70}
    append(args.root/'q_budget_progress.jsonl',receipt);progress['budget_projection']=receipt;write_json_atomic(args.root/'pipeline_progress.json',progress)
    if total>70:raise RuntimeError('STOP projected total exceeds70h before next model')
    return receipt

def step(args,progress,name,command,gpu=True,training=False):
    previous=[s for s in progress['steps'] if s['name']==name and s['returncode']==0]
    if previous:return
    env=os.environ.copy();env['PYTHONPATH']=str(Path.cwd()/'src');env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTORCH_CUDA_ALLOC_CONF',None);env.pop('PYTORCH_ALLOC_CONF',None)
    if not gpu:env['CUDA_VISIBLE_DEVICES']=''
    output=Path(command[command.index('--output-dir')+1]) if '--output-dir' in command else None
    attempts=3 if training else 1
    logdir=args.root/'logs';logdir.mkdir(exist_ok=True)
    for attempt in range(attempts):
        storage_gate()
        if gpu:assert_e2_cpu()
        if training and (output/'terminal_stop.json').exists():raise RuntimeError('STOP training terminal gate')
        active=command+(['--resume-last'] if training and (output/'last.json').exists() else [])
        if output and (output/'run_meta.json').exists():
            dest=args.root/'attempt_meta';dest.mkdir(exist_ok=True);write_json_atomic(dest/f'{name}_{len(progress["steps"])}.json',read(output/'run_meta.json'))
        started=time.monotonic();log=logdir/f'{name}_attempt{attempt}.log'
        progress.update(active_step=name,active_started_at=dt.datetime.now(dt.timezone.utc).isoformat(),active_command=active,active_attempt=attempt);write_json_atomic(args.root/'pipeline_progress.json',progress)
        with log.open('w') as stream:
            process=subprocess.Popen(active,cwd=Path.cwd(),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            progress['child_pid']=process.pid;write_json_atomic(args.root/'pipeline_progress.json',progress)
            try:
                while process.poll() is None:
                    if gpu and progress['charged_gpu_hours']+(time.monotonic()-started)/3600>LIMIT:raise RuntimeError('STOP actual GPU70h')
                    storage_gate();time.sleep(10)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                elapsed=time.monotonic()-started
                if gpu:progress['charged_gpu_hours']+=elapsed/3600
                progress['steps'].append({'name':name,'returncode':process.returncode,'elapsed_s':elapsed,'gpu':gpu,'interrupted':True});write_json_atomic(args.root/'pipeline_progress.json',progress);raise
        elapsed=time.monotonic()-started
        if gpu:progress['charged_gpu_hours']+=elapsed/3600
        progress['steps'].append({'name':name,'attempt':attempt,'returncode':process.returncode,'elapsed_s':elapsed,'gpu':gpu,'log':str(log),'log_sha256':sha256(log)})
        progress['active_step']=None;progress.pop('child_pid',None);write_json_atomic(args.root/'pipeline_progress.json',progress)
        if process.returncode==0:return
        append(args.root/'incidents.jsonl',{'step':name,'attempt':attempt,'exit':process.returncode,'at':dt.datetime.now(dt.timezone.utc).isoformat(),'last_exists':bool(output and (output/'last.json').exists())})
        if not training or not (output/'last.json').exists() or (output/'terminal_stop.json').exists():break
        if progress['charged_gpu_hours']>=70:break
    raise RuntimeError(f'{name} failed; retained original receipts, retries limited2')

def run(args):
    resource.setrlimit(resource.RLIMIT_NOFILE,(65536,65536));git=git_provenance(Path.cwd());storage_gate();args.root.mkdir(parents=True,exist_ok=True)
    if (args.root/'pipeline_stop.json').exists():raise RuntimeError('STOP existing terminal pipeline receipt; needs review')
    if (args.root/'pipeline_progress.json').exists():
        progress=read(args.root/'pipeline_progress.json')
        if progress.get('child_pid') and Path('/proc',str(progress['child_pid'])).exists():raise RuntimeError('Existing child process; do not duplicate')
    else:progress={'git':git,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'steps':[],'charged_gpu_hours':0.,'status':'running'}
    write_json_atomic(args.root/'run_meta.json',{'git':git,'command':sys.argv,'started_at':progress['started_at'],'gpu_limit_hours':LIMIT,'cuda_allocator':'default','ulimit_n':65536})
    manifest=read(args.derived/'manifests/run_meta.json');budget=read(args.derived/'preflight/budget_preflight.json')
    if not budget['passed'] or budget['total_projection_gpu_hours']>70:raise RuntimeError('STOP preflight70h gate')
    def cmd(module,*values):return [sys.executable,'-u','-m',module,*map(str,values)]
    root=args.derived/'Q';root.mkdir(exist_ok=True)
    def predict(name,run_name,meta,trained):
        step(args,progress,name,cmd('emotioncore.x03.q_predict','--run-name',run_name,'--manifest-meta',meta,'--cache-root',ET_CACHE,'--adapter',trained['best_checkpoint'],'--adapter-sha256',trained['best_checkpoint_sha256'],'--output-dir',root/('predict_'+run_name)))
    try:
        projection(args,progress,'preflight')
        step(args,progress,'precheck',cmd('emotioncore.x04b.precheck','--root',args.derived,'--output',args.derived/'precheck'))
        probe=read(args.derived/'precheck/summary.json')
        if probe['gate']!='pass' or probe['tested']!=20 or probe['peak_allocated_bytes']>30000000000:raise RuntimeError('STOP longest20 memory gate')
        for name in ('fold0','fold1','final'):
            if name=='final':
                step(args,progress,'futility_analysis',cmd('emotioncore.x04b.analysis','--root',root,'--output',args.derived/'interim','--interim'),gpu=False)
                futility=read(args.derived/'interim/futility.json')
                if futility['stop']:raise RuntimeError('STOP fold01 Q_AV futility point < -0.02')
            spec=manifest['runs'][name]
            step(args,progress,'train_'+name,cmd('emotioncore.x04b.train','--fold-name',name,'--manifest-meta',args.derived/'manifests/run_meta.json','--val-manifest',spec['val_path'],'--cache-root',ET_CACHE,'--output-dir',root/name),training=True)
            trained=read(root/name/'summary.json')
            if name=='fold0' and trained['best_av_macro_f1']<.45:raise RuntimeError('STOP fold0 AV macro-F1 below0.45')
            projection(args,progress,name)
            if name!='final':predict('predict_'+name,name,args.r3/'inference_manifests/run_meta.json',trained)
            else:
                predict('predict_final_et','final_et_test',args.r3/'inference_manifests/run_meta.json',trained)
                predict('predict_final_sims','final_sims_valid',args.r3/'sims_valid_manifests/run_meta.json',trained)
                step(args,progress,'predict_final_mer',cmd('emotioncore.x04b.predict_mer','--adapter',trained['best_checkpoint'],'--adapter-sha256',trained['best_checkpoint_sha256'],'--output-dir',root/'predict_mer_test'))
            projection(args,progress,name+'_predictions')
        step(args,progress,'final_analysis',cmd('emotioncore.x04b.analysis','--root',root,'--output',args.derived/'analysis'),gpu=False)
        progress.update(status='complete',ended_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json_atomic(args.root/'pipeline_progress.json',progress)
        meta=read(args.root/'run_meta.json');meta['ended_at']=progress['ended_at'];meta['pipeline_sha256']=sha256(args.root/'pipeline_progress.json');write_json_atomic(args.root/'run_meta.json',meta)
    except BaseException as exc:
        progress.update(status='stopped',reason=f'{type(exc).__name__}: {exc}',stopped_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json_atomic(args.root/'pipeline_stop.json',progress);write_json_atomic(args.root/'pipeline_progress.json',progress);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);p.add_argument('--r3',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X03/Q'));run(p.parse_args())
if __name__=='__main__':main()
