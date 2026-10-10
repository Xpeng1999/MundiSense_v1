"""Reviewer-authorized final-only continuation; original stop receipts stay intact."""
import argparse,copy,datetime as dt,fcntl,json,os,resource,sys
from pathlib import Path
from emotioncore.x04b.pipeline import read,step,projection,ET_CACHE,LIMIT
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03.q_manifest import sha256
from emotioncore.x04.prepare import storage_gate

def run(args):
    resource.setrlimit(resource.RLIMIT_NOFILE,(65536,65536))
    git=git_provenance(Path.cwd());storage_gate();args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'pipeline_progress.json').exists():raise RuntimeError('Continuation already launched; inspect receipts before any restart')
    old=read(args.original/'pipeline_stop.json')
    if old['reason']!='RuntimeError: STOP fold01 Q_AV futility point < -0.02':raise ValueError('Unexpected original stop')
    if old.get('child_pid') and Path('/proc',str(old['child_pid'])).exists():raise RuntimeError('Original child still active')
    done={x['name'] for x in old['steps'] if x['returncode']==0}
    if not {'train_fold0','train_fold1','predict_fold0','predict_fold1','futility_analysis'}<=done:raise ValueError('Missing frozen predecessors')
    if (args.derived/'Q/final').exists():raise RuntimeError('Final already exists; do not duplicate')
    manifest=read(args.derived/'manifests/run_meta.json');spec=manifest['runs']['final']
    for e in spec['epochs']:
        if sha256(Path(e['path']))!=e['sha256']:raise ValueError('Frozen final epoch changed')
    if sha256(Path(spec['val_path']))!=spec['val_sha256']:raise ValueError('Frozen final validation changed')
    progress=copy.deepcopy(old)
    for k in ('reason','stopped_at','child_pid','active_command','active_started_at','active_attempt'):progress.pop(k,None)
    progress.update(git=git,status='running',active_step=None,started_at=dt.datetime.now(dt.timezone.utc).isoformat(),original_stop_sha256=sha256(args.original/'pipeline_stop.json'),revision='X04B_CONTINUE_PROMPT 2026-10-10; final only; futility retired')
    write_json_atomic(args.root/'run_meta.json',{'git':git,'command':sys.argv,'started_at':progress['started_at'],'gpu_limit_hours':LIMIT,'prior_gpu_hours':old['charged_gpu_hours'],'original_stop_sha256':progress['original_stop_sha256'],'cuda_allocator':'default','ulimit_n':65536})
    def cmd(module,*values):return [sys.executable,'-u','-m',module,*map(str,values)]
    q=args.derived/'Q'
    try:
        receipt=projection(args,progress,'reviewer_continuation_preflight');write_json_atomic(args.root/'budget_preflight.json',receipt)
        step(args,progress,'revised_p1',cmd('emotioncore.x04b.analysis','--root',q,'--output',args.derived/'revised_p1','--interim','--within-fold-loso'),gpu=False)
        step(args,progress,'train_final',cmd('emotioncore.x04b.train','--fold-name','final','--manifest-meta',args.derived/'manifests/run_meta.json','--val-manifest',spec['val_path'],'--cache-root',ET_CACHE,'--output-dir',q/'final'),training=True)
        trained=read(q/'final/summary.json');projection(args,progress,'final')
        for split,meta in [('et_test',args.r3/'inference_manifests/run_meta.json'),('sims_valid',args.r3/'sims_valid_manifests/run_meta.json')]:
            step(args,progress,'predict_final_'+split,cmd('emotioncore.x03.q_predict','--run-name','final_'+split,'--manifest-meta',meta,'--cache-root',ET_CACHE,'--adapter',trained['best_checkpoint'],'--adapter-sha256',trained['best_checkpoint_sha256'],'--output-dir',q/('predict_final_'+split)))
        step(args,progress,'predict_final_mer',cmd('emotioncore.x04b.predict_mer','--adapter',trained['best_checkpoint'],'--adapter-sha256',trained['best_checkpoint_sha256'],'--output-dir',q/'predict_mer_test'))
        projection(args,progress,'final_predictions')
        step(args,progress,'revised_final_analysis',cmd('emotioncore.x04b.analysis','--root',q,'--output',args.derived/'analysis_revised','--within-fold-loso'),gpu=False)
        progress.update(status='complete',ended_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json_atomic(args.root/'pipeline_progress.json',progress)
    except BaseException as exc:
        progress.update(status='stopped',reason=f'{type(exc).__name__}: {exc}',stopped_at=dt.datetime.now(dt.timezone.utc).isoformat());write_json_atomic(args.root/'pipeline_progress.json',progress);write_json_atomic(args.root/'pipeline_stop.json',progress);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--derived',type=Path,required=True);p.add_argument('--r3',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X03/Q'));run(p.parse_args())
if __name__=='__main__':main()
