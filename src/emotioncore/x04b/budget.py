"""Budget exact frozen epoch selections, retaining X04a conservative timings."""
import argparse,datetime as dt,json,sys
from pathlib import Path
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl,sha256
from emotioncore.x04.budget import conservative_curve
from emotioncore.x04.prepare import storage_gate

def run(args):
    git=git_provenance(Path.cwd());storage_gate();meta=json.loads((args.manifests/'run_meta.json').read_text())
    benchpath=args.r3/'gc_step_benchmark/summary.json';bench=json.loads(benchpath.read_text());oldpath=args.r4/'budget_estimate/x04b_budget.json';old=json.loads(oldpath.read_text());ratio=old['observed_actual_to_benchmark_ratio'];runs={}
    for name,spec in meta['runs'].items():
        cost=0.;details=[]
        for epoch in spec['epochs']:
            p=Path(epoch['path'])
            if sha256(p)!=epoch['sha256']:raise ValueError('Manifest changed')
            rows=read_jsonl(p);mer_s=0.;et_s=0.;extra=0
            for i,row in enumerate(rows):
                if i%200==0:storage_gate()
                f=row['effective_form']
                if row['corpus']=='ET':et_s+=bench['forms'][f]['train_step_mean_s']
                else:
                    rec=json.loads((Path(row['cache_root'])/row['utt_id']/(f+'.json')).read_text())
                    if rec['status']!='ok' or rec['input_wav_sha256']!=row['wav_sha256'] or rec.get('input_video_sha256')!=row.get('video_sha256'):raise ValueError('Cache receipt mismatch')
                    seconds,detail=conservative_curve(bench['forms'][f]['samples'],[rec['shapes']['input_ids'][-1]],'train_step_s');mer_s+=float(seconds[0]);extra+=detail['extrapolated_count']
            cost+=et_s+mer_s;details.append({'epoch':len(details),'ET_training_s':et_s,'MER_training_s':mer_s,'MER_extrapolated':extra,'manifest_sha256':epoch['sha256']})
        val=read_jsonl(Path(spec['val_path']));validation=8*sum(bench['forms'][r['effective_form']]['inference_mean_s'] for r in val)
        prediction=old['runs'][name]['prediction_s_before_observed_ratio']
        runs[name]={'training_validation_h':(cost+validation)*ratio/3600,'prediction_h':prediction*ratio/3600,'epochs':details,'epoch_n':spec['epoch_n'],'validation_n':700,'validation_calls':8}
    raw=sum(v['training_validation_h']+v['prediction_h'] for v in runs.values());total=raw*1.2+.5+1
    result={'runs':runs,'gpu_limit_hours':70,'reserve_fraction':.2,'load_checkpoint_reserve_h':.5,'diagnostic_reserve_h':1,'total_projection_gpu_hours':total,'passed':total<=70,'observed_ratio':ratio,'method':'X03 ET form means; exact frozen MER50% class/form selections with monotone upper envelope and quadratic extrapolation; observed checkpointed actual/benchmark ratio','sources':{'benchmark':{'path':str(benchpath),'sha256':sha256(benchpath)},'X04a_budget':{'path':str(oldpath),'sha256':sha256(oldpath)},'manifests':{'path':str(args.manifests/'run_meta.json'),'sha256':sha256(args.manifests/'run_meta.json')}},'limitation':'No new Q training measurements; MER masked first-token loss overhead unmeasured and covered by 20% reserve. Recompute from every completed model.'}
    args.output.mkdir(parents=True,exist_ok=True);write_json_atomic(args.output/'budget_preflight.json',result);write_json_atomic(args.output/'run_meta.json',{'git':git,'command':sys.argv,'ended_at':dt.datetime.now(dt.timezone.utc).isoformat(),'learned_inference':False,'sha256':sha256(args.output/'budget_preflight.json')})
    if not result['passed']:raise RuntimeError('STOP budget exceeds70h')

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifests',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--r3',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X03/Q'));p.add_argument('--r4',type=Path,default=Path('/mnt/h/emotionAI/emotioncore_runs/X04a'));run(p.parse_args())
if __name__=='__main__':main()
