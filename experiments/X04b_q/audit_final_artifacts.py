"""Read frozen artifacts and emit aggregate-only final delivery checks."""
from pathlib import Path
import collections, datetime, hashlib, json, math
import numpy as np

H=Path('/mnt/h/emotionAI/emotioncore_runs/X04b')
R=Path('/home/administrator/emotion-data/emotioncore_runs/X04b')
R3=Path('/home/administrator/emotion-data/emotioncore_runs/X03/Q')
H4=Path('/mnt/h/emotionAI/emotioncore_runs/X04a')
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,value):
    value['passed']=True;value['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (H/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
q=H/'Q/final';m=read(q/'run_meta.json');s=read(q/'summary.json')
assert m['summary_sha256']==sha(q/'summary.json')
assert not m['git']['dirty'] and m['git']['commit'].startswith('ac2aab8')
assert m['cuda_allocator']=='default' and m['initialization']=='base only; never X03 adapter'
assert m['gradient_checkpointing']==dict(llm=True,vision=True,audio=True,use_reentrant=False)
manifest=[]
for e in m['epochs']:
    assert sha(e['path'])==e['sha256'];rr=rows(e['path'])
    assert len(rr)==17793 and collections.Counter(r['corpus'] for r in rr)=={'ET':13100,'MER':4693}
    assert len({r['utt_id'] for r in rr})==len(rr)
    manifest+=rr
assert hashlib.sha256(''.join(e['sha256'] for e in m['epochs']).encode()).hexdigest()==m['combined_epoch_sha256']
assert len(manifest)==s['trained_examples']==s['examples_seen']==35586
receipts=rows(q/'train_modality_receipts.jsonl')
assert len(receipts)==len(manifest) and sha(q/'train_modality_receipts.jsonl')==s['train_modality_receipts_sha256']
keys=['utt_id','epoch','corpus','assigned_form','effective_form','face_missing']
assert all(all(a[k]==b[k] for k in keys) for a,b in zip(manifest,receipts))
assert all(r['forward_has_audio']==(r['effective_form'] in ['A','AV']) and r['forward_has_video']==(r['effective_form'] in ['V','AV']) for r in receipts)
assert collections.Counter(r['effective_form'] for r in receipts)==s['modality_forward_counts']
assert sum(r['face_missing'] and r['assigned_form']!='A' for r in manifest)==s['face_missing_fallback_train']==140
val=rows(R3/'manifests/final_val.jsonl');valids={r['utt_id'] for r in val}
assert len(valids)==700 and sha(R3/'manifests/final_val.jsonl')==m['val_manifest_sha256']
assert not valids.intersection(r['utt_id'] for r in manifest)
assert len(s['evals'])==8
for e in s['evals']:
    p=q/'validation'/('eval'+str(e['eval_index'])+'.jsonl');rr=rows(p)
    assert sha(p)==e['predictions_sha256']
    assert len(rr)==len({r.get('utt_id',r.get('id')) for r in rr})==e['valid']==e['n']==700
    assert {r.get('utt_id',r.get('id')) for r in rr}==valids
    assert e['failures']==0 and e['coverage']==1
assert s['best_av_macro_f1']==max(e['macro_f1'] for e in s['evals'])
assert s['best_eval_index']==8
assert sha(Path(s['best_checkpoint'])/'adapter_model.safetensors')==s['best_checkpoint_sha256']
last=read(q/'last.json');assert last['position']==35586
assert sha(Path(last['path'])/'adapter/adapter_model.safetensors')==last['adapter_sha256']
state_files=list(Path(last['path']).glob('*.pt'))
assert len(state_files)==1 and sha(state_files[0])==last['state_sha256']
assert last['train_manifest_sha256']==m['train_manifest_sha256'] and last['val_manifest_sha256']==m['val_manifest_sha256']
assert s['skipped_oom']==0 and (q/'skipped_oom.jsonl').stat().st_size==0
assert sha(q/'skipped_oom.jsonl')==s['skipped_oom_sha256']
monitor=rows(q/'memory_monitor.jsonl');assert [r['optimizer_step'] for r in monitor]==list(range(50,4450,50))
assert all(r['max_memory_allocated_bytes']<=30_000_000_000 for r in monitor)
save('final_completion_verification',dict(ordered_manifest_receipts_equal=True,count=len(receipts),corpus_counts=dict(collections.Counter(r['corpus'] for r in receipts)),validation_count=8,all_validation_coverage=1.,summary_sha256=sha(q/'summary.json'),run_meta_sha256=sha(q/'run_meta.json'),best_adapter_sha256=s['best_checkpoint_sha256'],last=last,no_oom_skips=True,monitor_records=len(monitor),peak_cuda_allocated_mib=s['peak_cuda_allocated_mib']))

pred={}
for split,name,n,forms,meta_path,prefix in [('et_test','predict_final_et_test',1929,['A','V','AV'],R3/'inference_manifests/run_meta.json','final_et_test_'),('sims_valid','predict_final_sims_valid',647,['A','V','AV'],R3/'sims_valid_manifests/run_meta.json','final_sims_valid_'),('mer_test','predict_mer_test',572,['A','AV'],H4/'Q_media/summary.json','test_')]:
    root=H/'Q'/name;summary=read(root/'summary.json');meta=read(root/'run_meta.json');specs=read(meta_path)['outputs']
    assert meta['ended_at'] and meta['adapter_sha256']==s['best_checkpoint_sha256'] and not meta['git']['dirty']
    assert meta['summary_sha256']==sha(root/'summary.json')
    item={'n':n,'summary_sha256':sha(root/'summary.json'),'run_meta_sha256':sha(root/'run_meta.json'),'forms':{}}
    for f in forms:
        spec=summary[f];rr=rows(spec['path']);inp=specs[prefix+f];mm=rows(inp['path'])
        assert sha(spec['path'])==spec['sha256'] and sha(inp['path'])==inp['sha256']
        assert len(rr)==spec['n']==len({r['utt_id'] for r in rr})==n
        assert [r['utt_id'] for r in rr]==[r['utt_id'] for r in mm]
        assert all(r['effective_form']==t['effective_form'] and r['form']==f for r,t in zip(rr,mm))
        for r,t in zip(rr,mm):
            assert all(r[k]==t[k] for k in (['label'] if split=='sims_valid' else ['truth_M']))
            if split!='mer_test':
                assert r['forward_has_audio']==(r['effective_form'] in ['A','AV']) and r['forward_has_video']==(r['effective_form'] in ['V','AV'])
            prob=r.get('probabilities8',r['probabilities']);assert all(math.isfinite(v) and v>=0 for v in prob) and abs(sum(prob)-1)<1e-5
        item['forms'][f]={'n':n,'unique_ids':n,'ordered_manifest_ids_equal':True,'sha256':spec['sha256'],'path':spec['path'],'uniform':sum(r['effective_form']=='uniform' for r in rr),'face_missing_fallback':sum(r['face_missing_fallback'] for r in rr),'statuses':dict(collections.Counter(r['status'] for r in rr))}
    pred[split]=item
save('final_prediction_verification',pred)

a=H/'analysis_revised';am=read(a/'run_meta.json');main=read(a/'main_table.json');prior=read(H/'revised_p1/main_table.json');primary=read(a/'primary.json')
assert am['ended_at'] and not am['git']['dirty'] and am['p1_method']=='within_fold_loso' and not am['target_labels_used_for_fitting']
assert am['folds']==[0,1] and am['seed']==20261007
for f,digest in am['outputs'].items():assert sha(a/f)==digest
for path,spec in read(a/'source_artifact_index.json').items():assert sha(path)==spec['sha256']
cohorts={'oof':6533,'et_test':1929,'sims_valid':647,'mer_test':572}
for key,item in main.items():
    assert item['parameters_final']==prior[key]['parameters_final'] and item['parameters_oof']==prior[key]['parameters_oof'] and item['cv']==prior[key]['cv']
    assert len(item['parameters_oof'])==6
    for p in item['parameters_oof']:
        assert p['held_speaker'] not in p['fit_speakers'] and len(p['fit_speakers'])==2
    for split,spec in item['artifacts'].items():
        rr=rows(spec['path']);assert sha(spec['path'])==spec['sha256'] and len(rr)==len({r['utt_id'] for r in rr})==cohorts[split]
for f in ['secondary.json','raw_comparisons.json']:
    assert read(a/f)==read(H/'revised_p1'/f)
assert primary['H_ref']==read(H/'revised_p1/primary.json')['H_ref']
p2checks={}
for system in ['H_ref','Q_AV']:
    left=rows(a/'Q_mix'/system/'sims_valid.jsonl');right=rows(a/'Q_et'/system/'sims_valid.jsonl')
    assert [r['utt_id'] for r in left]==[r['utt_id'] for r in right]
    assert [r['label'] for r in left]==[r['label'] for r in right]
    pairs=[(l,r) for l,r in zip(left,right) if not l['forced_error'] and not r['forced_error']]
    x=np.array([l['op_n'][1]-sum(l['op_n'][2:6]) for l,r in pairs]);z=np.array([r['op_n'][1]-sum(r['op_n'][2:6]) for l,r in pairs]);y=np.array([l['label'] for l,r in pairs])
    def corr(u,v):
        u=u-u.mean();v=v-v.mean();return float(np.dot(u,v)/np.sqrt(np.dot(u,u)*np.dot(v,v)))
    rng=np.random.default_rng(20261007);boot=[]
    for i in range(2000):
        ix=rng.integers(len(pairs),size=len(pairs));boot.append(corr(x[ix],y[ix])-corr(z[ix],y[ix]))
    got=primary['P2'][system]
    assert got['n']==620 and got['excluded_joint_failures']==27 and got['replicates']==2000 and got['seed']==20261007
    assert abs(got['difference']-(corr(x,y)-corr(z,y)))<1e-12
    assert np.allclose(got['bootstrap_differences'],boot,atol=1e-12,rtol=0)
    assert np.allclose(got['difference_95'],np.percentile(boot,[2.5,97.5]),atol=1e-12,rtol=0)
    p2checks[system]={k:v for k,v in got.items() if k!='bootstrap_differences'}
progress=read(R/'continuation/pipeline_progress.json');assert progress['status']=='complete' and not (R/'continuation/pipeline_stop.json').exists()
assert sha(R/'pipeline_stop.json')==progress['original_stop_sha256']
gpu=sum(step['elapsed_s']/3600 for step in progress['steps'] if step['gpu'])
assert abs(gpu-progress['charged_gpu_hours'])<1e-10 and gpu<70
assert all(x['passed'] and x['projected_total_gpu_h']<70 for x in rows(R/'continuation/q_budget_progress.jsonl'))
assert all(s['attempt']==0 and s['returncode']==0 for s in progress['steps'])
save('final_analysis_verification',dict(analysis_sha256=sha(a/'run_meta.json'),analysis_outputs_and_source_sha_verified=True,revised_p1_and_target_parameters_unchanged=True,all_8_systems_all_4_cohorts_exact_coverage=True,independent_p2_bootstrap_equal=True,p2=p2checks,original_stop_unchanged=True,all_steps_first_attempt=True,charged_gpu_hours=gpu,budget_all_passed=True))
print(json.dumps({'passed':True,'p2':p2checks,'charged_gpu_hours':gpu},ensure_ascii=False))
