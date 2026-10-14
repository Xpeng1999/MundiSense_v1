"""Read-only delivery audit; copies existing metrics and never reads label columns."""
import datetime as dt
import json, shutil
from pathlib import Path
from emotioncore.x07.common import read,sha,spec,committed_protocol,load_completed,git,ROOT,PROTOCOL,now,write_new
p,registration=committed_protocol()
assert not git('status','--porcelain','--untracked-files=normal')
code_checks={k:sha(k)==v for k,v in p['code_sha256'].items()}
config_checks={k:sha(v['path'])==v['sha256'] for k,v in p['configs'].items()}
assert all(code_checks.values()) and all(config_checks.values())
asset_checks={name:all(sha(item['path'])==item['sha256'] for item in files) for name,files in p['inventory']['assets'].items()}
assert all(asset_checks.values())
dependency=read(ROOT/'dependency_recovery.json')
assert all(sha(item['v2_copy']['path'])==item['accepted_copy']['sha256'] for item in dependency['files'])
assert sha('configs/x07_lockbox_protocol.json')==p['revision']['supersedes']['protocol']['sha256']
assert sha(p['inventory']['meta_csv']['path'])==p['inventory']['meta_csv']['sha256']
leak=read(ROOT/'leakage.json')
audit={'at':now(),'root':str(ROOT),'protocol':spec(PROTOCOL),'registration':registration,
       'execution_git':git('rev-parse','HEAD'),'git_clean':True,'code_checks':code_checks,
       'config_checks':config_checks,'asset_checks':asset_checks,'dependency_files_unchanged':len(dependency['files']),'collector':spec(ROOT/'collect_delivery_1.py'),'meta_sha_unchanged':True,'labels_read_by_audit':False,
       'metrics_recomputed':False,'root_free_bytes':shutil.disk_usage('/').free,'splits':{}}
for split in ('valid','test'):
    d=ROOT/split; record={'directory_exists':d.exists()}
    for name in ('inputs.json','inference.started.json','complete.json','metrics.started.json','metrics.json'):
        if (d/name).exists():record[name]=spec(d/name)
    if (d/'inputs.json').exists():
        prepared=read(d/'inputs.json');failures=[];inputs=[]
        for uid,item in prepared['inputs'].items():
            assert sha(item['path'])==item['sha256']
            row=read(item['path']);assert row['id']==uid
            inputs.append({'id':uid,**item})
            if row['status']!='ok':failures.append({'id':uid,'reason':row['failure'],'input':item})
        record.update(input_count=len(inputs),input_receipts=inputs,failures=failures,
                      preparation_started_at=prepared['started_at'],preparation_ended_at=prepared['ended_at'])
    if (d/'complete.json').exists():
        done,rows=load_completed(d,p['inventory']['splits'][split]['ids'])
        assert {f.stem for f in (d/'predictions').glob('*.json')}==set(p['inventory']['splits'][split]['ids'])
        assert all(row['protocol_sha256']==registration['sha256'] for row in rows)
        record.update(predictions_count=len(rows),first_inference_at=done['first_inference_at'],
                      completed_at=done['completed_at'],peak_gpu_bytes=done['peak_gpu_bytes'],
                      resume_receipts=[spec(f) for f in sorted((d/'resumes').glob('*.json'))])
        if (d/'metrics.json').exists():
            metric=read(d/'metrics.json'); lock=read(d/'metrics.started.json')
            assert metric['identity_check']==leak['identity_checks'][split]
            assert dt.datetime.fromisoformat(lock['at'])>dt.datetime.fromisoformat(done['completed_at'])
            assert metric['inference_completion']==spec(d/'complete.json')
            record.update(identity_receipt_matches_L2=True,metric_lock_after_all_predictions=True,
                          metric_lock_at=lock['at'],metrics_started_at=metric['started_at'],metrics_ended_at=metric['ended_at'])
    else:
        assert not (d/'metrics.started.json').exists() and not (d/'metrics.json').exists()
    audit['splits'][split]=record
if audit['splits']['test']['directory_exists']:
    valid=read(ROOT/'valid/metrics.json')
    assert valid['L3_pass'] and valid['systems']['H3']['pearson_available']>=.78
    assert dt.datetime.fromisoformat(audit['splits']['test']['preparation_started_at'])>dt.datetime.fromisoformat(valid['ended_at'])
    audit['L3_passed_before_any_test_preparation']=True
jobs=[read(f) for f in sorted((ROOT/'gpu_jobs').glob('*/receipt.json'))]
assert all(j['status']!='running' for j in jobs)
audit['gpu_jobs']=jobs;audit['gpu_seconds']=sum(j['charged_seconds'] for j in jobs)
assert audit['gpu_seconds']<=14400
v1=Path('/mnt/h/emotionAI/emotioncore_runs/X07')
old=read(ROOT/'v1_preservation_before.json')
current={str(f.relative_to(v1)):{'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in v1.rglob('*') if f.is_file()}
assert current==old['files']
assert spec(v1/'STOP.json')==old['stop'] and spec(v1/'code/configs/x07_lockbox_protocol.json')==old['protocol']
assert not (v1/'test').exists()
audit['v1_preservation']={'all_file_sizes_and_mtimes_unchanged':True,'file_count':len(current),
    'STOP_unchanged':True,'protocol_unchanged':True,'test_directory_exists':False,
    'full_before_snapshot':spec(ROOT/'v1_preservation_before.json')}
expected_native={'aqgy4_0001__00033','aqgy4_0016__00007','aqgy4_0018__00020','aqgy5_0005__00033','aqgy5_0017__00027','aqgy5_0018__00007','aqgy5_0022__00023'}
actual_native={f['id'] for f in audit['splits']['valid'].get('failures',[]) if f['reason']=='STOP native decode failed'}
audit['original_seven_native_decode_failures_preserved']={'expected':sorted(expected_native),'actual':sorted(actual_native),'match':actual_native==expected_native}
# Preserve the original failure records and expose v2 IDs for reviewer comparison.
audit['v1_failure_audit']=spec('experiments/X07_lockbox/results/preparation_audit.json')
write_new(ROOT/'delivery_audit.json',audit)
print(json.dumps({'audit':spec(ROOT/'delivery_audit.json'),'gpu_seconds':audit['gpu_seconds'],
    'v1_preserved_file_count':len(current),'splits':{k:{x:v.get(x) for x in ('directory_exists','input_count','predictions_count','identity_receipt_matches_L2')} for k,v in audit['splits'].items()}}))
