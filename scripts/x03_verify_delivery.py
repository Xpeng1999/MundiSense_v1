"""Read-only audit of X03's final server artifacts, coverage and frozen gates."""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
import numpy as np
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl, sha256


def run(root):
    git=git_provenance(Path.cwd())
    frozen=json.loads(Path('configs/folds_v1.json').read_text())
    manifest=read_jsonl(Path('experiments/X01_benchmark/results/et_utterances.jsonl'))
    cv={r['utt_id'] for r in manifest if r['speaker_id'] in frozen['cv_pool_speakers']}
    et={r['utt_id'] for r in manifest if r['official_split']=='test'}
    assert len(cv)==16366 and len(et)==1929 and not cv&et
    expected_fold={s:f['fold'] for f in frozen['folds'] for s in f['outer_holdout_speakers']}
    face=set(json.loads(Path('configs/x03_face_missing_policy.json').read_text())['face_missing_utt_ids'])
    assert len(face&cv)==40 and len(face&et)==3
    for config,expected in [('folds_v1.json','85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce'),
        ('x03_inner_val_subsets.json','be2cd442346cfa978e22bea8ab2ae3776d52187b2581fe93e377fc4fb58d3cfd'),
        ('x03_face_missing_policy.json','ce65415b5f7129b62ef5edc7eda5b20874ed88931f5a7cc295c276b8b704b77c')]:
        assert sha256(Path('configs')/config)==expected
    adapters={};training={};raw={}
    for name in ['fold'+str(i) for i in range(5)]+['final']:
        summary=json.loads((root/f'Q/{name}/summary.json').read_text())
        checkpoint=Path(summary['best_checkpoint'])/'adapter_model.safetensors'
        assert sha256(checkpoint)==summary['best_checkpoint_sha256']
        assert len(summary['evals'])==8 and all(e['n']==700 and e['failures']==0 for e in summary['evals'])
        assert summary.get('skipped_oom',0)==0
        if name!='fold0':
            meta=json.loads((root/f'Q/{name}/run_meta.json').read_text())
            assert meta['cuda_allocator']=='default'
            assert all(meta['gradient_checkpointing'][n] for n in ('llm','vision','audio'))
        train=read_jsonl(root/f'Q/manifests/{name}_train.jsonl')
        allowed=set(frozen['folds'][int(name[-1]) if name!='final' else 0]['inner_train_speakers'])
        if name=='final':allowed=set(frozen['cv_pool_speakers'])-set(frozen['folds'][0]['inner_validation_speakers'])
        assert all(r['speaker_id'] in allowed for r in train)
        adapters[name]={'path':str(checkpoint),'sha256':sha256(checkpoint)}
        training[name]={'validation_events':8,'validation_n':700,'skipped_oom':summary.get('skipped_oom',0),
                        'train_speakers':sorted({r['speaker_id'] for r in train})}
    for form in ('A','V','AV'):
        rows=[r for i in range(5) for r in read_jsonl(root/f'Q/predict_fold{i}/{form}.jsonl')]
        assert len(rows)==16366 and {r['utt_id'] for r in rows}==cv
        assert all(r['fold']==expected_fold[r['speaker_id']] for r in rows)
        for r in rows:
            p=np.array(r['probabilities']);assert p.shape==(7,) and np.isfinite(p).all() and np.isclose(p.sum(),1)
            effective=r['effective_form']
            assert r['forward_has_audio']==(effective in ('A','AV'))
            assert r['forward_has_video']==(effective in ('V','AV'))
            if form=='V' and r['utt_id'] in face:assert r['pred'] is None and np.allclose(p,1/7)
            if form=='AV' and r['utt_id'] in face:assert effective=='A' and r['face_missing_fallback']
        raw[form]={'cv_n':len(rows),'unique_ids':len({r['utt_id'] for r in rows})}
        for split,expected in [('et_test',1929),('sims_valid',647)]:
            p=root/f'Q/predict_final_{split}/{form}.jsonl';target=read_jsonl(p)
            assert len(target)==expected and len({r['utt_id'] for r in target})==expected
            assert split!='et_test' or {r['utt_id'] for r in target}==et
            raw[form][split]={'n':len(target),'path':str(p),'sha256':sha256(p)}
    roots=[p.parent for parent in ('E1_v4','E2/eval','Q_calibrated','H') for p in (root/parent).glob('*/metrics.json')]+[root/'V2_reference']
    assert len(roots)==19
    coverage={}
    for p in roots:
        metric=json.loads((p/'metrics.json').read_text())
        for split,expected in [('oof',16366),('et_test',1929),('sims_valid',647)]:
            path=p/(split+'.jsonl');rows=read_jsonl(path)
            uids=[r.get('utt_id',r.get('id')) for r in rows]
            assert len(rows)==expected and len(set(uids))==expected
            if split=='oof':assert set(uids)==cv
            if split=='et_test':assert set(uids)==et
            for r in rows:
                for point in ('op_a','op_n'):
                    value=np.array(r[point]);assert value.shape==(7,) and np.isfinite(value).all() and np.isclose(value.sum(),1,atol=1e-5)
            if 'artifacts' in metric and split in metric['artifacts']:
                assert sha256(path)==metric['artifacts'][split]['sha256']
        coverage[str(p)]={'cv':16366,'et_test':1929,'sims_valid':647}
    budgets=read_jsonl(root/'Q/q_budget_progress.jsonl')
    assert len(budgets)==6 and all(r['projected_total_gpu_h']<=75 and r['status']=='within_budget' for r in budgets)
    pipeline=json.loads((root/'Q/pipeline_progress.json').read_text());assert pipeline['completed']
    table=json.loads((root/'analysis_complete/main_table_complete.json').read_text())
    assert len(table)==8
    for n,v in table.items():
        if n=='old_lora_ET_only':continue
        for split in ('cv_op_a','cv_op_n','et_test_op_n'):
            interval=v[split]['complete_speaker_bootstrap_95']
            assert interval['replicates']==2000 and len(interval['metrics'])==6 and len(interval['per_class'])==7
        assert v['latency']['p95_s']>=v['latency']['p50_s']>=0
        assert v['resources']['total_neural_artifact_count']>0
    latency=json.loads((root/'latency/summary.json').read_text());assert latency['count']==1929
    timings=read_jsonl(root/'latency/et_test_latency.jsonl');assert len(timings)==1929 and {r['utt_id'] for r in timings}==et
    resources=json.loads((root/'resources/summary.json').read_text())
    qresource=json.loads((root/'resources_q/summary.json').read_text())
    supplemental_s=sum(v.get('process_elapsed_s_including_load',0) for v in resources['stages'].values())+qresource['elapsed_s_including_load']+3.154910724
    assert supplemental_s<3600
    tests=json.loads((root/'delivery_tests_20261002.json').read_text());assert tests['returncode']==0 and '33 passed' in tests['stdout']
    selected_roots=[Path(v['selected_source']) for k,v in table.items() if k!='old_lora_ET_only']
    final_meta_paths=[root/'E1_v4/run_meta.json',root/'H/run_meta.json',root/'Q_calibrated/run_meta.json',root/'analysis/run_meta.json',root/'analysis_complete/run_meta.json',root/'latency/run_meta.json']
    assert all(not json.loads(p.read_text())['git']['dirty'] for p in final_meta_paths)
    # Audit output paths/commands, without opening sealed data or original weights.
    for p in final_meta_paths:
        command=json.loads(p.read_text()).get('command',[])
        assert not any('sims_test' in str(v).lower() or 'sims-test' in str(v).lower() for v in command)
    incident_files=list((root/'Q').glob('fold*/incidents.jsonl'))+list((root/'Q/final').glob('incidents.jsonl'))
    incident_count=sum(len(read_jsonl(p)) for p in incident_files)
    assert incident_count==0
    result={'status':'passed','git':git,'command':sys.argv,'ended_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'script_sha256':sha256(Path(__file__)),'frozen_configs_verified':True,'q_adapters':adapters,
            'q_training':training,'q_raw_coverage':raw,'system_coverage':coverage,'verified_systems':19,
            'q_training_incidents':incident_count,'q_oom_skipped_utterances':0,'all_model_budget_gates_passed':True,
            'q_accounted_training_prediction_gpu_h':budgets[-1]['projected_total_gpu_h']-1.5,
            'q_accounted_with_reserves_gpu_h':budgets[-1]['projected_total_gpu_h'],
            'supplemental_resource_process_seconds_including_load_and_failed_V1':supplemental_s,
            'supplemental_resource_charged_to_diagnostic_reserve':True,
            'test_receipt':{'path':str(root/'delivery_tests_20261002.json'),'sha256':sha256(root/'delivery_tests_20261002.json')},
            'sealed_sims_test':'No inference entry invoked; audited target manifests only et_test/sims_valid; sealed contents not opened',
            'original_project':'Only read model/YuNet paths; all outputs under emotioncore_runs or H input cache',
            'used_final_run_meta':{str(p):sha256(p) for p in final_meta_paths},
            'main_table_sha256':sha256(root/'analysis_complete/main_table_complete.json')}
    write_json_atomic(root/'delivery_verification.json',result)
    print(json.dumps({'status':result['status'],'systems':len(roots),'supplemental_gpu_process_seconds':supplemental_s}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--x03-root',type=Path,required=True);run(p.parse_args().x03_root)
