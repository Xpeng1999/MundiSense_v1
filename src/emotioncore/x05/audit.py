"""Independent receipt/coverage and confusion-matrix replay before delivery."""
from pathlib import Path
import subprocess
import numpy as np
from emotioncore.eval.metrics import LABELS,classification
from emotioncore.x05.common import ROOT,FMH,MANIFEST,read,read_jsonl,sha256,spec,start,finish
from emotioncore.x05.bundle import Bundle,verify_method_freeze


def run():
    output=ROOT/'delivery_audit';meta=start(output)
    bundle=read(ROOT/'deployment/calibration_bundle_v1.json')
    verify_method_freeze('configs/x05_method_freeze.json',bundle['method_freeze_sha256'],True)
    canonical={r['utt_id']:r for r in read_jsonl(MANIFEST)}
    fmh_ids={uid for uid,r in canonical.items() if r['speaker_id'] in FMH}
    checks={};provenance={}
    for version in ('et','mix'):
        root=ROOT/f'Q/{version}_fmh';summary=read(root/'summary.json')
        for form in ('A','V','AV'):
            s=summary['forms'][form];path=Path(s['path'])
            assert sha256(path)==s['sha256']
            rows=read_jsonl(path)
            assert len(rows)==3266 and {r['utt_id'] for r in rows}==fmh_ids
            for r in rows:
                truth=canonical[r['utt_id']]
                assert (r['truth_M'],r['speaker_id'])==(truth['labels']['M'],truth['speaker_id'])
                effective=r['effective_form']
                assert r['forward_has_audio']==(effective in ('A','AV'))
                assert r['forward_has_video']==(effective in ('V','AV'))
                p=np.asarray(r['probabilities']);assert p.shape==(7,) and np.isfinite(p).all() and (p>=0).all()
                assert abs(p.sum()-1)<1e-5
            checks[f'{version}_{form}']={'n':3266,'exact_ids_labels_speakers':True,'modality_receipts':True}
            provenance[str(path)]=s
    for c in ('A3','V2','V1'):
        root=ROOT/f'heads/D1_fmhx/{c}';summary=read(root/'summary.json')
        assert sha256(Path(summary['checkpoint']['path']))==summary['checkpoint']['sha256']
        ids=read_jsonl(root/'training_ids.jsonl')
        assert not {r['utt_id'] for r in ids}&fmh_ids
        assert all(r['corpus'] in ('ET','MER') and r['official_split']!='test' for r in ids)
        assert not {r['speaker_id'] for r in ids if r['corpus']=='ET'}&FMH
        assert summary['training']['validation_n']==0 and summary['training']['validation_seconds']==0
        assert summary['training']['epochs_executed']==summary['config']['best_epoch']
        checks[c]={'training_count':len(ids),'fmh_intersection':0,'no_validation':True,'checkpoint_sha_verified':True}
    deployment=read(ROOT/'deployment/summary.json');recomputed={}
    for name in ('H_ref','H3','H_A','H_V'):
        rows=read_jsonl(ROOT/f'deployment/{name}_fmh_loso.jsonl')
        assert len(rows)==3266 and {r['utt_id'] for r in rows}==fmh_ids
        scored=[{'truth':canonical[r['utt_id']]['labels']['M'],
                 'pred':LABELS[r['label']] if r['label']>=0 else None} for r in rows]
        result=classification(scored);reported=deployment['fmh_loso'][name]['all']
        for key in ('macro_f1','uar','wa','neutral_false_positive_rate'):
            assert abs(result[key]-reported[key])<1e-12,(name,key)
        for point in ('90','80'):
            accepted=[r['label']>=0 and r['confidence']>=r['thresholds'][point] for r in rows]
            selective=[{**s,'pred':s['pred'] if keep else None} for s,keep in zip(scored,accepted)]
            score=classification(selective)
            assert abs(score['macro_f1']-reported['rejection'][point]['reject_in_recall_macro_f1'])<1e-12
            assert abs(np.mean(accepted)-reported['rejection'][point]['coverage'])<1e-12
        recomputed[name]={key:result[key] for key in ('macro_f1','uar','wa','neutral_false_positive_rate')}
    fit=read_jsonl(Path(bundle['fitting_set']['path']))
    assert sha256(Path(bundle['fitting_set']['path']))==bundle['fitting_set']['sha256']
    assert len(fit)==9799 and len({r['utt_id'] for r in fit})==9799 and len({r['speaker_id'] for r in fit})==9
    gpu=[read(p) for p in (ROOT/'gpu_jobs').glob('*/summary.json')]
    assert len(gpu)==4 and all(r['status']=='completed' for r in gpu)
    gpu_s=sum(r['charged_gpu_s'] for r in gpu);assert gpu_s<=18000
    stamp=subprocess.check_output(['git','show','-s','--format=%cI',bundle['method_freeze_commit']],text=True).strip()
    from datetime import datetime
    assert datetime.fromisoformat(stamp)<datetime.fromisoformat(read(ROOT/'deployment/run_meta.json')['started_at'])
    unchanged=subprocess.check_output(['git','diff','69eb5263724cf32b290d107d960e91249cf12b7b','--',
        'src/emotioncore/x04','configs/folds_v1.json','configs/x03_inner_val_subsets.json','configs/x03_face_missing_policy.json'])
    assert not unchanged
    finish(output,meta,{'status':'passed','coverage_and_training':checks,'fmh_independent_recalculation':recomputed,
        'freeze_committed_before_q_mix_analysis':True,'freeze_commit_time':stamp,'fitting_n':9799,
        'fitting_speakers':9,'gpu_hours':gpu_s/3600,'X04a_and_frozen_split_sources_unchanged':True,
        'bundle':spec(ROOT/'deployment/calibration_bundle_v1.json'),'final_tests':read(ROOT/'final_tests/summary.json')['totals']})


if __name__=='__main__':run()
