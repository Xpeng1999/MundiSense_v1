import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from emotioncore.eval.metrics import LABELS
from emotioncore.x05.bundle import Bundle,verify_method_freeze
from emotioncore.x05.calibration import SYSTEMS


def parameters():
    systems={}
    for name in ('H_ref','H3','H_A','H_V'):
        components=SYSTEMS[name];m=len(components)
        systems[name]={'components':list(components),'fusion':[1/m]*m+[0.]*7,
            'calibration':{'method':'temperature','log_temperature':[0.],'bias':[0.]*7},
            'neutral_bias':0.,'thresholds':{'90':.5,'80':.7}}
    return {'version':1,'labels':list(LABELS),'systems':systems,'method_freeze_sha256':'placeholder'}


def test_roundtrip_and_deterministic_apply(tmp_path):
    freeze=tmp_path/'x05_method_freeze.json';freeze.write_text('{"version":1}\n')
    p=parameters();p['method_freeze_sha256']=hashlib.sha256(freeze.read_bytes()).hexdigest()
    path=tmp_path/'calibration_bundle_v1.json';path.write_text(json.dumps(p))
    loaded=Bundle.load(path);assert loaded.parameters==p
    probs={c:[.1,.8,.02,.02,.02,.02,.02] for c in set(sum([list(v) for v in SYSTEMS.values()],[]))}
    availability=dict(has_face=True,face_count=1,has_speech=True,face_quality=.8)
    first=loaded.apply(probs,availability)
    assert first==Bundle(p).apply(probs,availability)==loaded.apply(probs,availability)
    assert first[0]=='joy' and first[2]=='H_ref'
    assert loaded.apply(probs,{**availability,'has_face':False,'face_count':0})[2]=='H_A'
    assert loaded.apply(probs,{**availability,'has_speech':False})[2]=='H_V'
    assert loaded.apply({}, {**availability,'face_count':2})==('reject',0.,'reject')


def test_freeze_tampering_is_rejected(tmp_path):
    path=tmp_path/'x05_method_freeze.json';path.write_text('{"version":1}')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_method_freeze(path,digest)['version']==1
    path.write_text('{"version":2}')
    with pytest.raises(ValueError,match='SHA mismatch'):verify_method_freeze(path,digest)


def test_missing_components_fail_and_low_confidence_rejects():
    b=Bundle(parameters());availability=dict(has_face=False,face_count=0,has_speech=True)
    with pytest.raises(KeyError):b.apply({},availability)
    result=b.apply({'Q_A':[1/7]*7,'A3':[1/7]*7},availability)
    assert result[0]=='reject' and result[2]=='H_A'


def test_inference_entry_points_cannot_select_sealed_test():
    from emotioncore.x05.gpu_guard import TASKS
    assert set(TASKS)=={'reproduce_a3','fmhx_a3','fmh_q_et','fmh_q_mix'}
    # X05 inference CLIs expose version/encoder only; there is no split/path override.
    root=Path(__file__).parents[1]/'src/emotioncore/x05'
    for name in ('q_fmh.py','targets.py'):
        source=(root/name).read_text()
        assert '--split' not in source and '--manifest' not in source
        assert 'sims_test' not in source and 'CH-SIMS/test' not in source


def test_real_bundle_replays_reported_target_predictions():
    from emotioncore.x05.common import ROOT,read,read_jsonl
    from emotioncore.x05.deployment import target_maps
    bundle_path=ROOT/'deployment/calibration_bundle_v1.json'
    if not bundle_path.exists():pytest.skip('post-G deployment artifact not yet available')
    b=Bundle(read(bundle_path));assert Bundle(json.loads(json.dumps(b.parameters))).parameters==b.parameters
    ids,maps=target_maps('et_test',{})
    missing=set(read('configs/x03_face_missing_policy.json')['face_missing_utt_ids'])
    for name,params in b.parameters['systems'].items():
        expected={r['utt_id']:r for r in read_jsonl(ROOT/f'deployment/{name}_et_test.jsonl')}
        valid=[uid for uid in ids if uid not in missing and all(maps[c][uid].get('status')=='ok' for c in params['components'])][:30]
        assert len(valid)==30
        for uid in valid:
            availability=dict(has_face=name!='H_A',face_count=0 if name=='H_A' else 1,
                              has_speech=name!='H_V',full_system=name if name in ('H_ref','H3') else 'H_ref',
                              rejection_point=None)
            inputs={c:maps[c][uid]['probabilities'] for c in params['components']}
            label,confidence,routing=b.apply(inputs,availability)
            assert label==LABELS[expected[uid]['label']]
            assert confidence==pytest.approx(expected[uid]['confidence'],abs=1e-12)
            assert routing==name
