import json
import pytest
from emotioncore.x04.pipeline import resume_receipt, completed_head
from emotioncore.x02.face import sha256


def test_resume_keeps_gpu_ledger_and_requires_completed_preparation(tmp_path):
    path=tmp_path/'run_meta.json'
    saved={'status':'stopped','budget_sha256':'frozen','charged_gpu_hours':1.689,
           'steps':[{'name':n,'returncode':0} for n in
                    ('full_face','full_A3','full_V2','full_V1','data_evidence','heads_D0_A3_first_grid')]}
    path.write_text(json.dumps(saved))
    assert resume_receipt(path,'frozen')['charged_gpu_hours']==1.689
    with pytest.raises(ValueError):resume_receipt(path,'other')
    saved['steps'].pop();path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):resume_receipt(path,'frozen')
    saved['charged_gpu_hours']=20;path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):resume_receipt(path,'frozen')


def test_repeated_resume_validates_history_sha_and_monotonic_gpu_ledger(tmp_path):
    parent=tmp_path/'parent.json';child=tmp_path/'child.json'
    old={'status':'stopped','budget_sha256':'frozen','charged_gpu_hours':1.689,
         'steps':[{'name':n,'returncode':0} for n in
                  ('full_face','full_A3','full_V2','full_V1','data_evidence','heads_D0_A3_first_grid')]}
    parent.write_text(json.dumps(old))
    new={**old,'charged_gpu_hours':1.988,'steps':old['steps'][:-1],
         'resumed_from':{'path':str(parent),'sha256':sha256(parent)}}
    child.write_text(json.dumps(new));assert resume_receipt(child,'frozen')['charged_gpu_hours']==1.988
    new['charged_gpu_hours']=1.;child.write_text(json.dumps(new))
    with pytest.raises(ValueError):resume_receipt(child,'frozen')
    new['charged_gpu_hours']=1.988;child.write_text(json.dumps(new));parent.write_text('{}')
    with pytest.raises(ValueError):resume_receipt(child,'frozen')


def test_completed_head_reuse_rejects_changed_checkpoint(tmp_path):
    assert completed_head(tmp_path) is None
    (tmp_path/'oof.jsonl').write_text('frozen OOF')
    (tmp_path/'final_head.pt').write_bytes(b'frozen checkpoint')
    (tmp_path/'results.json').write_text(json.dumps({'oof_sha256':sha256(tmp_path/'oof.jsonl'),
        'checkpoint_sha256':sha256(tmp_path/'final_head.pt')}))
    (tmp_path/'run_meta.json').write_text(json.dumps({'ended_at':'complete','elapsed_s':973.}))
    assert completed_head(tmp_path)['elapsed_s']==973.
    (tmp_path/'final_head.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):completed_head(tmp_path)
