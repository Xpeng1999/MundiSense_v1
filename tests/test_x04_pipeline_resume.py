import json
import pytest
from emotioncore.x04.pipeline import resume_receipt


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
