import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x03.complete_analysis import speaker_intervals, resource_counts
from emotioncore.x03.latency import serial_stages


def test_speaker_bootstrap_preserves_forced_errors_and_ece():
    rows=[]
    for s in ('01','02','03'):
        for i,label in enumerate(LABELS):
            p=np.full(7,.04);p[i]=.76
            rows.append({'id':s+label,'speaker_id':s,'truth':label,
                         'pred':label if s!='03' else None,'probabilities':p.tolist()})
    result=speaker_intervals(rows,repetitions=50)
    assert result['speaker_count']==3
    assert result['metrics']['macro_f1']['lower_95']<1
    assert result['metrics']['ece15']['upper_95']>0
    assert result['per_class']['neutral']['recall']['lower_95']<1


def test_shared_stages_counted_once_and_heads_explicit():
    components=['Q_A','Q_AV','A3_h2','V2_h2','V1_h1']
    stages=serial_stages(components)
    assert stages.count('face')==1
    assert stages.count('Q_video_prepare')==1
    assert 'Q_A_forward' in stages and 'Q_AV_forward' in stages
    count=resource_counts(components)
    assert count['primary_model_count']==5
    assert count['classifier_head_count']==3
    assert count['total_neural_artifact_count']==8
