import json
from pathlib import Path
import pytest
from emotioncore.x06b.r2 import select,agreement
from emotioncore.x06b.dialogue import build


def test_selection_rejects_slow_and_uses_iou_then_larger_scale():
    half={'scale':.5,'throughput':.51,'iou_ge_07_fraction':.99}
    third={'scale':1/3,'throughput':.4,'iou_ge_07_fraction':.8}
    assert select([half,third]) is third
    assert select([half,dict(third,throughput=.6)]) is None
    half['throughput']=.4;half['iou_ge_07_fraction']=.8
    assert select([half,third]) is half


def test_agreement_missing_detections_remain_in_denominator():
    r=agreement({('06',0):{0:[0,0,10,10],1:[0,0,10,10]}},{('06',0):{0:[0,0,10,10]}})
    assert r['iou_frame_denominator']==2 and r['iou_ge_07_fraction']==.5


def test_dialogue_gap_uses_turn_numbers_not_speaker_sorted_ids():
    def row(s,n,d=1,scene='01'):
        return {'group':'g','dialogue':scene,'speaker_id':s,'utt_id':f'g_{scene}_{s}_{n:03}',
                'official_split':'train','mp4':{'duration_s':d},'wav':{'duration_s':d},'labels':{'M':'neutral'}}
    records=[row('06',1),row('10',2,2),row('11',3,3),row('06',4),row('06',5,scene='02')]
    t=build(records)['06'];gap=t['entries'][1]
    assert [r['duration_s'] for r in gap['other_turns']]==[2,3]
    assert len(gap['turn_pauses_s'])==3
    assert gap['end_s']-gap['start_s']==pytest.approx(5+sum(gap['turn_pauses_s']))
    assert t['entries'][3]['end_s']-t['entries'][3]['start_s']==3
    assert all(a['end_s']==b['start_s'] for a,b in zip(t['entries'],t['entries'][1:]))
    assert build(records)==build(records)


def test_frozen_hashes_and_scope():
    import hashlib
    r=json.loads(Path('configs/x06b_registration.json').read_text())
    for p,h in r['frozen_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert r['system']=='H3'
    from emotioncore.x06b.common import source_timeline
    for s in ['01','sims_test','mer_test']:
        with pytest.raises(RuntimeError,match='population'):source_timeline(s)


def test_box_rescaling_with_rounded_dimensions(monkeypatch):
    import numpy as np
    import cv2
    from emotioncore.x06b import detect
    class Detector:
        def setInputSize(self,shape):self.shape=shape
        def detect(self,image):
            assert image.shape[:2]==(34,67)
            return None,np.array([[6.7,3.4,13.4,6.8]])
    monkeypatch.setattr(detect._LOCAL,'detector',Detector(),raising=False)
    boxes,_=detect.detect(np.zeros((101,200,3),dtype=np.uint8),1/3)
    assert boxes[0]==pytest.approx([20,10.1,40,20.2])
