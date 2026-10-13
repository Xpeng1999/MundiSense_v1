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


def test_incremental_crop_matches_x02_original_resolution_and_releases(tmp_path):
    import cv2,numpy as np
    from emotioncore.x06b.face import Tracker
    from emotioncore.x06.face import Tracker as Old
    from emotioncore.x02 import face as x02
    frame=np.random.default_rng(13).integers(0,256,(100,200,3),dtype=np.uint8)
    boxes=[[40,20,30,40]]
    new=Tracker(tmp_path/'new');new.add(frame,0,boxes)
    assert new.resident_frame_count==0 and not hasattr(new,'encoded')
    result=new.finish();assert new.closed and not new.tracks and not new.paths
    old=Old(None);x02.add_detections(old.tracks,0,boxes)
    old.encoded[0]=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,78])[1].tobytes();old.requested=1
    expected=old.finish(tmp_path/'old')
    assert Path(result['crop_paths'][0]).read_bytes()==Path(expected['crop_paths'][0]).read_bytes()
    assert Path(result['whole_paths'][0]).read_bytes()==Path(expected['whole_paths'][0]).read_bytes()
    with pytest.raises(RuntimeError):new.add(frame,1,boxes)


def test_detector_pipeline_queue_has_finite_frame_bound(monkeypatch):
    from emotioncore.x06b import detect
    produced=consumed=0
    def frames():
        nonlocal produced
        for i in range(100):
            produced+=1;assert produced-consumed<=4
            yield i,object()
    monkeypatch.setattr(detect,'detect',lambda frame,scale:([],0))
    for item,frame,boxes,seconds in detect.pipeline(frames(),.5,workers=2,capacity=4):consumed+=1
    assert consumed==100


def test_lowest_energy_window_is_interior_not_edges():
    import numpy as np
    from emotioncore.x06b.dialogue_streams import lowest_rms
    x=np.r_[np.ones(1600),np.zeros(1600),np.ones(1600)]
    assert lowest_rms(x)==0.
    assert lowest_rms(np.ones(10))==1.


def test_r3_bootstrap_is_paired_and_reproducible():
    import numpy as np
    from emotioncore.x06b.r3_analysis import paired
    y=np.tile(np.arange(7),3);a={'op_n':y,'r90':y}
    r=paired(y,a,a,iterations=50)
    assert r==paired(y,a,a,iterations=50)
    assert r['op_n']['ci95']==[0.,0.]
    b={'op_n':y,'r90':np.full(len(y),-1)}
    assert paired(y,a,b,iterations=50)['r90']['delta']==-1.


def test_r4_averaging_mixed_face_routes_does_not_require_h_ref():
    from emotioncore.x06b.r4_analysis import score
    bundle=json.loads(Path('configs/calibration_bundle_v1.1.json').read_text())
    p=[.1,.3,.05,.2,.05,.1,.2]
    segments=[{'start_s':0.,'end_s':1.,'has_face':True,'track_scores':[1.],
               'component_probs':{'Q_AV':p,'V2':p,'A3':p}},
              {'start_s':1.,'end_s':2.,'has_face':False,'track_scores':[],
               'component_probs':{'Q_AV':p,'Q_A':p,'V2':[1/7]*7,'A3':p}}]
    assert 'label' in score(bundle,{'start_s':0.,'end_s':2.},segments)


def test_live_window_bank_releases_frames_and_matches_incremental_window(tmp_path):
    import numpy as np
    from emotioncore.x06b.realtime import WindowBank
    bank=WindowBank(tmp_path);pixels=np.zeros((20,20,3),dtype=np.uint8)
    for i in range(5):bank.add(i,pixels,[])
    bank.update(('speech',0.),.6)
    bank.prune({('speech',0.)},.6)
    assert list(bank.frames)==[3,4]
    for i in range(5,10):bank.add(i,pixels,[])
    face=bank.finish(('speech',0.),1.6)
    assert face['decoded_frames']==8
    bank.prune(set(),2.)
    assert not bank.frames and not bank.tracks and bank.high_water<=10


def test_live_window_bank_refuses_to_hide_missing_frames(tmp_path):
    import numpy as np
    from emotioncore.x06b.realtime import WindowBank
    bank=WindowBank(tmp_path);bank.add(1,np.zeros((20,20,3),dtype=np.uint8),[])
    with pytest.raises(RuntimeError,match='missed'):bank.update(('speech',0.),.4)


def test_budget_preserves_r4_new_correct_boundary_control():
    from emotioncore.x06b.preflight import estimate
    d={'06':{'entries':[{'kind':'utterance','duration_s':2.},{'kind':'gap','start_s':2.,'end_s':8.}]}}
    result=estimate(1,d,{'both':2.,'Q_A':.3,'V1':.1},10.,.5)
    assert result['stage_gpu_seconds']['R3_correct']==pytest.approx(1.6)
    assert result['stage_gpu_seconds']['R4_correct_on_dialogue']==pytest.approx(1.6)
    assert result['metadata_gap_windows']==3


def test_stage_failure_cannot_launch_later_gpu(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from emotioncore.x06b import stages
    calls=[];monkeypatch.setattr(stages,'no_stop',lambda:None)
    def invoke(cmd,**kwargs):calls.append(cmd);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(stages.subprocess,'run',invoke)
    records,failed=stages.execute(tmp_path,[('r3_analysis','fmh'),('gpu_guard','r3_et_test')])
    assert len(calls)==1 and failed=='r3_analysis_fmh'
