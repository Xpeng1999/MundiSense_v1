import numpy as np
import pytest
from emotioncore.x05 import calibration as c
from emotioncore.x05.metrics import confusion,classification,measure
from emotioncore.x05.gate import route


@pytest.mark.parametrize('face,count,speech,expected',[(False,0,True,'H_A'),
    (True,1,False,'H_V'),(False,0,False,'reject'),(True,2,True,'reject')])
def test_routing(face,count,speech,expected):
    assert route(face,count,.4,speech)[0]==expected
    if count>1:assert route(face,count,.4,speech)[1]=='multi_face'


def test_rejects_stay_in_recall_denominator():
    y=np.array([0,0,1,1]);pred=np.array([0,-1,1,-1])
    result=classification(confusion(y,pred))
    np.testing.assert_array_equal(result['recall'][:2],[.5,.5])
    assert result['neutral_false_positive_rate']==0
    assert result['f1'][0]==pytest.approx(2/3)


def test_four_fold_fit_never_reads_held_labels(monkeypatch):
    folds=np.repeat(np.arange(5),7);x=np.zeros((35,1,7));seen=[]
    def labels(ids):
        assert not np.any(folds[ids]==2)
        seen.extend(ids);return ids%7
    monkeypatch.setattr(c,'fit_base',lambda x,y,m:{})
    monkeypatch.setattr(c,'base_scores',lambda x,p:np.zeros((len(x),7)))
    monkeypatch.setattr(c,'fit_operating',lambda x,y:{})
    held,_,_,fit=c.fit_held_fold(x,labels,folds,2,'temperature')
    assert set(folds[seen])=={0,1,3,4}
    assert set(held).isdisjoint(fit)


def test_fmh_loso_fit_never_reads_held_speaker_labels(monkeypatch):
    speakers=np.repeat(['06','10','11'],7);x=np.zeros((21,1,7));seen=[]
    def labels(ids):
        assert not np.any(speakers[ids]=='10')
        seen.extend(ids);return ids%7
    monkeypatch.setattr(c,'fit_base',lambda x,y,m:{})
    monkeypatch.setattr(c,'base_scores',lambda x,p:np.zeros((len(x),7)))
    monkeypatch.setattr(c,'fit_operating',lambda x,y:{})
    held,_,_,fit=c.fit_held_speaker(x[:7],np.arange(7),x,labels,speakers,'10','temperature')
    assert set(speakers[seen])=={'06','11'} and set(held).isdisjoint(fit)


def test_selection_tolerance_and_order():
    assert c.select_method(dict(zip(c.METHODS,[1.0019,1.0,1.0001])))=='temperature'
    assert c.select_method(dict(zip(c.METHODS,[1.01,1.001,1.0])))=='temperature_bias'


def test_confidence_is_probability_before_neutral_shift():
    scores=np.log([[.2,.7,.02,.02,.02,.02,.02]])
    p,label,confidence,_=c.predictions(scores,2.)
    assert label.tolist()==[0]
    assert confidence[0]==pytest.approx(.2)


def test_calibrator_nll_and_metrics_finite():
    rng=np.random.default_rng(123);y=np.tile(np.arange(7),10)
    logits=rng.normal(size=(70,7));logits[np.arange(70),y]+=2
    for method in c.METHODS:
        params=c.fit_calibrator(logits,y,method)
        assert c.nll(c.calibrate(logits,params),y)<=c.nll(logits,y)+1e-4
    p=c.softmax(logits);pred=p.argmax(1);confidence=p.max(1)
    result=measure(y,p,pred,confidence,np.repeat(['a','b'],35),{'90':.5})
    assert result['bootstrap']['replicates']==2000
    assert 0<=result['rejection']['90']['coverage']<=1
