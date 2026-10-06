import numpy as np
import pytest
import torch
from emotioncore.x04.heads import masked_ce, masks, validate_training, build_model, choose
from emotioncore.x02 import head_h2
from emotioncore.x02.folds import validate_oof


@pytest.mark.parametrize("corpus,truth,absent", [("ET",0,[7]),("MER",7,[4,5]),("CREMA",4,[6,7])])
def test_absent_class_gradient_is_exact_zero(corpus, truth, absent):
    logits = torch.randn(2,8,requires_grad=True)
    loss = masked_ce(logits,torch.tensor([truth,truth]),masks([corpus,corpus],"cpu")).sum()
    loss.backward()
    assert torch.equal(logits.grad[:,absent],torch.zeros_like(logits.grad[:,absent]))
    assert torch.isfinite(logits.grad).all()


def test_d0_preserves_original_initial_logits_and_rng():
    torch.manual_seed(20260928)
    old = head_h2.build_model(8,False,"cpu");old.eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(20260928)
    new = build_model(8,False,"cpu");new.eval()
    assert torch.equal(torch.random.get_rng_state(),state)
    x = torch.arange(48,dtype=torch.float32).reshape(2,3,8); n=torch.tensor([2,3])
    assert torch.allclose(old(x,n),new(x,n)[:,:7],atol=1e-7)


def test_selection_leakage_and_refit_distinction():
    rows=[dict(utt_id="et_a",corpus="ET",speaker_id="a",official_split="train"),
          dict(utt_id="et_b",corpus="ET",speaker_id="b",official_split="train"),
          dict(utt_id="mer_test",corpus="MER",speaker_id="unknown",official_split="test"),
          dict(utt_id="mer_excluded",corpus="MER",speaker_id="unknown",official_split="train"),
          dict(utt_id="sims",corpus="SIMS",speaker_id="unknown",official_split="train")]
    for index in (0,1,2,3,4):
        with pytest.raises(ValueError):
            validate_training(rows,np.array([index]),["a"],["b"],{"mer_excluded"},stage="selection")
    validate_training(rows,np.array([1]),["a"],[],set(),stage="refit")


def test_oof_missing_duplicate_rejected():
    with pytest.raises(ValueError):validate_oof({"a","b"},[{"utt_id":"a"}])
    with pytest.raises(ValueError):validate_oof({"a"},[{"utt_id":"a"},{"utt_id":"a"}])


def test_config_ties_match_x02_then_lower_alpha():
    def record(w,a):return dict(learning_rate=1e-4,weight_decay=w,alpha=a,validation=dict(macro_f1=.5,uar=.6))
    assert choose([record(.01,.25),record(0,1),record(0,.25)])["alpha"]==.25
    assert choose([record(.01,1),record(0,1)])["weight_decay"]==0
