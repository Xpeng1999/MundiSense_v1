import copy,json
import pytest
from emotioncore.x04b.prepare import stratified_half,assert_split
from emotioncore.x04b.loss import masked_mer_loss,forward_loss,mer_target,token_ids

def test_stratified_epoch_reproducible_and_half():
    rows=[{'utt_id':str(i),'truth_M':str(i%6)} for i in range(102)]
    labels=['neutral','joy','sadness','anger','surprise','worried']
    for r in rows:r['truth_M']=labels[int(r['truth_M'])]
    assert stratified_half(rows,0)==stratified_half(list(reversed(rows)),0)
    assert len(stratified_half(rows,0))==51
    assert stratified_half(rows,0)!=stratified_half(rows,1)

def test_training_split_rejects_all_forbidden_sources():
    base=dict(utt_id='et',speaker_id='02',corpus='ET')
    with pytest.raises(ValueError):assert_split([base],{'et'},set(),{'02'})
    for r in [dict(utt_id='test',corpus='MER',split='mer_test'),dict(utt_id='sims',corpus='SIMS')]:
        with pytest.raises(ValueError):assert_split([r],{'et'},{'mer'},set())
    with pytest.raises(ValueError):assert_split([dict(utt_id='excluded',corpus='MER',split='mer_train')],set(),{'mer'},set())

def test_masked_first_token_zero_gradients_other_labels():
    import torch
    x=torch.randn(1,6,20,requires_grad=True);y=torch.tensor([[-100,-100,-100,3,10,11]])
    loss=masked_mer_loss(x,y,[1,2,3,4,5,6]);loss.backward()
    assert torch.equal(x.grad[0,2,[0,7,8,9,10,11,12,13,14,15,16,17,18,19]],torch.zeros(14))
    assert x.grad[0,3,7]!=0 # other target positions remain ordinary LM CE
    assert torch.isfinite(loss)

def test_et_original_loss_is_same_object_and_arguments():
    import torch
    from types import SimpleNamespace
    expected=torch.tensor(1.23456789,requires_grad=True);values={'input_ids':torch.tensor([[1,2]]),'labels':torch.tensor([[-100,2]])}
    def thinker(**kw):
        assert kw['labels'] is values['labels'] and kw['use_cache'] is False and kw['use_audio_in_video'] is False
        return SimpleNamespace(loss=expected)
    result,loss=forward_loss(thinker,values,'ET',[])
    assert loss is expected and torch.equal(loss,result.loss)
