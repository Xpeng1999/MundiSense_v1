"""MER masks exactly one label-token position; ET uses original model loss."""
from emotioncore.eval.metrics import LABELS
from emotioncore.x03.q_train import prepare_target
from emotioncore.x04b.prepare import MER_LABELS

def token_ids(tokenizer):
    labels=(*LABELS,'other','worried')
    ids={label:tokenizer(label,add_special_tokens=False).input_ids[0] for label in labels}
    if len(set(ids.values()))!=9:raise ValueError('worried/X03 eight label first-token collision')
    return ids

def mer_target(values,truth,tokenizer):
    import torch
    if truth not in MER_LABELS:raise ValueError('MER label outside frozen six')
    answer=tokenizer(truth+'"}',add_special_tokens=False,return_tensors='pt').input_ids.to(values['input_ids'].device)
    end=torch.tensor([[tokenizer.convert_tokens_to_ids('<|im_end|>')]],dtype=answer.dtype,device=answer.device)
    answer=torch.cat((answer,end),dim=1);prompt=values['input_ids']
    values['input_ids']=torch.cat((prompt,answer),dim=1);values['attention_mask']=torch.ones_like(values['input_ids']);values['labels']=torch.cat((torch.full_like(prompt,-100),answer),dim=1)
    return values

def masked_mer_loss(logits,labels,allowed):
    import torch
    import torch.nn.functional as F
    if logits.shape[0]!=1 or labels.shape[0]!=1:raise ValueError('X03 microbatch must remain one')
    shift=labels[:,1:];valid=shift.ne(-100);indices=valid[0].nonzero().flatten()
    if len(indices)==0:raise ValueError('No label suffix')
    first=int(indices[0]);target=int(shift[0,first]);allowed=list(allowed)
    if target not in allowed:raise ValueError('MER target token outside six-class set')
    scores=logits.float()
    first_loss=F.cross_entropy(scores[0,first,allowed][None,:],torch.tensor([allowed.index(target)],device=labels.device),reduction='sum')
    tail=indices[1:]
    tail_loss=F.cross_entropy(scores[0,tail],shift[0,tail],reduction='sum') if len(tail) else first_loss.new_zeros(())
    return (first_loss+tail_loss)/len(indices)

def forward_loss(thinker,values,corpus,allowed):
    if corpus=='ET':
        result=thinker(**values,use_cache=False,use_audio_in_video=False)
        return result,result.loss
    if corpus!='MER':raise ValueError('Unknown training corpus')
    labels=values['labels'];inputs={k:v for k,v in values.items() if k!='labels'}
    result=thinker(**inputs,use_cache=False,use_audio_in_video=False)
    return result,masked_mer_loss(result.logits,labels,allowed)
