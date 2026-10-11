"""Calibration and rejection metrics, with fixed speaker-cluster bootstrap."""
import numpy as np
from emotioncore.eval.metrics import LABELS

SEED=20261011
REPETITIONS=2000


def confusion(y, pred, classes=7):
    matrix=np.zeros((classes,classes+1),dtype=float)
    np.add.at(matrix,(y,np.where(pred<0,classes,pred)),1)
    return matrix


def classification(matrix):
    n=matrix.shape[-2]
    tp=np.diagonal(matrix[...,:n],axis1=-2,axis2=-1)
    support=matrix.sum(-1); predicted=matrix[...,:n].sum(-2)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros_like(tp),where=support+predicted>0)
    denom=support.sum(-1)
    return {'macro_f1':f1.mean(-1),'uar':recall.mean(-1),
            'wa':np.divide(tp.sum(-1),denom,out=np.zeros_like(denom),where=denom>0),
            'neutral_false_positive_rate':np.divide(matrix[...,0,1:n].sum(-1),support[...,0],
                out=np.zeros_like(tp[...,0]),where=support[...,0]>0),
            'nonneutral_macro_recall':recall[...,1:].mean(-1),
            'precision':precision,'recall':recall,'f1':f1,'support':support}


def bin_sums(confidence,correct):
    bins=np.minimum((confidence*15).astype(int),14)
    result=np.zeros((15,3))
    np.add.at(result,(bins,0),1)
    np.add.at(result,(bins,1),confidence)
    np.add.at(result,(bins,2),correct)
    return result


def ece(bins):
    return np.abs(bins[...,1]-bins[...,2]).sum(-1)/np.maximum(bins[...,0].sum(-1),1)


def reliability(bins):
    return [{'bin':k,'left':k/15,'right':(k+1)/15,'n':int(b[0]),
             'confidence':float(b[1]/b[0]) if b[0] else None,
             'accuracy':float(b[2]/b[0]) if b[0] else None} for k,b in enumerate(bins)]


def measure(y,p,pred,confidence,speakers,thresholds=None,bootstrap=True,label_names=None):
    y=np.asarray(y); pred=np.asarray(pred); speakers=np.asarray(speakers)
    keys=sorted(set(speakers)); classes=p.shape[1]
    label_names=list(LABELS[:classes]) if label_names is None else label_names
    matrices=[]; bins_a=[]; bins_n=[]; cbins=[]; sums=[]
    accepted={k:[] for k in (thresholds or {})}
    for speaker in keys:
        mask=speakers==speaker; yy=y[mask]; pp=p[mask]; guesses=pred[mask]
        matrices.append(confusion(yy,guesses,classes))
        bins_a.append(bin_sums(pp.max(1),(pp.argmax(1)==yy)&(guesses>=0)))
        bins_n.append(bin_sums(confidence[mask],guesses==yy))
        cbins.append(np.stack([bin_sums(pp[:,c],yy==c) for c in range(classes)]))
        loss=-np.log(np.clip(pp[np.arange(len(yy)),yy],1e-12,1)).sum()
        brier=np.square(pp-np.eye(classes)[yy]).sum()
        sums.append([len(yy),loss,brier])
        for key,th in (thresholds or {}).items():
            cutoff=np.broadcast_to(th,len(y))[mask]
            keep=(confidence[mask]>=cutoff)&(guesses>=0)
            accepted[key].append(confusion(yy,np.where(keep,guesses,-1),classes))
    matrices=np.asarray(matrices); bins_a=np.asarray(bins_a); bins_n=np.asarray(bins_n)
    cbins=np.asarray(cbins); sums=np.asarray(sums)
    total=matrices.sum(0); point=classification(total)
    out={k:float(v) for k,v in point.items() if k not in ('precision','recall','f1','support')}
    out.update(n=len(y),speaker_count=len(keys),nll=float(sums[:,1].sum()/len(y)),
        brier=float(sums[:,2].sum()/len(y)),top_label_ece=float(ece(bins_a.sum(0))),
        selected_label_ece=float(ece(bins_n.sum(0))),
        classwise_ece={label_names[c]:float(ece(cbins.sum(0)[c])) for c in range(classes)},
        per_class={label_names[c]:{k:float(point[k][c]) for k in ('precision','recall','f1','support')}
                   for c in range(classes)},confusion_with_reject_column=total.astype(int).tolist(),
        reliability={'op_a_top':reliability(bins_a.sum(0)),'op_n_selected':reliability(bins_n.sum(0))})
    if bootstrap:
        rng=np.random.default_rng(SEED)
        draws=rng.integers(len(keys),size=(REPETITIONS,len(keys)))
        weights=np.zeros((REPETITIONS,len(keys)))
        np.add.at(weights,(np.repeat(np.arange(REPETITIONS),len(keys)),draws.ravel()),1)
        boot=classification(np.einsum('rg,gij->rij',weights,matrices))
        counts=weights@sums[:,0]
        boot.update(nll=(weights@sums[:,1])/counts,brier=(weights@sums[:,2])/counts,
            top_label_ece=ece(np.einsum('rg,gij->rij',weights,bins_a)),
            selected_label_ece=ece(np.einsum('rg,gij->rij',weights,bins_n)))
        out['bootstrap']={'seed':SEED,'replicates':REPETITIONS,'unit':'speaker_id',
            'descriptive_only':len(keys)<=3,
            'ci95':{k:np.percentile(v,[2.5,97.5],axis=0).tolist() for k,v in boot.items()}}
    out['rejection']={}
    for key,group_matrices in accepted.items():
        group_matrices=np.asarray(group_matrices); total=group_matrices.sum(0)
        all_m=classification(total); accepted_total=total.copy(); accepted_total[:,-1]=0
        kept=classification(accepted_total); count=accepted_total.sum()
        item={'coverage':float(count/len(y)),'accepted_n':int(count),
              'accepted_macro_f1':float(kept['macro_f1']),
              'reject_in_recall_macro_f1':float(all_m['macro_f1']),
              'reject_in_recall_uar':float(all_m['uar'])}
        if bootstrap:
            mb=np.einsum('rg,gij->rij',weights,group_matrices)
            b_all=classification(mb); ma=mb.copy();ma[:,:,-1]=0;b_acc=classification(ma)
            series={'coverage':ma.sum((1,2))/mb.sum((1,2)),
                    'accepted_macro_f1':b_acc['macro_f1'],
                    'reject_in_recall_macro_f1':b_all['macro_f1']}
            item['ci95']={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in series.items()}
        out['rejection'][key]=item
    return out


def selective_curve(y,pred,confidence):
    # All tied scores enter together, so ordering by utterance ID cannot improve risk.
    order=np.argsort(-confidence,kind='stable'); sorted_conf=confidence[order]
    ends=np.r_[np.flatnonzero(np.diff(sorted_conf)!=0)+1,len(order)]
    errors=np.cumsum(pred[order]!=y[order]); risk=errors[ends-1]/ends
    coverage=ends/len(y); aurc=float(np.sum(risk*np.diff(np.r_[0.,coverage])))
    points=[]
    for target in np.linspace(.05,1,20):
        n=int(ends[np.searchsorted(coverage,target)])
        keep=order[:n]; m=classification(confusion(y[keep],pred[keep]))
        full=classification(confusion(y,np.where(confidence>=sorted_conf[n-1],pred,-1)))
        points.append({'coverage':n/len(y),'risk':float(np.mean(pred[keep]!=y[keep])),
            'accepted_macro_f1':float(m['macro_f1']),
            'reject_in_recall_macro_f1':float(full['macro_f1']), 'threshold':float(sorted_conf[n-1])})
    display=np.unique(np.searchsorted(coverage,np.linspace(coverage[0],1,101)))
    return {'aurc':aurc,'tie_policy':'accept all score ties together',
            'curve':{'coverage':coverage[display].tolist(),'risk':risk[display].tolist()},'macro_f1_curve':points}
