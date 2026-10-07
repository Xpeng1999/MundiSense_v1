"""Retain longest20 AV memory gate for the expanded Q_mix training inputs."""
import argparse,datetime as dt,json,os,sys,time
from pathlib import Path
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl,sha256
from emotioncore.x03.q_smoke import MODEL,model_and_optimizer
from emotioncore.x03.q_train import load_input,prepare_target
from emotioncore.x04.prepare import storage_gate
from emotioncore.x04b.loss import token_ids,mer_target,forward_loss

def run(args):
    import torch
    from transformers import Qwen2_5OmniProcessor
    git=git_provenance(Path.cwd());storage_gate()
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') or os.environ.get('PYTORCH_ALLOC_CONF'):raise ValueError('Default allocator only')
    meta=json.loads((args.root/'manifests/run_meta.json').read_text());unique={}
    mer=json.loads((args.r4/'Q_media/summary.json').read_text());mer_av={r['utt_id']:r for r in read_jsonl(Path(mer['outputs']['train_AV']['path']))}
    for spec in meta['runs'].values():
        for epoch in spec['epochs']:
            p=Path(epoch['path'])
            if sha256(p)!=epoch['sha256']:raise ValueError('Epoch changed')
            for row in read_jsonl(p):
                if row['face_missing']:continue
                if row['corpus']=='MER':row={**row,**mer_av[row['utt_id']]}
                row={**row,'assigned_form':'AV','effective_form':'AV'}
                unique[row['utt_id']]=row
    ranked=[]
    for row in unique.values():
        p=Path(row['cache_root'])/row['utt_id']/'AV.json';rec=json.loads(p.read_text())
        ranked.append((rec['shapes']['input_ids'][-1],row['utt_id'],row))
    ranked.sort(key=lambda x:(-x[0],x[1]));selected=ranked[:20]
    args.output.mkdir(parents=True,exist_ok=True);selection=[{'rank':i+1,'length':n,'utt_id':uid,'corpus':row['corpus'],'row':row} for i,(n,uid,row) in enumerate(selected)]
    write_json_atomic(args.output/'selection.json',selection)
    started=time.monotonic();processor=Qwen2_5OmniProcessor.from_pretrained(str(MODEL),local_files_only=True);ids=token_ids(processor.tokenizer)
    model,optimizer,info=model_and_optimizer();model.train();receipts=[]
    run_meta={'git':git,'command':sys.argv,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'selection_sha256':sha256(args.output/'selection.json'),'gradient_checkpointing':info['gradient_checkpointing'],'cuda_allocator':'default'}
    write_json_atomic(args.output/'run_meta.json',run_meta)
    for n,uid,row in selected:
        storage_gate();model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        root=Path(row['cache_root']);rec=json.loads((root/uid/'AV.json').read_text())
        if sha256(root/uid/'AV.npz')!=rec['sha256']:raise ValueError('Selected cache SHA mismatch')
        values=load_input(root,row);values=(prepare_target if row['corpus']=='ET' else mer_target)(values,row['truth_M'],processor.tokenizer)
        result,loss=forward_loss(model,values,row['corpus'],[ids[c] for c in ('neutral','joy','sadness','anger','surprise','worried')]);loss.backward();torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()
        receipts.append({'utt_id':uid,'corpus':row['corpus'],'input_token_length':n,'peak_allocated_bytes':peak,'loss':float(loss.detach()),'passed':peak<=30000000000});del result,loss,values
        write_json_atomic(args.output/'progress.json',{'done':len(receipts),'total':20,'receipts':receipts})
        if peak>30000000000:raise RuntimeError('STOP longest20 peak exceeds30GB')
    result={'gate':'pass','tested':20,'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in receipts),'receipts':receipts,'elapsed_s':time.monotonic()-started,'selection_sha256':run_meta['selection_sha256']}
    write_json_atomic(args.output/'summary.json',result);run_meta['ended_at']=dt.datetime.now(dt.timezone.utc).isoformat();run_meta['summary_sha256']=sha256(args.output/'summary.json');write_json_atomic(args.output/'run_meta.json',run_meta)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--r4',type=Path,default=Path('/mnt/h/emotionAI/emotioncore_runs/X04a'));run(p.parse_args())
if __name__=='__main__':main()
