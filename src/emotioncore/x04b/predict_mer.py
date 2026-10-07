"""Q_mix MER-test: unchanged prompt, restricted six-native first-token scores."""
import argparse,datetime as dt,json,os,sys,time
from pathlib import Path
import numpy as np
from emotioncore.x02.runtime import git_provenance,write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl,sha256
from emotioncore.x03.q_smoke import MODEL,model_and_optimizer,assert_modality_receipt
from emotioncore.x03.q_train import load_input
from emotioncore.x04.prepare import write_rows,storage_gate
from emotioncore.x04b.loss import token_ids
from emotioncore.x04b.prepare import MER_LABELS
from emotioncore.x04.heads import LABELS8

def run(args):
    import torch
    from transformers import Qwen2_5OmniProcessor
    git=git_provenance(Path.cwd());storage_gate()
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') or os.environ.get('PYTORCH_ALLOC_CONF'):raise ValueError('Default allocator only')
    if sha256(args.adapter/'adapter_model.safetensors')!=args.adapter_sha256:raise ValueError('Adapter changed')
    manifests=json.loads((args.r4/'Q_media/summary.json').read_text())['outputs'];rows={}
    for f in ('A','AV'):
        s=manifests['test_'+f];p=Path(s['path'])
        if sha256(p)!=s['sha256']:raise ValueError('MERtest source changed')
        rows[f]=read_jsonl(p)
        if len(rows[f])!=572 or len({r['utt_id'] for r in rows[f]})!=572 or any(r['split']!='mer_test' for r in rows[f]):raise ValueError('MER-test coverage')
    args.output_dir.mkdir(parents=True,exist_ok=True);meta={'git':git,'command':sys.argv,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),'adapter_sha256':args.adapter_sha256,'training':False,'classes':MER_LABELS}
    write_json_atomic(args.output_dir/'run_meta.json',meta)
    processor=Qwen2_5OmniProcessor.from_pretrained(str(MODEL),local_files_only=True);ids=token_ids(processor.tokenizer);allowed=[ids[c] for c in MER_LABELS]
    model,opt,info=model_and_optimizer();del opt;model.load_adapter(str(args.adapter),adapter_name='best');model.set_adapter('best');model.eval();torch.set_num_threads(4);output={}
    with torch.inference_mode():
        for f,rr in rows.items():
            preds=[]
            for i,row in enumerate(rr):
                if i%50==0:storage_gate()
                values=load_input(args.r4/'Q_inputs',row);assert_modality_receipt(row['effective_form'],set(values));result=model(**values,use_cache=False,use_audio_in_video=False)
                p6=result.logits[0,-1].float()[allowed].softmax(0).cpu().numpy();p8=np.zeros(8)
                for c,p in zip(MER_LABELS,p6):p8[LABELS8.index(c)]=p
                p7=p8[:7]/p8[:7].sum();preds.append({**row,'probabilities':p7.tolist(),'probabilities8':p8.tolist(),'status':'face_missing_fallback' if row['face_missing_fallback'] else 'ok','form':f});del values,result
                if (i+1)%50==0:write_json_atomic(args.output_dir/'progress.json',{'form':f,'done':i+1,'total':572})
            path=args.output_dir/(f+'.jsonl');write_rows(path,preds);output[f]={'path':str(path),'sha256':sha256(path),'n':572}
    write_json_atomic(args.output_dir/'summary.json',output);meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),summary_sha256=sha256(args.output_dir/'summary.json'),gradient_checkpointing=info['gradient_checkpointing']);write_json_atomic(args.output_dir/'run_meta.json',meta)

def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter',type=Path,required=True);p.add_argument('--adapter-sha256',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--r4',type=Path,default=Path('/mnt/h/emotionAI/emotioncore_runs/X04a'));run(p.parse_args())
if __name__=='__main__':main()
