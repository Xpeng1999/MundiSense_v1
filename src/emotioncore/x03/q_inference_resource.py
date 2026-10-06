"""Condition-specific memory probes on the longest 20 cached ET-test inputs."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import model_and_optimizer, assert_modality_receipt
from emotioncore.x03.q_train import load_input


def run(args):
    import torch
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') or os.environ.get('PYTORCH_ALLOC_CONF'):
        raise RuntimeError('Default allocator required')
    git=git_provenance(Path.cwd());started=time.monotonic()
    torch.set_num_threads(4)
    train=json.loads((args.q_root/'final/summary.json').read_text())
    adapter=Path(train['best_checkpoint'])
    if sha256(adapter/'adapter_model.safetensors')!=train['best_checkpoint_sha256']:
        raise ValueError('Final adapter SHA changed')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json_atomic(args.output_dir/'run_meta.json',{'git':git,'command':sys.argv,
                      'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'adapter_sha256':train['best_checkpoint_sha256']})
    model,optimizer,info=model_and_optimizer();del optimizer
    model.load_adapter(str(adapter),adapter_name='best');model.set_adapter('best');model.eval()
    result={}
    with torch.inference_mode():
        for form in ('A','V','AV'):
            path=args.q_root/'inference_manifests'/f'final_et_test_{form}.jsonl'
            rows=[r for r in read_jsonl(path) if r['effective_form']!='uniform']
            def length(r):
                cache=json.loads((args.cache_root/r['utt_id']/(r['effective_form']+'.json')).read_text())
                return cache['shapes']['input_ids'][-1]
            rows=sorted(rows,key=lambda r:(-length(r),r['utt_id']))[:20]
            torch.cuda.reset_peak_memory_stats()
            probes=[]
            for row in rows:
                values=load_input(args.cache_root,row)
                assert_modality_receipt(row['effective_form'],set(values))
                output=model(**values,use_cache=False,use_audio_in_video=False)
                torch.cuda.synchronize()
                probes.append({'utt_id':row['utt_id'],'effective_form':row['effective_form'],
                               'tokens':length(row),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
                del values,output
            result[form]={'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'selection_rule':'20 longest input_ids in frozen ET-test '+form+' manifest',
                          'manifest_sha256':sha256(path),'samples':probes}
    write_json_atomic(args.output_dir/'summary.json',{'conditions':result,'gradient_checkpointing':info['gradient_checkpointing'],
                      'cuda_allocator':'default','elapsed_s_including_load':time.monotonic()-started,
                      'limitation':'longest 20 ET-test inputs per form; full formal inference overall peak also retained'})
    write_json_atomic(args.output_dir/'run_meta.json',{'git':git,'command':sys.argv,
                      'ended_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                      'adapter_sha256':train['best_checkpoint_sha256'],'summary_sha256':sha256(args.output_dir/'summary.json')})


def main():
    p=argparse.ArgumentParser()
    for flag in ('q-root','cache-root','output-dir'):p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
