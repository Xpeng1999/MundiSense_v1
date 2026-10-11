"""Only final Q inference on the fixed 06/10/11 FMH cohort; no metric analysis."""
import argparse
import os
from pathlib import Path
import time

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x03.q_smoke import MODEL, model_and_optimizer, assert_modality_receipt
from emotioncore.x03.q_train import load_input
from emotioncore.x04.prepare import write_rows
from emotioncore.x05.common import (R3, R4B, ROOT, FMH, MANIFEST, read, read_jsonl,
    sha256, spec, start, finish, storage_gate)

CACHE = Path('/mnt/h/emotionAI/emotioncore_models/X03/q_inputs')


def fmh_manifests(output):
    all_rows=read_jsonl(MANIFEST)
    expected={r['utt_id']:r for r in all_rows if r['speaker_id'] in FMH}
    if len(expected)!=3266:
        raise ValueError('STOP FMH coverage incomplete')
    sources={}; manifests={}
    frozen=read(R3/'Q/inference_manifests/run_meta.json')
    for form in ('A','V','AV'):
        result=[]
        for fold in range(5):
            entry=frozen['outputs'][f'fold{fold}_{form}']; path=Path(entry['path'])
            if sha256(path)!=entry['sha256']:
                raise ValueError('STOP Q source manifest changed')
            sources[str(path)]=entry
            result.extend({**r,'fold':None,'split':'fmh'} for r in read_jsonl(path)
                          if r['speaker_id'] in FMH)
        result.sort(key=lambda r:r['utt_id'])
        if len(result)!=3266 or {r['utt_id'] for r in result}!=set(expected):
            raise ValueError('STOP FMH modality coverage incomplete')
        for r in result:
            if r['truth_M']!=expected[r['utt_id']]['labels']['M']:
                raise ValueError('STOP FMH label mismatch')
            if r['effective_form']=='uniform':
                if not(form=='V' and r['face_missing']):
                    raise ValueError('STOP invalid modality fallback')
            else:
                for ext in ('json','npz'):
                    if not(CACHE/r['utt_id']/f"{r['effective_form']}.{ext}").is_file():
                        raise ValueError('STOP incomplete FMH input cache')
        path=output/f'{form}_manifest.jsonl'; write_rows(path,result)
        manifests[form]=result; sources[str(path)]=spec(path)
    for path in (R3/'Q/manifests/final_train.jsonl',R4B/'manifests/final_epoch0.jsonl',
                 R4B/'manifests/final_epoch1.jsonl'):
        training=[r for r in read_jsonl(path) if r.get('corpus','ET')=='ET']
        if {r['speaker_id'] for r in training}&FMH or {r['utt_id'] for r in training}&set(expected):
            raise ValueError('STOP FMH overlaps final Q training')
        sources[str(path)]=spec(path)
    return manifests,sources


def run(version):
    import torch
    from transformers import Qwen2_5OmniProcessor
    if os.environ.get('X05_GPU_GUARDED')!=f'fmh_q_{version}':
        raise RuntimeError('Use the X05 GPU watchdog')
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') or os.environ.get('PYTORCH_ALLOC_CONF'):
        raise ValueError('Default CUDA allocator required')
    if not read(ROOT/'reproduce_A3/summary.json')['passed']:
        raise RuntimeError('STOP A3 reproduction must pass first')
    output=ROOT/f'Q/{version}_fmh'; meta=start(output)
    manifests,sources=fmh_manifests(output)
    root=R3/'Q' if version=='et' else R4B/'Q'
    summary=read(root/'final/summary.json')
    adapter=Path(summary['best_checkpoint'])
    if sha256(adapter/'adapter_model.safetensors')!=summary['best_checkpoint_sha256']:
        raise ValueError('STOP final adapter SHA mismatch')
    meta.update(version=version,adapter=str(adapter),adapter_sha256=summary['best_checkpoint_sha256'],
        sources=sources,training=False,analysis=False,base_index=spec(MODEL/'model.safetensors.index.json'),
        cache_root=str(CACHE),settings='identical X04b final target inference; first-token seven-class scores')
    write_json_atomic(output/'run_meta.json',meta)
    torch.set_num_threads(4)
    processor=Qwen2_5OmniProcessor.from_pretrained(str(MODEL),local_files_only=True)
    tokens=[processor.tokenizer(c,add_special_tokens=False).input_ids[0] for c in LABELS]
    if len(set(tokens))!=7:
        raise ValueError('STOP seven-class token collision')
    model,optimizer,info=model_and_optimizer(); del optimizer
    model.load_adapter(str(adapter),adapter_name='best'); model.set_adapter('best')
    model.eval(); model.requires_grad_(False)
    result={}; started=time.monotonic()
    with torch.inference_mode():
        for form,rows in manifests.items():
            path=output/f'{form}.jsonl'
            with path.open('w') as stream:
                for i,row in enumerate(rows):
                    if i%25==0: storage_gate()
                    tic=time.monotonic()
                    if row['effective_form']=='uniform':
                        p=[1/7]*7; has_a=has_v=False; status='face_missing'
                    else:
                        values=load_input(CACHE,row)
                        assert_modality_receipt(row['effective_form'],set(values))
                        has_a='input_features' in values; has_v='pixel_values_videos' in values
                        # Any OOM aborts this run; no omitted sentences are permitted.
                        raw=model(**values,use_cache=False,use_audio_in_video=False)
                        p=raw.logits[0,-1].float()[tokens].softmax(0).cpu().tolist()
                        status='face_missing_fallback' if row['face_missing_fallback'] else 'ok'
                        del values,raw
                    import json
                    record={k:row[k] for k in ('utt_id','speaker_id','truth_M','effective_form',
                                                'face_missing','face_missing_fallback')}
                    record.update(form=form,split='fmh',fold=None,status=status,probabilities=p,
                                  forward_has_audio=has_a,forward_has_video=has_v,latency_s=time.monotonic()-tic)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if (i+1)%50==0:
                        stream.flush()
                        write_json_atomic(output/'progress.json',{'form':form,'done':i+1,'total':3266,
                            'inference_elapsed_s':time.monotonic()-started})
            result[form]={**spec(path),'n':len(rows),'uniform':sum(r['effective_form']=='uniform' for r in rows)}
    finish(output,meta,{'forms':result,'analysis_performed':False,'oom_skips':0,
                        'inference_elapsed_s':time.monotonic()-started,'model_settings':info})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--version',choices=('et','mix'),required=True)
    run(p.parse_args().version)
