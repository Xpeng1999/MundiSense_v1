"""Measure encoder GPU peaks sequentially on the 20 longest ET-test clips.

These are serial-stage measurements, not estimates of simultaneous residency.
Q's full ET-test inference peak is taken from its existing formal receipt.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MEDIA_ROOT


def worker(stage: str, model: Path):
    import torch
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    protocol = sys.stdout

    class Stream:
        def write(self, text):
            if text.startswith('{'):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError:
                    pass
                else:
                    if 'status' in value:
                        torch.cuda.synchronize()
                        value['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                        value['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
                        value['nvidia_smi_used_mib'] = subprocess.check_output(
                            ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
                        text = json.dumps(value) + ('\n' if text.endswith('\n') else '')
            return protocol.write(text)
        def flush(self):
            protocol.flush()
        def __getattr__(self, key):
            return getattr(protocol, key)
    sys.stdout = Stream()
    names = {'A3':'audio_a3','A2':'audio_a2','V2':'visual_v2','V1':'visual_v1','SenseVoice':'asr_sensevoice'}
    importlib.import_module('emotioncore.x02.'+names[stage]).worker_main(model)


def run(args):
    started=time.monotonic()
    git=git_provenance(Path.cwd())
    r2,r3=args.x02_root,args.x03_root
    rows=sorted([r for r in read_jsonl(args.manifest) if r['official_split']=='test'],
                key=lambda r:(-r['wav']['duration_s'],r['utt_id']))[:20]
    selection=[{'utt_id':r['utt_id'],'audio_duration_s':r['wav']['duration_s'],
                'wav_sha256':r['wav']['sha256'],'mp4_sha256':r['mp4']['sha256']} for r in rows]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json_atomic(args.output_dir/'selection.json',selection)
    meta={'git':git,'command':sys.argv,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),
          'selection_sha256':sha256(args.output_dir/'selection.json'),'manifest_sha256':sha256(args.manifest)}
    write_json_atomic(args.output_dir/'run_meta.json',meta)
    summary={}
    models={'A3':r2/'models/A3','A2':Path('/home/administrator/projects/emotion/omni_models/Qwen2.5-Omni-7B'),
            'V2':r2/'models/V2','V1':Path('/home/administrator/emotion-data/open_resources_v1/models/emotiefflib/enet_b0_8_best_afew.pt'),
            'SenseVoice':Path('/home/administrator/emotion-data/open_resources_v1/models/SenseVoiceSmall')}
    for stage,model in models.items():
        start=time.monotonic()
        env=os.environ.copy()
        env.pop('CUDA_VISIBLE_DEVICES',None)
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None);env.pop('PYTORCH_ALLOC_CONF',None)
        python = '/home/administrator/emotion-data/open_resources_v1/envs/audio/bin/python' if stage=='SenseVoice' else sys.executable
        child=PersistentJSONWorker([python,'-u','-m','emotioncore.x03.stage_resource','--worker',stage,'--model',str(model)],
                                   cwd=Path.cwd(),env=env,stderr_path=args.output_dir/(stage+'_errors.log'))
        measured=[]
        try:
            for row in rows:
                uid=row['utt_id']
                task={'id':uid,'output':str(args.output_dir/(stage+'_scratch.json' if stage=='SenseVoice' else stage+'_scratch.npz'))}
                if stage.startswith('V'):
                    face=json.loads((r2/f'face/receipts/{uid}.json').read_text())
                    task['crops']=face['crop_paths']
                else:
                    task.update(wav=str(MEDIA_ROOT/row['wav']['path']),input_sha256=row['wav']['sha256'])
                value=child.request(task,180)
                if value['status']!='ok':
                    raise RuntimeError(f'{stage} resource measurement failed: {uid}: {value}')
                measured.append({'utt_id':uid,**value})
        finally:
            child.close()
        write_json_atomic(args.output_dir/(stage+'_receipts.json'),measured)
        summary[stage]={'sample_count':len(measured),'peak_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in measured),
                        'peak_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in measured),
                        'nvidia_smi_observed_max_mib':max(int(r['nvidia_smi_used_mib'].splitlines()[0]) for r in measured),
                        'process_elapsed_s_including_load':time.monotonic()-start,'model':str(model),
                        'receipts':{'path':str(args.output_dir/(stage+'_receipts.json')),'sha256':sha256(args.output_dir/(stage+'_receipts.json'))}}
        write_json_atomic(args.output_dir/'progress.json',summary)
    q=json.loads((r3/'Q/predict_final_et_test/summary.json').read_text())
    summary['Q']={'sample_count':1929,'peak_cuda_allocated_bytes':int(q['peak_cuda_allocated_mib']*2**20),
                   'source':str(r3/'Q/predict_final_et_test/summary.json'),'sha256':sha256(r3/'Q/predict_final_et_test/summary.json')}
    result={'method':'maximum live CUDA allocated memory per isolated serial stage; expert probes use 20 longest ET-test audio clips, Q uses all 1929 formal ET-test clips',
            'limitation':'not a simultaneous-residency memory benchmark; CPU face/T1/heads/fusion allocate no CUDA memory',
            'stages':summary,'elapsed_s_including_load':time.monotonic()-started}
    write_json_atomic(args.output_dir/'summary.json',result)
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),summary_sha256=sha256(args.output_dir/'summary.json'))
    write_json_atomic(args.output_dir/'run_meta.json',meta)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',choices=('A3','A2','V1','V2','SenseVoice'))
    p.add_argument('--model',type=Path)
    for flag in ('manifest','x02-root','x03-root','output-dir'):
        p.add_argument('--'+flag,type=Path)
    args=p.parse_args()
    if args.worker:
        worker(args.worker,args.model)
    else:
        run(args)


if __name__=='__main__':
    main()
