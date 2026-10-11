"""CPU-only D1_fmhx target probabilities from existing frozen feature packs."""
import argparse
import numpy as np
from emotioncore.x02 import head_h1
from emotioncore.x04 import targets as old
from emotioncore.x04.prepare import write_rows
from emotioncore.x05.heads import load_pack
from emotioncore.x05.common import (ROOT,R2,R4,MANIFEST,FMH,read,read_jsonl,spec,start,finish)


def run(encoder):
    import torch
    job=ROOT/f'heads/D1_fmhx/{encoder}';output=job/'targets';meta=start(output)
    training=read(job/'summary.json')
    if not training['passed'] or training['inputs']['fmh_overlap']:
        raise ValueError('STOP invalid D1_fmhx training receipt')
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    checkpoint_path=job/'final_head.pt'
    if spec(checkpoint_path)!=training['checkpoint']:
        raise ValueError('STOP D1_fmhx checkpoint changed')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    et=read_jsonl(MANIFEST);cv=head_h1.load_rows(MANIFEST,read('configs/folds_v1.json'))
    mer=read_jsonl(R4/'manifests/mer_test.jsonl')
    sims=old.load_sims_rows(R2/'sims_valid/sims_valid_adapter.jsonl',
                           __import__('pathlib').Path('experiments/X01_benchmark/results/sims_valid.jsonl'))
    cohorts={'fmh':([r for r in cv if r['speaker_id'] in FMH],R2/f'packs/{encoder}/pack_meta.json'),
             'et_test':([r for r in et if r['official_split']=='test'],R4/f'target_packs/et_test/{encoder}/pack_meta.json'),
             'mer_test':(mer,R4/f'target_packs/mer_test/{encoder}/pack_meta.json'),
             'sims_valid':(sims,R4/f'target_packs/sims_valid/{encoder}/pack_meta.json')}
    meta.update(checkpoint=spec(checkpoint_path),device='cpu',analysis_performed=False)
    outputs={}
    for split,(rows,pack_path) in cohorts.items():
        if len(rows)!={'fmh':3266,'et_test':1929,'mer_test':572,'sims_valid':647}[split]:
            raise ValueError('STOP incomplete target/FMH coverage')
        frames,lengths,pack_spec=load_pack(pack_path,None if split=='fmh' else [r['utt_id'] for r in rows])
        indices=np.asarray([i for i,r in enumerate(cv) if r['speaker_id'] in FMH]) if split=='fmh' else np.arange(len(rows))
        data=old.Frames([frames],[lengths]);model=old.build_model(data.dimension,data.layered,'cpu')
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        raw=np.full((len(rows),8),1/8,dtype=np.float32)
        with torch.inference_mode():
            for positions in old.batches(np.flatnonzero(lengths[indices]>0)):
                x,n=data.batch(indices[positions],'cpu')
                raw[positions]=model(x,n).softmax(1).cpu().numpy()
        p=raw[:,:7]/raw[:,:7].sum(1,keepdims=True);records=[]
        for i,row in enumerate(rows):
            r={'utt_id':row['utt_id'],'speaker_id':row['speaker_id'],
               'probabilities':p[i].astype(float).tolist(),'probabilities8':raw[i].astype(float).tolist(),
               'status':'ok' if lengths[indices[i]] else 'failed',
               'feature_available':bool(lengths[indices[i]])}
            if split in ('fmh','et_test'):r.update(truth_M=row['labels']['M'],fold=None)
            if split=='mer_test':r.update(truth_M=row['label'],corpus='MER')
            if split=='sims_valid':r.update(label=row['label'],audio_missing=bool(row.get('audio_missing',False)))
            records.append(r)
        path=output/f'{split}.jsonl';write_rows(path,records)
        outputs[split]={**spec(path),'n':len(rows),'pack':pack_spec}
        del model,data,frames
    finish(output,meta,{'artifacts':outputs,'analysis_performed':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--encoder',choices=('A3','V2','V1'),required=True)
    run(p.parse_args().encoder)
