"""Complete the prescribed D0 OOF check while other heads are training."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x04.analysis import d0_reproduction
from emotioncore.x04.heads import FOLDS_SHA
from emotioncore.x04.pipeline import completed_head
from emotioncore.x04.prepare import require_h, storage_gate


def run(args):
    git=git_provenance(Path.cwd());require_h(args.output);storage_gate()
    folds_path=Path('configs/folds_v1.json')
    if sha256(folds_path)!=FOLDS_SHA:raise ValueError('Frozen folds changed')
    folds=json.loads(folds_path.read_text())
    cv={r['utt_id']:r for r in read_jsonl(Path('experiments/X01_benchmark/results/et_utterances.jsonl'))
        if r['speaker_id'] in folds['cv_pool_speakers']}
    if len(cv)!=16366:raise ValueError('STOP CV count differs')
    expected={s:f['fold'] for f in folds['folds'] for s in f['outer_holdout_speakers']}
    meta={'git':git,'command':sys.argv,'started_at':dt.datetime.now(dt.timezone.utc).isoformat(),
          'device':'CPU','Q_training':False,'model_inference':False,'folds_sha256':FOLDS_SHA,'sources':{}}
    args.output.mkdir(parents=True,exist_ok=True);write_json_atomic(args.output/'run_meta.json',meta)
    reproduction={}
    for encoder in ('A3','V2','V1'):
        job=args.derived/f'heads/D0/{encoder}'
        if not completed_head(job):raise ValueError('D0 head not yet complete')
        fresh=read_jsonl(job/'oof.jsonl')
        if any(r['utt_id'] not in cv or r['speaker_id']!=cv[r['utt_id']]['speaker_id']
               or r['truth_M']!=cv[r['utt_id']]['labels']['M']
               or r['fold']!=expected[r['speaker_id']] for r in fresh):
            raise ValueError('STOP D0 CV membership/speaker/label/fold mismatch')
        reproduction[encoder]=d0_reproduction(fresh,args.r2/f'heads/{encoder}_h2_cv/oof.jsonl',cv)
        meta['sources'][encoder]={'oof_path':str(job/'oof.jsonl'),'oof_sha256':sha256(job/'oof.jsonl'),
            'run_meta_path':str(job/'run_meta.json'),'run_meta_sha256':sha256(job/'run_meta.json')}
    write_json_atomic(args.output/'d0_reproducibility.json',reproduction)
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                result_sha256=sha256(args.output/'d0_reproducibility.json'))
    write_json_atomic(args.output/'run_meta.json',meta)


def main():
    p=argparse.ArgumentParser();p.add_argument('--derived',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--r2',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X02'))
    run(p.parse_args())


if __name__=='__main__':main()
