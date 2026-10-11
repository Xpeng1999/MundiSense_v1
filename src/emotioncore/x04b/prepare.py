"""Freeze two stratified MER half-epochs and preserve X03 ET assignments."""
from __future__ import annotations
import argparse, collections, datetime as dt, json, math, random, sys
from pathlib import Path
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import FOLD_SHA, SUBSET_SHA, assignment, read_jsonl, sha256, write_jsonl
from emotioncore.x03.budget_v2 import FACE_SHA
from emotioncore.x04.prepare import require_h, storage_gate

SEED=20261007
MER_LABELS=('neutral','joy','sadness','anger','surprise','worried')

def stratified_half(rows, epoch):
    groups=collections.defaultdict(list)
    for r in rows:
        if r['truth_M'] not in MER_LABELS:raise ValueError('Unknown MER label')
        groups[r['truth_M']].append(r)
    n=len(rows)//2
    counts={c:len(groups[c])//2 for c in sorted(groups)}
    order=sorted(groups,key=lambda c:(-(len(groups[c])%2),c))
    for c in order[:n-sum(counts.values())]:counts[c]+=1
    rng=random.Random(SEED+epoch);selected=[]
    for c in sorted(groups):
        g=sorted(groups[c],key=lambda r:r['utt_id']);rng.shuffle(g);selected+=g[:counts[c]]
    return sorted(selected,key=lambda r:r['utt_id'])

def assert_split(rows, et_ids, mer_ids, forbidden_speakers):
    ids=[r['utt_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate training IDs')
    for r in rows:
        if r['corpus']=='ET':
            if r['utt_id'] not in et_ids or r['speaker_id'] in forbidden_speakers:raise ValueError('ET holdout/inner leakage')
        elif r['corpus']=='MER':
            if r['utt_id'] not in mer_ids or r['split']!='mer_train':raise ValueError('MER-test/excluded leakage')
        else:raise ValueError('CH-SIMS or other corpus in training')

def run(args):
    git=git_provenance(Path.cwd());storage_gate();require_h(args.output)
    for path,digest in [(Path('configs/folds_v1.json'),FOLD_SHA),(Path('configs/x03_inner_val_subsets.json'),SUBSET_SHA),(Path('configs/x03_face_missing_policy.json'),FACE_SHA)]:
        if sha256(path)!=digest:raise ValueError('Frozen protocol changed')
    foldspec=json.loads(Path('configs/folds_v1.json').read_text())
    mer_meta=json.loads((args.r4/'Q_media/summary.json').read_text());maps={}
    for f in ('A','V','AV'):
        spec=mer_meta['outputs']['train_'+f];path=Path(spec['path'])
        if sha256(path)!=spec['sha256']:raise ValueError('MER frozen manifest changed')
        rows=read_jsonl(path);maps[f]={r['utt_id']:r for r in rows}
        if len(maps[f])!=9386 or len(rows)!=9386:raise ValueError('MER coverage')
    if any(set(maps[f])!=set(maps['A']) for f in maps):raise ValueError('MER form IDs mismatch')
    dedup=read_jsonl(args.r4/'manifests/mer_train_dedup.jsonl');mer_ids={r['utt_id'] for r in dedup}
    test_ids={r['utt_id'] for r in read_jsonl(args.r4/'manifests/mer_test.jsonl')}
    if mer_ids!=set(maps['A']) or mer_ids&test_ids:raise ValueError('MER source split leakage')
    args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((args.r3/'manifests/summary.json').read_text());runs={}
    for name in ('fold0','fold1','final'):
        spec=source[name];trainpath=Path(spec['train_path']);valpath=Path(spec['val_path'])
        if sha256(trainpath)!=spec['train_sha256'] or sha256(valpath)!=spec['val_sha256']:raise ValueError('X03 manifests changed')
        et=read_jsonl(trainpath);val=read_jsonl(valpath)
        index=0 if name=='final' else int(name[-1]);fold=foldspec['folds'][index]
        forbidden=set(fold['inner_validation_speakers'])|(set(fold['outer_holdout_speakers']) if name!='final' else set())
        epochs=[]
        for epoch in range(2):
            selected=stratified_half(list(maps['A'].values()),epoch)
            forms=assignment([r['utt_id'] for r in selected],SEED+epoch)
            mixed=[{**r,'corpus':'ET','cache_root':'/mnt/h/emotionAI/emotioncore_models/X03/q_inputs','epoch':epoch} for r in et]
            for r in selected:
                row=maps[forms[r['utt_id']]][r['utt_id']]
                mixed.append({**row,'corpus':'MER','cache_root':str(args.r4/'Q_inputs'),'epoch':epoch})
            for row in mixed:
                receipt=Path(row['cache_root'])/row['utt_id']/(row['effective_form']+'.json')
                if not receipt.is_file():raise FileNotFoundError(f'Frozen cache missing: {receipt}')
            assert_split(mixed,{r['utt_id'] for r in et},mer_ids,forbidden)
            rng=random.Random(20260929+epoch+(0 if name=='final' else index*100));rng.shuffle(mixed)
            path=args.output/f'{name}_epoch{epoch}.jsonl';digest=write_jsonl(path,mixed)
            counts=dict(collections.Counter(r['truth_M'] for r in mixed if r['corpus']=='MER'))
            epochs.append({'path':str(path),'sha256':digest,'n':len(mixed),'ET_n':len(et),'MER_n':4693,'MER_seed':SEED+epoch,'MER_classes':counts,'assigned_forms':dict(collections.Counter(r['assigned_form'] for r in mixed)),'fallbacks':sum(r['face_missing'] and r['assigned_form']!='A' for r in mixed)})
        runs[name]={'epochs':epochs,'epoch_n':len(mixed),'train_count':2*len(mixed),'train_sha256':sha256(args.output/f'{name}_epoch0.jsonl')+sha256(args.output/f'{name}_epoch1.jsonl'),'val_path':str(valpath),'val_sha256':spec['val_sha256'],'val_count':len(val),'ET_source':spec}
    meta={'git':git,'command':sys.argv,'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'folds_sha256':FOLD_SHA,'subsets_sha256':SUBSET_SHA,'face_policy_sha256':FACE_SHA,'runs':runs,'MER_source':mer_meta['outputs'],'initialization':'base only','epoch_indexing':'zero based: seeds 20261007,20261008','stratification':'largest remainder, class-name tie break, exact4693','shuffle':'X03 seed formula after concatenating frozen ET and sorted sampled MER'}
    write_json_atomic(args.output/'run_meta.json',meta)
    write_json_atomic(args.output/'summary.json',runs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--r3',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X03/Q'));p.add_argument('--r4',type=Path,default=Path('/mnt/h/emotionAI/emotioncore_runs/X04a'));p.add_argument('--output',type=Path,required=True);run(p.parse_args())
if __name__=='__main__':main()
