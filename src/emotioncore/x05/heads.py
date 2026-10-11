"""Replay X04a final training, then train only the authorized D1_fmhx heads."""
import argparse
import os
from pathlib import Path

import numpy as np
from emotioncore.x02 import head_h1
from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x04 import heads as old
from emotioncore.x04.prepare import write_rows
from emotioncore.x05.common import (R2, R3, R4, R4B, ROOT, DEDUP, FMH, MANIFEST,
    FOLD_SHA, read, read_jsonl, sha256, spec, start, finish)


def select_training(cv, extra, excluded, reproduce=False):
    """Exact original order, with only the authorized ET speaker exclusion."""
    et = [i for i, r in enumerate(cv) if reproduce or r['speaker_id'] not in FMH]
    mer = [len(cv)+i for i, r in enumerate(extra)
           if r['corpus'] == 'MER' and r['official_split'] == 'train'
           and r['utt_id'] not in excluded]
    rows = cv+extra
    indices = np.asarray(et+mer, dtype=int)
    old.validate_training(rows, indices, [] if reproduce else FMH, [], excluded, stage='final')
    for i in indices:
        r = rows[i]
        if r['corpus'] not in ('ET', 'MER'):
            raise ValueError('STOP unapproved training corpus')
        if r['corpus'] == 'ET' and not reproduce and r['speaker_id'] in FMH:
            raise ValueError('STOP FMH training overlap')
    return rows, indices


def load_pack(path, ids=None):
    metadata = read(path)
    if ids is not None and metadata.get('ids') != ids:
        raise ValueError('STOP frozen extra feature order differs')
    for key in ('frame', 'lengths'):
        if sha256(Path(metadata[key+'_file'])) != metadata[key+'_sha256']:
            raise ValueError('STOP feature pack SHA changed')
    return (np.load(metadata['frame_file'], mmap_mode='r'),
            np.load(metadata['lengths_file']), spec(path))


def load_data(encoder, reproduce):
    folds = read('configs/folds_v1.json')
    cv = head_h1.load_rows(MANIFEST, folds)
    for row in cv:
        row.update(corpus='ET', label=row['labels']['M'])
    extra_path = R4 / 'manifests/all_features.jsonl'
    budget = read(R4 / 'budget_preflight.json')
    if (not budget['passed'] or sha256(extra_path) != budget['all_features_manifest_sha256']
            or sha256(DEDUP) != budget['dedup_report_sha256']):
        raise ValueError('STOP original D1 data provenance changed')
    extra = read_jsonl(extra_path)
    excluded = set(read(DEDUP)['excluded_ids'])
    rows, indices = select_training(cv, extra, excluded, reproduce)
    et_ids = {rows[i]['utt_id'] for i in indices if rows[i]['corpus']=='ET'}
    mer_ids = {rows[i]['utt_id'] for i in indices if rows[i]['corpus']=='MER'}
    expected_n = 16366 if reproduce else 13100
    if len(et_ids) != expected_n or len(mer_ids) != 9386:
        raise ValueError('STOP training coverage differs from frozen D1')
    # D1 MER membership must equal the saved X04a training receipt exactly.
    old_ids = read_jsonl(R4 / f'heads/D1/{encoder}/fold0/training_ids.jsonl')
    if mer_ids != {r['utt_id'] for r in old_ids if r['corpus']=='MER'}:
        raise ValueError('STOP D1 MER training membership changed')
    if not reproduce:
        et_final = read_jsonl(R3 / 'Q/manifests/final_train.jsonl')
        mix_final = read_jsonl(R4B / 'manifests/final_epoch0.jsonl')
        for q in (et_final, [r for r in mix_final if r['corpus']=='ET']):
            if et_ids != {r['utt_id'] for r in q} or FMH & {r['speaker_id'] for r in q}:
                raise ValueError('STOP final Q and D1_fmhx ET membership differs')
    ep = R2 / f'packs/{encoder}/pack_meta.json'
    ep_meta = read(ep)
    if ep_meta['folds_sha256'] != FOLD_SHA or ep_meta['manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('STOP frozen ET pack provenance changed')
    ef, el, es = load_pack(ep)
    nf, nl, ns = load_pack(R4 / f'packs/{encoder}/pack_meta.json', [r['utt_id'] for r in extra])
    if len(ef) != len(cv) or len(nf) != len(extra):
        raise ValueError('STOP feature pack coverage differs')
    return old.Frames([ef, nf], [el, nl]), rows, indices, {
        'et_pack': es, 'extra_pack': ns, 'extra_manifest': spec(extra_path), 'dedup': spec(DEDUP),
        'ET_count': len(et_ids), 'MER_count': len(mer_ids),
        'ET_speakers': sorted({rows[i]['speaker_id'] for i in indices if rows[i]['corpus']=='ET'}),
        'fmh_overlap': len(FMH & {rows[i]['speaker_id'] for i in indices if rows[i]['corpus']=='ET'})}


def reproduce_et_target(checkpoint, output):
    """ET-only wrapper around the unchanged X04a target forward functions."""
    import torch
    from emotioncore.x04 import targets
    rows = [r for r in read_jsonl(MANIFEST) if r['official_split']=='test']
    frames, lengths, _ = load_pack(R4 / 'target_packs/et_test/A3/pack_meta.json',
                                   [r['utt_id'] for r in rows])
    data = targets.Frames([frames], [lengths])
    model = targets.build_model(data.dimension, data.layered, 'cpu')
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    raw = np.full((len(rows), 8), 1/8, dtype=np.float32)
    with torch.inference_mode():
        for indices in targets.batches(np.flatnonzero(lengths>0)):
            x, n = data.batch(indices, 'cpu')
            raw[indices] = model(x, n).softmax(1).cpu().numpy()
    reference_path = R4 / 'heads/D1/A3/targets/et_test.jsonl'
    reference = {r['utt_id']:r for r in read_jsonl(reference_path)}
    if len(rows)!=1929 or {r['utt_id'] for r in rows}!=set(reference):
        raise ValueError('STOP reproduction ET-test coverage incomplete')
    correct = [int(raw[i,:7].argmax()) == int(np.argmax(reference[r['utt_id']]['probabilities']))
               for i,r in enumerate(rows)]
    probs = raw[:,:7]/raw[:,:7].sum(1,keepdims=True)
    target_path = output/'et_test.jsonl'
    write_rows(target_path, [{'utt_id':r['utt_id'],'probabilities':p.astype(float).tolist()}
                            for r,p in zip(rows,probs)])
    return {'n': len(rows), 'argmax_agreement': float(np.mean(correct)),
            'reference': spec(reference_path), 'predictions': spec(target_path),
            'target_implementation': 'unchanged emotioncore.x04.targets Frames/build_model/batches/forward',
            'label_metrics_computed': False}


def run(encoder, reproduce=False):
    import torch
    if reproduce and encoder != 'A3':
        raise ValueError('Only the A3 reproduction is authorized')
    if encoder == 'A3' and os.environ.get('X05_GPU_GUARDED') != ('reproduce_a3' if reproduce else 'fmhx_a3'):
        raise RuntimeError('Use the X05 GPU watchdog')
    if not reproduce and not read(ROOT/'reproduce_A3/summary.json')['passed']:
        raise RuntimeError('STOP A3 reproduction has not passed')
    output = ROOT / ('reproduce_A3' if reproduce else f'heads/D1_fmhx/{encoder}')
    meta = start(output)
    device = 'cuda:0' if encoder=='A3' else 'cpu'
    torch.set_num_threads(8 if encoder=='A3' else 4); torch.set_num_interop_threads(1)
    result_path = R4/f'heads/D1/{encoder}/results.json'
    config = read(result_path)['final_config']
    data, rows, indices, inputs = load_data(encoder, reproduce)
    meta.update(encoder=encoder, device=device, arm='D1_reproduction' if reproduce else 'D1_fmhx',
        seed=head_h1.SEED, final_config=config, original_results=spec(result_path), inputs=inputs,
        validation_used=False, hyperparameter_search=False, Q_training=False,
        original_training_source=spec(Path('src/emotioncore/x04/heads.py')))
    write_json_atomic(output/'run_meta.json',meta)
    training_path = output/'training_ids.jsonl'
    write_rows(training_path,[{k:rows[i][k] for k in ('utt_id','corpus','speaker_id','official_split')}
                              for i in indices])
    model, receipt = old.train(data, rows, indices, None, config, head_h1.SEED, device,
                               fixed_epochs=config['best_epoch'])
    checkpoint={'state_dict':model.state_dict(),'encoder':encoder,'arm':'D1',
                'head':'h2_masked8','config':config}
    path=output/'final_head.pt'; torch.save(checkpoint,path)
    result={'training':receipt,'checkpoint':spec(path),'training_ids':spec(training_path),
            'inputs':inputs,'config':config,'seed':head_h1.SEED,'passed':True}
    if reproduce:
        reference_path=R4/'heads/D1/A3/final_head.pt'
        reference=torch.load(reference_path,map_location='cpu',weights_only=False)['state_dict']
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        equal=set(state)==set(reference) and all(torch.equal(state[k],reference[k]) for k in state)
        result.update(reference_checkpoint=spec(reference_path),weights_equal=equal)
        if not equal:
            result['et_test_reproduction']=reproduce_et_target({'state_dict':state},output)
            result['passed']=result['et_test_reproduction']['argmax_agreement']>=.995
    finish(output,meta,result)
    if not result['passed']:
        raise RuntimeError('STOP A3 reproduction failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--encoder',choices=('A3','V2','V1'),required=True)
    parser.add_argument('--reproduce',action='store_true')
    args=parser.parse_args(); run(args.encoder,args.reproduce)
