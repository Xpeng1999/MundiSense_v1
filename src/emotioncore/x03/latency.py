"""Serial stage accounting on ET-test; never fit parameters or score new labels.

Frozen encoder receipts include input reading, preprocessing, forward and output
writing. Q receipts omit its processor, which is timed here in an isolated CPU
worker. Models may be reused across conditions; face preparation is counted once.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.x02 import head_h1, head_h2
from emotioncore.x02.final_eval import load_test_frames
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic
from emotioncore.x03.fusion import calibrated_logits, fusion_logits, softmax
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MEDIA_ROOT


def serial_stages(components: list[str]) -> list[str]:
    stages = []
    if any(c.startswith('V') or c == 'Q_AV' for c in components):
        stages.append('face')
    if 'Q_AV' in components:
        stages += ['Q_video_prepare', 'Q_AV_processor', 'Q_AV_forward']
    if 'Q_A' in components:
        stages += ['Q_A_processor', 'Q_A_forward']
    for name in ('A3', 'A2', 'V2', 'V1'):
        if any(c.startswith(name + '_') for c in components):
            stages += [name, name + '_head']
    if 'T1_r' in components:
        stages += ['SenseVoice', 'T1']
    return stages


def run(args):
    import torch
    from transformers import AutoModelForSequenceClassification

    started = time.monotonic()
    git = git_provenance(Path.cwd())
    r2, r3 = args.x02_root, args.x03_root
    rows = [r for r in read_jsonl(args.manifest) if r['official_split'] == 'test']
    if len(rows) != 1929 or len({r['utt_id'] for r in rows}) != 1929:
        raise ValueError('ET-test coverage changed')
    torch.set_num_threads(4)
    missing = set(json.loads(Path('configs/x03_face_missing_policy.json').read_text())['face_missing_utt_ids'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {'git': git, 'command': sys.argv, 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'manifest_sha256': sha256(args.manifest), 'cuda': False, 'threads': 4}
    write_json_atomic(args.output_dir / 'run_meta.json', meta)
    heads = {}
    for name, encoder in (('A3_h2', 'A3'), ('A2_h2', 'A2'), ('V2_h2', 'V2')):
        checkpoint = r2 / 'final' / name / 'final_head.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        frames, lengths = load_test_frames(rows, r2 / 'features' / encoder / 'receipts', encoder)
        model = head_h2.build_model(frames.shape[-1], encoder == 'A3', 'cpu').eval()
        model.load_state_dict(ck['state_dict'])
        heads[encoder] = (model, frames, lengths)
    ck = torch.load(r2 / 'final/V1_h1/final_head.pt', map_location='cpu', weights_only=False)
    vectors, available = head_h1.load_features(rows, r2 / 'features/V1/receipts', 'V1')
    v1 = torch.nn.Linear(vectors.shape[1], 7).eval()
    v1.load_state_dict(ck['state_dict'])
    normalized = (vectors - ck['mean']) / ck['std']
    t1_ck = torch.load(r2 / 'heads/T1_cv/final_model.pt', map_location='cpu', weights_only=False)
    t1 = AutoModelForSequenceClassification.from_pretrained(str(r2 / 'models/T1'), num_labels=7,
                                                           local_files_only=True).eval()
    t1.load_state_dict(t1_ck['state_dict'])
    with np.load(r2 / 'tokens/T1_r_test.npz') as archive:
        ids, masks = archive['input_ids'].copy(), archive['attention_mask'].copy()
        failed = archive['failed'].copy()
        if list(map(str, archive['utt_ids'])) != [r['utt_id'] for r in rows]:
            raise ValueError('T1 token order mismatch')
    q_raw = {form: {r['utt_id']: r for r in read_jsonl(r3 / f'Q/predict_final_et_test/{form}.jsonl')}
             for form in ('A', 'AV')}
    e_best = json.loads((r3 / 'E1_v4/summary.json').read_text())['best_e1']
    h_best = json.loads((r3 / 'H/summary.json').read_text())['best_h']
    roots = {'E_best': r3 / 'E1_v4' / e_best, 'Q_AV': r3 / 'Q_calibrated/Q_AV',
             'Q_A': r3 / 'Q_calibrated/Q_A', 'Q_fusion': r3 / 'H/Q_fusion',
             'H_best': r3 / 'H' / h_best, 'A3_reference': r3 / 'E1_v4/E1_A3',
             'V2_reference': r3 / 'V2_reference'}
    specs = {}
    for name, root in roots.items():
        metric = json.loads((root / 'metrics.json').read_text())
        components = metric.get('components', [name])
        if name == 'V2_reference':
            components = ['V2_h2']
        param_path = root / 'parameters_final.json'
        params = json.loads(param_path.read_text()) if param_path.exists() else metric.get('parameters_final')
        specs[name] = (list(components), params)
    # Raw frozen probabilities used only to time already-fitted postprocessing.
    from emotioncore.x03.h_fusion import stack_inputs
    from emotioncore.x03.fusion import load_sources
    probability = {}
    for name, (components, params) in specs.items():
        if params and 'fusion' in params:
            loader = stack_inputs if any(c.startswith('Q_') for c in components) else load_sources
            if loader is stack_inputs:
                uids, logp, _, _ = loader(tuple(components), 'et_test', r3 / 'Q', args.artifact_index, missing)
            else:
                uids, logp, _, _ = loader(args.artifact_index, tuple(components), 'et_test', missing)
            probability[name] = dict(zip(uids, logp, strict=True))
        else:
            form = 'AV' if name == 'Q_AV' else 'A' if name == 'Q_A' else None
            if form:
                probability[name] = {uid: np.log(np.clip(row['probabilities'], 1e-12, 1))[None]
                                     for uid, row in q_raw[form].items()}
            else:
                uids, logp, _, _ = load_sources(args.artifact_index, ('V2_h2',), 'et_test', missing)
                probability[name] = dict(zip(uids, logp, strict=True))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    child = PersistentJSONWorker([sys.executable, '-u', '-m', 'emotioncore.x03.q_smoke', '--worker'],
                                 cwd=Path.cwd(), env=env, stderr_path=args.output_dir / 'processor_errors.log')

    def processor(index, form):
        row = rows[index]
        uid = row['utt_id']
        effective = q_raw[form][uid]['effective_form']
        media = json.loads((r3 / f'media/receipts/{uid}.json').read_text())
        task = {'condition': effective, 'wav': str(MEDIA_ROOT / row['wav']['path']),
                'wav_sha256': row['wav']['sha256'], 'output': str(args.output_dir / 'scratch_input.npz')}
        if effective == 'AV':
            task.update(video=media['output'], video_sha256=media['output_sha256'])
        start = time.monotonic()
        result = child.request(task, 120)
        elapsed = time.monotonic() - start
        if result['status'] != 'ok':
            raise RuntimeError(f'Q latency processor failed: {uid}: {result}')
        return elapsed

    def head(index, name):
        start = time.monotonic()
        if name == 'V1':
            if available[index] and rows[index]['utt_id'] not in missing:
                torch.softmax(v1(torch.from_numpy(normalized[index:index+1]).float()), -1).numpy()
        elif name == 'T1':
            if not failed[index]:
                # Same padded 128-token input as frozen T1 classifier inference.
                torch.softmax(t1(input_ids=torch.from_numpy(ids[index:index+1].astype(np.int64)),
                                 attention_mask=torch.from_numpy(masks[index:index+1].astype(np.int64))).logits, -1).numpy()
        else:
            model, frames, lengths = heads[name]
            if lengths[index] and not (name.startswith('V') and rows[index]['utt_id'] in missing):
                x, n = head_h2.tensor_batch(frames, lengths, np.array([index]), 'cpu')
                torch.softmax(model(x, n), -1).numpy()
        return time.monotonic() - start

    timings = []
    try:
        with torch.inference_mode():
            for index in range(8):
                for form in ('A', 'AV'):
                    processor(index, form)
                for name in (*heads, 'V1', 'T1'):
                    head(index, name)
            with (args.output_dir / 'et_test_latency.jsonl').open('w') as stream:
                for index, row in enumerate(rows):
                    uid = row['utt_id']
                    stages = {}
                    for name, folder in [('face', 'face'), ('SenseVoice', 'asr/SenseVoiceSmall')]+[(n, f'features/{n}') for n in ('A3','A2','V2','V1')]:
                        receipt = json.loads((r2 / folder / 'receipts' / (uid + '.json')).read_text())
                        if receipt['id'] != uid or receipt.get('input_sha256') != row['wav' if name.startswith('A') or name=='SenseVoice' else 'mp4']['sha256']:
                            raise ValueError('Latency source provenance mismatch: ' + uid)
                        stages[name] = float(receipt['elapsed_s'])
                        if name.startswith('V') and uid in missing:
                            stages[name] = 0.0
                    stages['Q_video_prepare'] = json.loads((r3 / f'media/receipts/{uid}.json').read_text())['elapsed_s']
                    for form in ('A','AV'):
                        stages[f'Q_{form}_processor'] = processor(index, form)
                        stages[f'Q_{form}_forward'] = q_raw[form][uid]['latency_s']
                    for name in (*heads, 'V1', 'T1'):
                        stages[name if name=='T1' else name+'_head'] = head(index, name)
                    totals = {}
                    for name, (components, params) in specs.items():
                        start = time.monotonic()
                        logits = probability[name][uid]
                        if params and 'fusion' in params:
                            logits = fusion_logits(logits[None], np.array(params['fusion']))
                        if params and 'calibration' in params:
                            logits = calibrated_logits(logits, np.array(params['calibration']))
                            logits[:, 0] += params['neutral_bias']
                        softmax(logits)
                        post_s = time.monotonic() - start
                        totals[name] = sum(stages[s] for s in serial_stages(components)) + post_s
                        stages[name+'_post'] = post_s
                    item = {'utt_id': uid, 'stages': stages, 'serial_totals': totals}
                    timings.append(item)
                    stream.write(json.dumps(item)+'\n'); stream.flush()
                    if (index+1) % 100 == 0:
                        write_json_atomic(args.output_dir / 'progress.json', {'completed': index+1, 'total': len(rows)})
    finally:
        child.close()
    summary = {'method': 'serial sum of frozen per-utterance encoder/ASR/face/Q-forward receipts plus isolated CPU Q processor, CPU classifier heads and frozen postprocessing',
               'count': len(rows), 'model_loading_excluded': True, 'warmup': 8,
               'limitation': 'stage-accounted steady-state latency, not a co-resident deployment wall-clock benchmark; cache output serialization and IPC included',
               'devices': 'face/Q processor/T1/classifier heads/fusion CPU; frozen encoders/ASR/Q forward CUDA',
               'systems': {name: {'p50_s': float(np.percentile([r['serial_totals'][name] for r in timings],50)),
                                  'p95_s': float(np.percentile([r['serial_totals'][name] for r in timings],95)),
                                  'stages': serial_stages(components)+[name+'_post']}
                           for name,(components,_) in specs.items()},
               'sample_timings': {'path': str(args.output_dir / 'et_test_latency.jsonl'), 'sha256': sha256(args.output_dir / 'et_test_latency.jsonl')}}
    write_json_atomic(args.output_dir / 'summary.json', summary)
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
                summary_sha256=sha256(args.output_dir/'summary.json'))
    write_json_atomic(args.output_dir / 'run_meta.json', meta)


def main():
    p=argparse.ArgumentParser()
    for flag in ('manifest','x02-root','x03-root','artifact-index','output-dir'):
        p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
