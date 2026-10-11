"""Same registered VAD grid, parallelized by recording with immutable CPU caches."""
import argparse
import concurrent.futures
import math
import multiprocessing as mp
import os
import signal as signals
import time
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.aishell import read_channel_zero, reference, candidates
from emotioncore.x06.vad import Posterior, segments, mask_segments, intervals_from_mask, boundary_counts, prf

MODELS = None


def initialize():
    global MODELS
    MODELS = {name: Posterior(name) for name in ('fsmn', 'silero')}


def one(task):
    annotation, reuse = task
    audio_path = annotation.parent.parent / 'wav' / (annotation.stem + '.flac')
    output = ROOT / 'aishell_train_parallel'
    posteriors = {}; receipts = {}; elapsed = {}
    if reuse:
        with np.load(ROOT / 'aishell_train/posteriors' / f'{annotation.stem}_fsmn.npz') as cache:
            n = len(cache['probability'])
        truth, valid = reference(annotation, n)
        for name in MODELS:
            path = ROOT / 'aishell_train/posteriors' / f'{annotation.stem}_{name}.npz'
            with np.load(path) as cache:
                assert np.array_equal(cache['truth'], truth) and np.array_equal(cache['valid'], valid)
                posteriors[name] = cache['probability'].copy()
            receipts[name] = spec(path); elapsed[name] = 0.
    else:
        audio = read_channel_zero(audio_path)
        truth, valid = reference(annotation, math.ceil(len(audio) / 160))
        for name, model in MODELS.items():
            tic = time.monotonic()
            posteriors[name] = model.score(audio, valid)
            elapsed[name] = time.monotonic() - tic
            path = output / 'posteriors' / f'{annotation.stem}_{name}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, probability=posteriors[name], truth=truth, valid=valid)
            receipts[name] = spec(path)
    target = intervals_from_mask(truth & valid)
    values = []
    for config in candidates():
        found = segments(posteriors[config['model']], config['threshold'], config['min_silence'], valid)
        mask = mask_segments(found, len(truth))
        values.append({'config': config,
            'frame_counts': [int((mask & truth & valid).sum()), int((mask & ~truth & valid).sum()), int((~mask & truth & valid).sum())],
            'boundary_counts': boundary_counts(found, target).tolist(), 'segment_count': len(found),
            'segment_seconds': sum(r['end_s'] - r['start_s'] for r in found),
            'decision_delay_seconds': sum(r['decision_s'] - r['end_s'] for r in found)})
    result = {'rttm': spec(annotation), 'audio': spec(audio_path), 'reused_cpu_cache': reuse,
        'posteriors': receipts, 'inference_seconds': elapsed, 'values': values,
        'eligible_10ms_frames': int(valid.sum()), 'overlap_frames_excluded': int((~valid).sum())}
    write_json_atomic(output / 'recordings' / f'{annotation.stem}.json', result)
    return result


def run(replace_pid):
    output = ROOT / 'aishell_train_parallel'
    meta = start(output)
    assert_tuning_source('aishell_train')
    original = ROOT / 'aishell_train'
    if (original / 'freeze.json').exists():
        raise RuntimeError('Do not repeat a completed selection')
    args = Path(f'/proc/{replace_pid}/cmdline').read_bytes().split(b'\0')
    if b'emotioncore.x06.aishell' not in args or b'train' not in args:
        raise RuntimeError('Replacement PID is not the known serial CPU selection')
    os.kill(replace_pid, signals.SIGTERM)
    for _ in range(100):
        if not Path(f'/proc/{replace_pid}').exists():
            break
        time.sleep(.1)
    if Path(f'/proc/{replace_pid}').exists():
        raise RuntimeError('Serial job did not exit; do not run concurrently')
    completed = read(original / 'progress.json')['done']
    old_meta = read(original / 'run_meta.json')
    finish(original, old_meta, {'status': 'superseded_cpu_execution', 'completed_recordings': completed,
        'selection_performed': False, 'replacement': str(output), 'reason': 'parallelize unchanged per-recording inference',
        'gpu_seconds': 0})
    meta['serial_cache_run'] = spec(original / 'run_meta.json')
    meta['reused_completed_prefix'] = completed
    meta['workers'] = 8
    write_json_atomic(output / 'run_meta.json', meta)
    annotations = [p for d in ('train_S', 'train_M', 'train_L')
                   for p in sorted((AISHELL / d / 'TextGrid').glob('*.rttm'))]
    assert len(annotations) == 130 and len({p.stem for p in annotations}) == 130
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        futures = [pool.submit(one, (path, i < completed)) for i, path in enumerate(annotations)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            storage_gate()
            write_json_atomic(output / 'progress.json', {'done': len(results), 'total': 130,
                                'reused_cpu_cache_n': sum(r['reused_cpu_cache'] for r in results)})
    order = {str(path): i for i, path in enumerate(annotations)}
    results.sort(key=lambda r: order[r['rttm']['path']])
    values = []
    for i, config in enumerate(candidates()):
        entries = [r['values'][i] for r in results]
        count = sum(r['segment_count'] for r in entries)
        values.append({'config': config, 'frame': prf(np.sum([r['frame_counts'] for r in entries], axis=0)),
            'boundary': prf(np.sum([r['boundary_counts'] for r in entries], axis=0)), 'segment_count': count,
            'mean_segment_s': sum(r['segment_seconds'] for r in entries) / max(1, count),
            'mean_endpoint_delay_s': sum(r['decision_delay_seconds'] for r in entries) / max(1, count)})
    best = max(r['frame']['f1'] for r in values)
    chosen = min((r for r in values if r['frame']['f1'] >= best - .005),
                 key=lambda r: (r['config']['min_silence'], r['mean_endpoint_delay_s'], abs(r['config']['threshold'] - .5), r['config']['model']))
    write_json_atomic(output / 'freeze.json', {'selected': chosen['config'], 'training_source': 'AISHELL-4 train only',
        'selection': 'highest 10ms speech F1; within 0.005 prefer smaller minimum silence and lower endpoint delay',
        'exact_tie': 'threshold closest to 0.5, then model name', 'min_speech_s': .3, 'max_speech_s': 15,
        'postprocessing': 'shared registered probability-threshold state machine; overlap resets state',
        'model_artifacts': spec(ROOT / 'vad_models/summary.json')})
    write_json_atomic(output / 'sources.json', {'recordings': sorted(results, key=lambda r: r['rttm']['path'])})
    finish(output, meta, {'split': 'train', 'recordings': 130, 'channel': 0, 'results': values,
                          'sources': spec(output / 'sources.json'), 'gpu_seconds': 0, 'workers': 8,
                          'reused_completed_prefix': completed})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--replace-pid', type=int, required=True)
    run(parser.parse_args().replace_pid)
