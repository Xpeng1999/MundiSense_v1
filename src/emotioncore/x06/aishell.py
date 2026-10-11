"""AISHELL-4 train-only VAD selection, followed by a frozen test report."""
import argparse
import math
import time
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.vad import Posterior, THRESHOLDS, SILENCES, STEP, segments, mask_segments, intervals_from_mask, boundary_counts, prf


def read_channel_zero(path):
    import soundfile as sf
    import scipy.signal
    pieces = []
    with sf.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        for block in audio.blocks(blocksize=sample_rate * 60, dtype='float32', always_2d=True):
            pieces.append(block[:, 0].copy())
    signal = np.concatenate(pieces)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        signal = scipy.signal.resample_poly(signal, 16000 // divisor, sample_rate // divisor)
    return np.asarray(signal, dtype=np.float32)


def reference(path, n):
    by_speaker = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != 'SPEAKER':
            continue
        start, duration = float(fields[3]), float(fields[4])
        a, b = max(0, math.ceil(start / STEP - .5)), min(n, math.ceil((start + duration) / STEP - .5))
        by_speaker.setdefault(fields[7], np.zeros(n, dtype=bool))[a:b] = True
    active = np.zeros(n, dtype=np.int16)
    for speaker in by_speaker.values():
        active += speaker
    return active == 1, active <= 1


def candidates():
    return [{'model': model, 'threshold': threshold, 'min_silence': silence}
            for model in ('silero', 'fsmn') for threshold in THRESHOLDS for silence in SILENCES]


def run(split):
    output = ROOT / f'aishell_{split}'
    meta = start(output)
    if split == 'train':
        assert_tuning_source('aishell_train')
        configs = candidates()
    else:
        configs = [read(ROOT / 'aishell_train_parallel/freeze.json')['selected']]
    models = {name: Posterior(name) for name in sorted({c['model'] for c in configs})}
    states = [{'config': c, 'frame_counts': np.zeros(3, dtype=np.int64),
               'boundary_counts': np.zeros(3, dtype=np.int64), 'segment_count': 0,
               'segment_seconds': 0., 'decision_delay_seconds': 0.} for c in configs]
    directories = ('train_S', 'train_M', 'train_L') if split == 'train' else ('test',)
    annotations = [p for d in directories for p in sorted((AISHELL / d / 'TextGrid').glob('*.rttm'))]
    expected = 130 if split == 'train' else 20
    assert len(annotations) == expected
    sources = []; inference = {name: 0. for name in models}
    for index, annotation in enumerate(annotations):
        audio_path = annotation.parent.parent / 'wav' / (annotation.stem + '.flac')
        signal = read_channel_zero(audio_path)
        truth, valid = reference(annotation, math.ceil(len(signal) / 160))
        target_segments = intervals_from_mask(truth & valid)
        posteriors = {}
        for name, model in models.items():
            tic = time.monotonic(); probability = model.score(signal, valid)
            inference[name] += time.monotonic() - tic
            path = output / 'posteriors' / f'{annotation.stem}_{name}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, probability=probability, valid=valid, truth=truth)
            posteriors[name] = probability
        for state in states:
            config = state['config']
            detected = segments(posteriors[config['model']], config['threshold'], config['min_silence'], valid)
            predicted = mask_segments(detected, len(truth))
            state['frame_counts'] += [int((predicted & truth & valid).sum()),
                                      int((predicted & ~truth & valid).sum()), int((~predicted & truth & valid).sum())]
            state['boundary_counts'] += boundary_counts(detected, target_segments)
            state['segment_count'] += len(detected)
            state['segment_seconds'] += sum(r['end_s'] - r['start_s'] for r in detected)
            state['decision_delay_seconds'] += sum(r['decision_s'] - r['end_s'] for r in detected)
        sources.append({'rttm': spec(annotation), 'audio': spec(audio_path),
                        'samples_16k': len(signal), 'eligible_10ms_frames': int(valid.sum()),
                        'overlap_frames_excluded': int((~valid).sum())})
        write_json_atomic(output / 'progress.json', {'done': index + 1, 'total': expected, 'inference_seconds': inference})
    result = []
    for state in states:
        count = state['segment_count']
        result.append({'config': state['config'], 'frame': prf(state['frame_counts']),
                       'boundary': prf(state['boundary_counts']), 'segment_count': count,
                       'mean_segment_s': state['segment_seconds'] / max(1, count),
                       'mean_endpoint_delay_s': state['decision_delay_seconds'] / max(1, count)})
    if split == 'train':
        best = max(r['frame']['f1'] for r in result)
        tied = [r for r in result if r['frame']['f1'] >= best - .005]
        chosen = min(tied, key=lambda r: (r['config']['min_silence'], r['mean_endpoint_delay_s'],
                                         abs(r['config']['threshold'] - .5), r['config']['model']))
        write_json_atomic(output / 'freeze.json', {'selected': chosen['config'], 'training_source': 'AISHELL-4 train only',
            'selection': 'highest 10ms speech F1; within 0.005 prefer smaller minimum silence and lower endpoint delay',
            'exact_tie': 'threshold closest to 0.5, then model name', 'min_speech_s': .3, 'max_speech_s': 15,
            'postprocessing': 'shared registered probability-threshold state machine; overlap resets state',
            'model_artifacts': spec(ROOT / 'vad_models/summary.json')})
    write_json_atomic(output / 'sources.json', {'files': sources})
    finish(output, meta, {'split': split, 'recordings': expected, 'channel': 0, 'results': result,
                          'inference_seconds': inference, 'sources': spec(output / 'sources.json'),
                          'gpu_seconds': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('split', choices=('train', 'test'))
    run(parser.parse_args().split)
