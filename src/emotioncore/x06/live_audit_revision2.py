"""CPU audit of live inputs/decisions before launching the next replay."""
import argparse
import hashlib
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, checked, selected_vad
from emotioncore.x06.service import Service
from emotioncore.x06.decision import apply


def video_pixels(path):
    import cv2
    capture = cv2.VideoCapture(str(path)); h = hashlib.sha256(); n = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h.update(frame.tobytes()); n += 1
    finally:
        capture.release()
    return n, h.hexdigest()


def inputs_equal(a, b):
    if a['wav_sha256'] != sha256(a['wav']) or b['wav_sha256'] != sha256(b['wav']):
        raise ValueError('STOP audio artifact changed')
    aa, bb = Service.audio(a['wav']), Service.audio(b['wav'])
    audio = aa.shape == bb.shape and np.array_equal(aa, bb)
    crops = [sha256(p) for p in a['face']['crop_paths']] == [sha256(p) for p in b['face']['crop_paths']]
    qa, qb = a.get('q_video'), b.get('q_video')
    video = qa is None and qb is None
    if qa is not None and qb is not None:
        # Failed media receipts intentionally have no output. Service consumes
        # no Q video in this case; never treat a successful/missing pair as equal.
        video = qa['status'] == qb['status']
        if video and qa['status'] == 'ok':
            video = video_pixels(qa['output']) == video_pixels(qb['output'])
    return {'audio_samples': audio, 'expert_crop_bytes': crops, 'q_video_pixels': video,
            'track_scores': a['face']['track_scores'] == b['face']['track_scores']}


def run(system, retry=False):
    output = REV / ('live_audit_' + system + ('_retry' if retry else '')); meta = start(output)
    silence = selected_vad()['min_silence']; key = '04' if silence == .4 else '08'
    full_source = read(REV / ('emotion_' + key) / 'summary.json')['predictions']
    full = {r['segment_id']: r for r in rows(checked(full_source))}
    directory = REV / 'inputs_04' if silence == .4 else ROOT / 'segment_inputs'
    prepared = read(directory / 'summary.json')['inputs']
    inputs = {r['segment_id']: r for r in rows(checked(prepared))}
    live_source = read(ROOT / f'realtime_{system}/summary.json')['events_file']
    live = rows(checked(live_source)); alpha = read(REV / 's5/freeze.json')['alpha']
    service = Service.__new__(Service); service.bundle = read('configs/calibration_bundle_v1.1.json'); service.previous = {}
    state = {}; results = []; mismatch = []; probability_delta = {}
    for event in live:
        uid = event['segment_id']; reference = full[uid]
        fresh_input = read(ROOT / f'realtime_{system}/inputs/{uid}/input.json')
        same = inputs_equal(inputs[uid], fresh_input)
        speech = not event['visual_only']
        name = system if event['has_face'] and speech else 'H_A' if speech else 'H_V'
        old_label = service.decide(reference['component_probs'], name, has_speech=speech, use_gate=False)['label']
        new_label = service.decide(event['component_probs'], name, has_speech=speech, use_gate=False)['label']
        expected = apply(service.bundle, reference['component_probs'], system,
            has_face=reference['has_face'], speech=speech, scores=reference['track_scores'],
            alpha=alpha, state=state, stream_id=reference['speaker_id'])
        for component, p in event['component_probs'].items():
            probability_delta[component] = max(probability_delta.get(component, 0.),
                float(np.max(np.abs(np.asarray(p) - reference['component_probs'][component]))))
        item = {'segment_id': uid, 'same_inputs': same, 'ungated_frozen_label_equal': old_label == new_label,
                'final_label_equal': event['label'] == expected['label'], 'live': event['label'],
                'fullspeed': expected['label'], 'ungated_live': new_label, 'ungated_fullspeed': old_label}
        results.append(item)
        if not all(same.values()) or old_label != new_label or event['label'] != expected['label']:
            mismatch.append(item)
        if all(same.values()) and old_label != new_label:
            write_rows(output / 'checked.jsonl', results)
            stop = {'status': 'STOP_s6a', 'system': system, 'segment': item, 'at': now()}
            write_json_atomic(REV / 's6a_runtime_stop.json', stop)
            finish(output, meta, {**stop, 'checked': spec(output / 'checked.jsonl'), 'gpu_seconds': 0})
            raise RuntimeError('STOP S6a: identical live/fullspeed inputs produced different labels')
    write_rows(output / 'checked.jsonl', results); write_json_atomic(output / 'differences.json', mismatch)
    finish(output, meta, {'status': 'completed', 'system': system, 'n': len(live),
        'input_equal_n': sum(all(r['same_inputs'].values()) for r in results),
        'ungated_label_agreement': sum(r['ungated_frozen_label_equal'] for r in results) / len(results),
        'final_label_agreement': sum(r['final_label_equal'] for r in results) / len(results),
        'max_absolute_probability_difference': probability_delta,
        'checked': spec(output / 'checked.jsonl'), 'differences': spec(output / 'differences.json'),
        'live': live_source, 'fullspeed': full_source, 'gpu_seconds': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('system', choices=('H3', 'H_ref'))
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args(); run(args.system, args.retry)
