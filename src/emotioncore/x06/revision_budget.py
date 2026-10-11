"""CPU-only revision-1 startup audit, before any additional model inference."""
import math
import statistics
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.vad import segments
from emotioncore.x06.online_vad import silence_windows

HISTORICAL = {
    'equivalence_attempt2/summary.json': '2a63cc7740fbf13dffb3167eddb8add80579886159d24acfed60db0e7ec4376b',
    'equivalence_attempt2/predictions.jsonl': 'fceddc74efff5c7b0278d19bbe81f4d8ad903dbc389161ea0d5d23cc37562d8e',
    'stream_equivalence/summary.json': '7041bb133406bfc5880c57536cb4d6845faa1c2039c0449b0bb4e02cfa773e87',
    'stream_equivalence/predictions.jsonl': 'e82eba38aeefcb9bd0c87a5d9a13a5f69dc3c89c3934d0dc8d571df3577d3d7d',
    'stream_oracle_inputs/inputs.jsonl': '49ce9bec28204cf7145e774e46149d3f44cf5e633d627a4513ed6ac5c10650eb',
    'gpu_jobs/stream_equivalence/summary.json': '21b2bf416dcde926f0a56016ddd018fd03a91d077a14c1ae1e750701fd2d7af2',
}


def check_s6a(summary):
    if (summary['n'] != 1929 or summary['status'] != 'completed'
            or set(summary['systems']) != {'H3', 'H_ref'}
            or any(v['without_gate_mismatches'] != 0 for v in summary['systems'].values())):
        raise RuntimeError('STOP S6a: identical inputs produced different outputs')


def estimate(used, oracle_n, candidates, rates, frames, duration, face_per_frame, q_media, visual):
    """Retain the preregistered serial-service bound and 20% + 300s reserve.

    The latest actual-stream wall rate includes initialization and CPU I/O.
    Full-speed stage estimates intentionally do not assume cross-config cache
    sharing or that earlier concurrent CPU contention will disappear.
    """
    oracle = oracle_n * rates['both']
    fullspeed = sum(c[k] * rate for c in candidates.values()
                    for k, rate in [('speech', rates['both']), ('visual', visual)])
    replay = {}
    for name, c in candidates.items():
        serial = 2 * frames * face_per_frame
        serial += c['et_speech'] * (2 * q_media + rates['Q_AV'] + rates['both']
                                   + rates['A3'] + rates['V2'] + rates['V1'])
        serial += c['et_visual'] * 2 * (q_media + visual)
        replay[name] = {'input_s': 2 * duration, 'serial_service_s': serial,
                        'reserved_base_s': max(2 * duration, serial)}
    replay_base = max(r['reserved_base_s'] for r in replay.values())
    future_base = oracle + fullspeed + replay_base
    reserve = .2 * future_base + 300
    return {'correct_boundary_s': oracle, 'two_vad_configs_s': fullspeed,
            'replay_candidates': replay, 'replay_base_s': replay_base,
            'reserve_s': reserve, 'future_base_s': future_base,
            'estimated_total_gpu_hours': (used + future_base + reserve) / 3600}


def run():
    output = ROOT / 'revision_1_budget'
    meta = start(output)
    verified = {}
    for relative, expected in HISTORICAL.items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError('STOP immutable historical artifact changed: ' + relative)
        verified[relative] = spec(path)
    original = read(ROOT / 'equivalence_attempt2/summary.json')
    check_s6a(original)
    old = rows(ROOT / 'stream_equivalence/predictions.jsonl')
    inputs = rows(ROOT / 'stream_oracle_inputs/inputs.jsonl')
    old_ids = [r['utt_id'] for r in old]
    if (len(old_ids) != 1184 or len(set(old_ids)) != 1184 or len(inputs) != 1929
            or old_ids != [r['utt_id'] for r in inputs[:1184]]):
        raise ValueError('STOP ET reuse population mismatch')
    frozen = read('configs/x06_vad_freeze.json')['selected']
    if frozen != {'model': 'fsmn', 'threshold': .3, 'min_silence': .8}:
        raise ValueError('STOP original VAD freeze changed')
    candidates = {}; posterior_sources = {}
    for silence in (.4, .8):
        per_speaker = {}
        for speaker in sorted(FMH | TEST):
            path = ROOT / f'cuts/{speaker}_posterior.npy'
            probability = np.load(path)
            duration = read(ROOT / f'preflight/timeline_{speaker}.json')['duration_s']
            voice = segments(probability, frozen['threshold'], silence)
            visual = silence_windows(voice, duration)
            per_speaker[speaker] = {'speech': len(voice), 'visual': len(visual)}
            posterior_sources[speaker] = spec(path)
            if silence == .8:
                recorded = read(ROOT / 'cuts/summary.json')['streams'][speaker]
                assert (len(voice), len(visual)) == (recorded['speech_segments'], recorded['silence_windows'])
        candidates[str(silence)] = {
            kind: sum(v[kind] for v in per_speaker.values()) for kind in ('speech', 'visual')}
        candidates[str(silence)].update({
            'et_' + kind: sum(v[kind] for s, v in per_speaker.items() if s in TEST)
            for kind in ('speech', 'visual')})
        candidates[str(silence)]['per_speaker'] = per_speaker
    job_paths = sorted((ROOT / 'gpu_jobs').glob('*/summary.json'))
    used = sum(read(p)['charged_gpu_s'] for p in job_paths)
    latest = read(ROOT / 'gpu_jobs/stream_equivalence/summary.json')
    rates = {key: statistics.mean(r['timing'].get(key, 0) for r in old)
             for key in ('Q_A', 'Q_AV', 'A3', 'V2', 'V1', 'audio_decode')}
    rates['both'] = latest['charged_gpu_s'] / len(old)
    previous = read(ROOT / 'budget_recheck/summary.json')
    qv = read(ROOT / 'preflight/summary.json')['q_rates']['V']['seconds_per_sentence']
    visual_rate = qv + rates['V2'] + rates['V1']
    duration = sum(read(ROOT / f'preflight/timeline_{s}.json')['duration_s'] for s in TEST)
    frames = sum(math.ceil(read(ROOT / f'preflight/timeline_{s}.json')['duration_s'] * 5) for s in TEST)
    result = estimate(used, 745 + 3266, candidates, rates, frames, duration,
                      previous['face_cpu_seconds_per_5fps_frame'], previous['q_media_seconds_per_segment'], visual_rate)
    passed = result['estimated_total_gpu_hours'] <= MAX_GPU_SECONDS / 3600
    finish(output, meta, {**result, 'status': 'passed' if passed else 'STOP_budget_estimate',
        'limit_gpu_hours': MAX_GPU_SECONDS / 3600, 'used_gpu_s': used, 'used_gpu_hours': used / 3600,
        'new_gpu_seconds': 0, 'reuse_et_n': 1184, 'remaining_et_n': 745, 'fmh_oracle_n': 3266,
        's6a': {s: {'n': 1929, 'agreement_without_gates': 1.0} for s in ('H3', 'H_ref')},
        'counts': candidates, 'rates_s': rates, 'visual_s': visual_rate,
        'face_cpu_seconds_per_5fps_frame': previous['face_cpu_seconds_per_5fps_frame'],
        'q_media_seconds_per_segment': previous['q_media_seconds_per_segment'],
        'et_stream_frames_per_system': frames, 'historical_verified': verified,
        'ledger': [spec(p) for p in job_paths], 'posterior_sources': posterior_sources,
        'original_estimator': spec(ROOT / 'budget_recheck/summary.json'),
        'assumptions': ['Latest measured actual-stream full-process GPU rate, including initialization and CPU contention',
            'Both VAD configurations receive inference; no unimplemented cross-configuration cache savings assumed',
            'Replay reserves the more costly candidate because FMH emotion-based selection has not run',
            'Original serial-service bound and 20% future-time plus 300s reserve retained'],
        'emotion_predictions_read': 'existing ET prefix only for timing and immutable-ID reuse audit; no new outputs',
        'emotion_labels_read_for_selection': False, 'original_stop_retained': True})
    print(json.dumps(read(output / 'summary.json'), ensure_ascii=False))
    if not passed:
        raise RuntimeError('STOP revision-1 startup estimate exceeds 18 GPU hours; no inference launched')


if __name__ == '__main__':
    run()
