"""Recompute remaining budget from actual segment counts before more GPU work."""
import statistics
from emotioncore.x06.common import *


def run():
    output = ROOT / 'budget_recheck'
    meta = start(output)
    eq = read(ROOT / 'equivalence_attempt2/summary.json')
    if eq['status'] != 'completed' or any(x['agreement'] < .95 for x in eq['systems'].values()):
        raise RuntimeError('STOP equivalence gate has not passed')
    prior = [read(p) for p in (ROOT / 'gpu_jobs').glob('*/summary.json')]
    used = sum(p['charged_gpu_s'] for p in prior)
    cuts = read(ROOT / 'cuts/summary.json')['streams']
    counts = {cohort: {kind: sum(v[kind] for s, v in cuts.items() if (s in FMH) == (cohort == 'fmh'))
                       for kind in ('speech_segments', 'silence_windows')} for cohort in ('fmh', 'et_test')}
    predictions = rows(ROOT / 'equivalence_attempt2/predictions.jsonl')
    stages = {name: statistics.mean(p['timing'].get(name, 0) for p in predictions)
              for name in ('Q_A', 'Q_AV', 'A3', 'V2', 'V1', 'audio_decode')}
    eq_job = read(ROOT / 'gpu_jobs/equivalence_attempt2/summary.json')
    both = eq_job['charged_gpu_s'] / len(predictions)
    qv = read(ROOT / 'preflight/summary.json')['q_rates']['V']['seconds_per_sentence']
    visual = qv + stages['V2'] + stages['V1']
    oracle = rows(ROOT / 'oracle_inputs/inputs.jsonl')
    frame_work = sum(p['face']['decode_seconds'] + p['face']['face_seconds'] for p in oracle)
    frame_count = sum(p['face']['requested_frames'] for p in oracle)
    per_frame = frame_work / frame_count
    q_media = statistics.mean((p['q_video'] or {}).get('elapsed_s', 0) for p in oracle)
    stream_frames = sum(__import__('math').ceil(read(ROOT / f'preflight/timeline_{s}.json')['duration_s'] * 5) for s in TEST)
    audio_duration = 2 * sum(read(ROOT / f'preflight/timeline_{s}.json')['duration_s'] for s in TEST)
    fullspeed = sum(c['speech_segments'] for c in counts.values()) * both
    fullspeed += sum(c['silence_windows'] for c in counts.values()) * visual
    et = counts['et_test']
    # Online 5fps tracking also processes silent/gap frames. Count the complete
    # input stream, rather than scaling face work by VAD segment count.
    real_service = 2 * stream_frames * per_frame
    real_service += et['speech_segments'] * (2 * q_media + stages['Q_AV'] + both
                         + stages['A3'] + stages['V2'] + stages['V1'])
    real_service += et['silence_windows'] * 2 * (q_media + visual)
    remaining = (fullspeed + max(audio_duration, real_service)) * 1.2 + 300
    total = used + remaining
    summary = {'status': 'passed' if total <= MAX_GPU_SECONDS else 'STOP_budget_estimate',
               'used_gpu_hours': used / 3600, 'estimated_total_gpu_hours': total / 3600,
               'estimated_remaining_gpu_hours': remaining / 3600, 'limit_gpu_hours': 15,
               'counts': counts, 'measured_seconds_per_stage': stages, 'both_system_seconds': both,
               'face_cpu_seconds_per_5fps_frame': per_frame, 'et_stream_frames_per_system': stream_frames,
               'q_media_seconds_per_segment': q_media, 'fullspeed_s': fullspeed,
               'realtime_input_s': audio_duration, 'realtime_serial_service_s': real_service,
               'reserve': '20% of future service/playback time plus 300s initialization',
               'cuts': spec(ROOT / 'cuts/summary.json'), 'equivalence': spec(ROOT / 'equivalence_attempt2/summary.json')}
    finish(output, meta, summary)
    if summary['status'] != 'passed':
        raise RuntimeError('STOP updated estimate exceeds 15 GPU hours')


if __name__ == '__main__':
    run()
