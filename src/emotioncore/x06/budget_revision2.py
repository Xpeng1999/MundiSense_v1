"""Original serial upper bound and reserve, with the revised replay population."""
import argparse
import math
import statistics
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, TASK_OUTPUTS, authorize, checked, ledger, done


def replay_cost(system, counts, frames, duration, rates, face, media, visual):
    # Summing H3 and H_ref on identical populations exactly recovers revision 1.
    speech = media + (rates['Q_AV'] + rates['A3'] + rates['V2'] + rates['V1']
                      if system == 'H3' else rates['both'])
    serial = frames * face + counts['speech'] * speech + counts['visual'] * (media + visual)
    return {'serial_service_s': serial, 'input_s': duration, 'base_s': max(serial, duration)}


def run(stage):
    output = REV / ('budget_' + stage)
    meta = start(output)
    authorize()
    original = read(checked(read('experiments/X06a_stream/results/revision_1_budget.json')['source']))
    log = ledger(); used = sum(r['charged_gpu_s'] for r in log)
    rates = dict(original['rates_s']); rate_source = original['historical_verified']['stream_equivalence/predictions.jsonl']
    candidates = []
    for task in ('rev2_et', 'fmh_stream_oracle', 'rev2_s2_04', 'rev2_s2_08'):
        if done(task):
            candidates.append((read(ROOT / f'gpu_jobs/{task}/run_meta.json')['ended_at'], task))
    if candidates:
        task = max(candidates)[1]
        summary = read(TASK_OUTPUTS[task] / 'summary.json')
        rate_source = summary.get('new_predictions', summary['predictions'])
        predictions = rows(checked(rate_source))
        rates = {key: statistics.mean(r['timing'].get(key, 0) for r in predictions)
                 for key in ('Q_A', 'Q_AV', 'A3', 'V2', 'V1', 'audio_decode')}
        rates['both'] = read(ROOT / f'gpu_jobs/{task}/summary.json')['charged_gpu_s'] / len(predictions)
    qv = read(ROOT / 'preflight/summary.json')['q_rates']['V']['seconds_per_sentence']
    visual = qv + rates['V2'] + rates['V1']
    remaining_oracle = (0 if done('rev2_et') else 745) + (0 if done('fmh_stream_oracle') else 3266)
    oracle = remaining_oracle * rates['both']
    fullspeed = sum(c['speech'] * rates['both'] + c['visual'] * visual
                    for key, c in original['counts'].items()
                    if not done('rev2_s2_' + ('04' if key == '0.4' else '08')))
    selected = read(REV / 's2/freeze.json')['min_silence'] if (REV / 's2/freeze.json').exists() else None
    replay = {}
    face = original['face_cpu_seconds_per_5fps_frame']; media = original['q_media_seconds_per_segment']
    replay_rate_source = None
    if done('realtime_H3'):
        live = read(ROOT / 'realtime_H3/summary.json')
        # Later delivery stages can use measured H3 producer costs when present.
        replay_rate_source = spec(ROOT / 'realtime_H3/summary.json')
        streams = list(live.get('streams', {}).values())
        if isinstance(streams, list) and streams and all('totals' in s for s in streams):
            frames_all = sum(math.ceil(s['input_duration_s'] * 5) for s in streams)
            face = sum(s['totals']['decode'] + s['totals']['face_detection'] for s in streams) / frames_all
    for silence, counts in original['counts'].items():
        if selected is not None and float(silence) != selected:
            continue
        systems = {}
        for system, speakers in [('H3', sorted(TEST)), ('H_ref', ['21'])]:
            if done('realtime_' + system):
                continue
            c = {k: sum(counts['per_speaker'][s][k] for s in speakers) for k in ('speech', 'visual')}
            timelines = [read(ROOT / f'preflight/timeline_{s}.json') for s in speakers]
            duration = sum(t['duration_s'] for t in timelines)
            frames = sum(math.ceil(t['duration_s'] * 5) for t in timelines)
            systems[system] = {**replay_cost(system, c, frames, duration, rates, face, media, visual),
                               'speakers': speakers, 'counts': c, 'frames': frames}
        replay[silence] = {'systems': systems, 'base_s': sum(v['base_s'] for v in systems.values())}
    replay_base = max((r['base_s'] for r in replay.values()), default=0.)
    future = oracle + fullspeed + replay_base
    reserve = .2 * future + (300 if future else 0)
    total = (used + future + reserve) / 3600
    passed = total <= MAX_GPU_SECONDS / 3600
    summary = {'status': 'passed' if passed else 'STOP_budget_estimate', 'stage': stage,
        'used_gpu_s': used, 'used_gpu_hours': used / 3600, 'remaining_correct_boundary_n': remaining_oracle,
        'correct_boundary_s': oracle, 'remaining_vad_configs_s': fullspeed,
        'replay_candidates': replay, 'replay_base_s': replay_base, 'future_base_s': future,
        'reserve_s': reserve, 'estimated_total_gpu_hours': total, 'limit_gpu_hours': 18,
        'rates_s': rates, 'rate_source': rate_source, 'replay_rate_source': replay_rate_source,
        'face_seconds_per_frame': face, 'media_seconds_per_segment': media,
        'selected_min_silence': selected, 'ledger': [r['source'] for r in log],
        'prior_budget': read('experiments/X06a_stream/results/revision_1_budget.json')['source'],
        'reserve_rule': '20% of remaining work plus 300s; zero after all work completed',
        'gpu_seconds': 0, 'emotion_labels_read_for_selection': False}
    finish(output, meta, summary)
    write_json_atomic(REV / 'budget_current.json', spec(output / 'summary.json'))
    print(json.dumps({'stage': stage, 'status': summary['status'], 'used_hours': used / 3600,
                      'estimated_total_hours': total, 'replay_hours': replay_base / 3600}))
    if not passed:
        raise RuntimeError('STOP revised total estimate exceeds 18h; do not launch next stage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', required=True)
    run(parser.parse_args().stage)
