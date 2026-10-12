"""S2 FMH-only silence selection, S3 gates, then S5 FMH-only EMA selection."""
import argparse
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, checked, s6b_gate, selected_vad
from emotioncore.x06.analysis import truths, truth_evaluation, describe, choose_alpha
from emotioncore.x06.gate import route


def speech_records(silence):
    source = read(REV / ('emotion_04' if silence == .4 else 'emotion_08') / 'summary.json')['predictions']
    return [r for r in rows(checked(source)) if not r['visual_only']], source


def choose_silence(bundle, candidates):
    if set(candidates) != {.4, .8}:
        raise ValueError('STOP both registered silence candidates required')
    values = {}
    for silence, speech in candidates.items():
        speakers = {r['speaker_id'] for r in speech}
        assert_tuning_source('fmh', speakers)
        if speakers != set(FMH) or any(r['cohort'] != 'fmh' for r in speech):
            raise ValueError('STOP silence selection requires only all three FMH streams')
        values[str(silence)] = truth_evaluation(bundle, speech, FMH, 'H3')[0]
    maximum = max(r['macro_f1'] for r in values.values())
    chosen = min(float(s) for s, r in values.items() if r['macro_f1'] >= maximum - .005)
    return chosen, values


def baselines():
    result = {}; sources = {}
    for cohort, speakers, path in [('fmh', FMH, ROOT / 'fmh_stream_oracle/summary.json'),
                                  ('et_test', TEST, REV / 'et_remaining/summary.json')]:
        source = read(path)['predictions']; fresh = rows(checked(source)); sources[cohort] = source
        truth = {r['utt_id']: r['label'] for s in speakers for r in truths(s)}
        if {r['utt_id'] for r in fresh} != set(truth):
            raise ValueError('STOP correct-boundary baseline population incomplete')
        result[cohort] = {system: describe([truth[r['utt_id']] for r in fresh],
                                         [r['systems'][system] for r in fresh]) for system in ('H3', 'H_ref')}
    return result, sources


def s2():
    s6b_gate(read(REV / 's6b_fmh/summary.json'))
    output = REV / 's2'; meta = start(output)
    bundle = read('configs/calibration_bundle_v1.1.json')
    candidates = {}; sources = {}
    for silence in (.4, .8):
        candidates[silence], sources[str(silence)] = speech_records(silence)
    selected, fm_metrics = choose_silence(bundle, {s: [r for r in p if r['cohort'] == 'fmh']
                                                  for s, p in candidates.items()})
    freeze = {'model': 'fsmn', 'threshold': .3, 'min_silence': selected,
        'selection_system': 'H3', 'metric': 'S2 macro-F1, OP-R90 and gates, rejects in denominator',
        'rule': 'FMH only; within 0.005 select 0.4 seconds', 'fmh_speakers': sorted(FMH),
        'candidates': fm_metrics, 'frozen_at': now(), 'selection_sources': sources,
        'selection_label_sources': [spec(ROOT / f'preflight/timeline_{s}.json') for s in sorted(FMH)],
        'original_vad_freeze': spec('configs/x06_vad_freeze.json'), 'fitted_parameters_changed': False}
    write_json_atomic(output / 'freeze.json', freeze)
    # All ET label access in this phase occurs after the FMH decision is sealed.
    original, baseline_sources = baselines(); reports = {}
    for silence, speech in candidates.items():
        reports[str(silence)] = {}
        for cohort, speakers in [('fmh', FMH), ('et_test', TEST)]:
            reports[str(silence)][cohort] = {}
            current = [r for r in speech if r['speaker_id'] in speakers]
            for system in ('H3', 'H_ref'):
                metrics, decisions = truth_evaluation(bundle, current, speakers, system)
                metrics['macro_f1_minus_correct_boundaries'] = metrics['macro_f1'] - original[cohort][system]['macro_f1']
                path = output / f'{silence}_{cohort}_{system}.jsonl'; write_rows(path, decisions)
                reports[str(silence)][cohort][system] = {'metrics': metrics, 'predictions': spec(path)}
    finish(output, meta, {'min_silence': selected, 'freeze': spec(output / 'freeze.json'), 'reports': reports,
                         'correct_boundaries': original, 'baseline_sources': baseline_sources, 'gpu_seconds': 0})
    print(json.dumps({'selected_min_silence': selected, 'fmh_candidates': fm_metrics}))


def s3():
    output = REV / 's3'; meta = start(output)
    selected = selected_vad()['min_silence']; speech, source = speech_records(selected)
    report = read(REV / 's2/summary.json')['reports'][str(selected)]; results = {}
    for cohort, speakers in [('fmh', FMH), ('et_test', TEST)]:
        current = [r for r in speech if r['speaker_id'] in speakers]
        triggers = [r for r in current if route(r['has_face'], True, r['track_scores'])[1] == 'ambiguous_multi_face']
        path = output / f'{cohort}_triggers.jsonl'
        write_rows(path, [{k: r[k] for k in ('segment_id', 'speaker_id', 'start_s', 'end_s', 'track_scores')}
                          for r in triggers])
        results[cohort] = {'speech_events': len(current), 'ambiguous_multi_face_events': len(triggers),
            'event_trigger_rate': len(triggers) / len(current), 'trigger_details': spec(path),
            'truth_metrics': {system: report[cohort][system]['metrics'] for system in ('H3', 'H_ref')}}
    finish(output, meta, {'min_silence': selected, 'reports': results, 'source': source,
                         'gate_rule_changed': False, 'gpu_seconds': 0})


def s5():
    if not (REV / 's3/summary.json').exists():
        raise RuntimeError('STOP S3 must precede S5')
    output = REV / 's5'; meta = start(output)
    selected = selected_vad()['min_silence']; speech, source = speech_records(selected)
    bundle = read('configs/calibration_bundle_v1.1.json')
    alpha, candidates = choose_alpha(bundle, [r for r in speech if r['cohort'] == 'fmh'])
    freeze = {'alpha': alpha, 'selection_system': 'H3', 'applied_systems': ['H3', 'H_ref'],
        'rule': 'FMH S5 macro-F1; within 0.005 select smaller alpha', 'fmh_speakers': sorted(FMH),
        'min_silence': selected, 'candidates': candidates, 'frozen_at': now(),
        'selection_label_sources': [spec(ROOT / f'preflight/timeline_{s}.json') for s in sorted(FMH)],
        'neutral_bias_and_thresholds_refitted': False, 'source': source,
        'silence_freeze': spec(REV / 's2/freeze.json')}
    write_json_atomic(output / 'freeze.json', freeze)
    reports = {}
    for cohort, speakers in [('fmh', FMH), ('et_test', TEST)]:
        current = [r for r in speech if r['speaker_id'] in speakers]; reports[cohort] = {}
        for system in ('H3', 'H_ref'):
            zero, _ = truth_evaluation(bundle, current, speakers, system, 0.)
            metrics, decisions = truth_evaluation(bundle, current, speakers, system, alpha)
            metrics['macro_f1_minus_same_order_alpha_zero'] = metrics['macro_f1'] - zero['macro_f1']
            path = output / f'{cohort}_{system}.jsonl'; write_rows(path, decisions)
            reports[cohort][system] = {'alpha_zero': zero, 'selected': metrics, 'predictions': spec(path)}
    finish(output, meta, {'alpha': alpha, 'min_silence': selected, 'reports': reports,
                         'freeze': spec(output / 'freeze.json'), 'gpu_seconds': 0})
    print(json.dumps({'selected_alpha': alpha, 'min_silence': selected}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=('s2', 's3', 's5'))
    {'s2': s2, 's3': s3, 's5': s5}[parser.parse_args().stage]()
