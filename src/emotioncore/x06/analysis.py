"""FMH-only EMA selection, then sealed-parameter reports on ET-test."""
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x06.common import *
from emotioncore.x06.aggregate import overlaps, components_for_truth, metrics
from emotioncore.x06.decision import apply, from_probability
from emotioncore.x06.gate import route


def truths(speaker):
    return [r for r in read(ROOT / f'preflight/timeline_{speaker}.json')['entries'] if r['kind'] == 'utterance']


def describe(truth, decisions):
    labels = [d['label'] for d in decisions]
    result = metrics(truth, labels)
    keep = [i for i, label in enumerate(labels) if label != 'reject']
    result['accepted_metrics'] = metrics([truth[i] for i in keep], [labels[i] for i in keep])
    result['ambiguous_multi_face_n'] = sum(d.get('reason') == 'ambiguous_multi_face' for d in decisions)
    result['ambiguous_multi_face_rate'] = result['ambiguous_multi_face_n'] / max(1, len(decisions))
    return result


def sequence(bundle, records, system, alpha):
    state = {}; output = []
    for row in sorted(records, key=lambda r: (r['speaker_id'], r['start_s'])):
        decision = apply(bundle, row['component_probs'], system, has_face=row['has_face'],
                         scores=row['track_scores'], alpha=alpha, state=state, stream_id=row['speaker_id'])
        output.append({**row, 'decision': decision})
    return output


def truth_evaluation(bundle, speech, speakers, system, alpha=None):
    # alpha=None is the exact S2 ordering; numerical alpha uses the separate S5
    # calibrated-probability aggregation ordering, including its alpha=0 control.
    events = speech if alpha is None else sequence(bundle, speech, system, alpha)
    results = []; labels = []; split = covered = merged = vad_count = 0
    covered_seconds = truth_seconds = 0.
    for speaker in sorted(speakers):
        sentences = truths(speaker)
        segments = [s for s in events if s['speaker_id'] == speaker]
        vad_count += len(segments)
        merged += sum(sum(t['start_s'] < s['end_s'] and t['end_s'] > s['start_s']
                          for t in sentences) > 1 for s in segments)
        for truth in sentences:
            labels.append(truth['label'])
            matched = overlaps(truth, segments)
            covered += bool(matched); split += len(matched) > 1
            truth_seconds += truth['end_s'] - truth['start_s']
            covered_seconds += sum(w for _, w in matched)
            if not matched:
                decision = {'label': 'reject', 'confidence': 0., 'reason': 'uncovered_by_vad'}
            else:
                has_face = any(segments[i]['has_face'] for i, _ in matched)
                ambiguous = any(route(segments[i]['has_face'], True, segments[i]['track_scores'])[1]
                                == 'ambiguous_multi_face' for i, _ in matched)
                if alpha is None:
                    p = components_for_truth(truth, segments)
                    decision = apply(bundle, p, system, has_face=has_face, scores=[1., .5] if ambiguous else [])
                else:
                    total = sum(w for _, w in matched)
                    p = sum(w * np.asarray(segments[i]['decision']['calibrated_probabilities']) for i, w in matched) / total
                    name = system if has_face else 'H_A'
                    decision = from_probability(p, bundle['systems'][name], ambiguous=ambiguous, routing=name,
                                                reason='both_available' if has_face else 'no_face')
            results.append({'utt_id': truth['utt_id'], 'speaker_id': speaker, 'truth': truth['label'], **decision})
    result = describe(labels, results)
    result.update(vad_truth_coverage=covered / len(labels), vad_duration_coverage=covered_seconds / truth_seconds,
                  uncovered_truth_n=len(labels) - covered, truth_split_fraction=split / len(labels),
                  merged_vad_segment_fraction=merged / max(1, vad_count), vad_speech_segments=vad_count)
    if alpha is not None:
        pairs = [(a['decision']['label'], b['decision']['label']) for a, b in zip(events, events[1:])
                 if a['speaker_id'] == b['speaker_id']]
        accepted_pairs = [(a, b) for a, b in pairs if a != 'reject' and b != 'reject']
        result['label_flip_rate_including_reject'] = sum(a != b for a, b in pairs) / max(1, len(pairs))
        result['label_flip_rate_adjacent_accepted'] = sum(a != b for a, b in accepted_pairs) / max(1, len(accepted_pairs))
    return result, results


def choose_alpha(bundle, speech):
    speakers = {r['speaker_id'] for r in speech}
    assert_tuning_source('fmh', speakers)
    if speakers != set(FMH) or any(r['cohort'] != 'fmh' for r in speech):
        raise ValueError('STOP smoothing selection requires all three FMH streams only')
    values = {str(alpha): truth_evaluation(bundle, speech, FMH, 'H3', alpha)[0] for alpha in (0., .2, .4)}
    maximum = max(r['macro_f1'] for r in values.values())
    chosen = min(float(alpha) for alpha, r in values.items() if r['macro_f1'] >= maximum - .005)
    return chosen, values


def fmh_baseline(bundle):
    source = read(ROOT / 'fmh_stream_oracle/summary.json')['predictions']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('STOP FMH correct-boundary prediction SHA changed')
    fresh = {r['utt_id']: r for r in rows(source['path'])}
    out = {system: [] for system in ('H3', 'H_ref')}
    for speaker in sorted(FMH):
        for truth in truths(speaker):
            row = fresh[truth['utt_id']]
            for system in out:
                out[system].append({'utt_id': truth['utt_id'], 'speaker_id': speaker,
                    'truth': truth['label'], **row['systems'][system]})
    return out, {'actual_stream_correct_boundaries': source}


def run():
    output = ROOT / 'smoothing'
    meta = start(output)
    bundle = read('configs/calibration_bundle_v1.1.json')
    source = read(ROOT / 'emotion_segments/summary.json')['predictions']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('STOP segment predictions SHA changed')
    speech = [r for r in rows(source['path']) if not r['visual_only']]
    fmh = [r for r in speech if r['cohort'] == 'fmh']
    alpha, candidates = choose_alpha(bundle, fmh)
    freeze = {'alpha': alpha, 'selection_system': 'H3', 'applied_systems': ['H3', 'H_ref'],
              'selection': 'FMH-only S5 macro-F1; within 0.005 choose smaller alpha',
              'fmh_speakers': sorted(FMH), 'candidates': candidates, 'frozen_at': now(),
              'selection_label_sources': [spec(ROOT / f'preflight/timeline_{s}.json') for s in sorted(FMH)],
              'neutral_bias_and_thresholds_refitted': False, 'bundle': spec('configs/calibration_bundle_v1.1.json')}
    write_json_atomic(output / 'freeze.json', freeze)
    # ET-test labels are accessed only below, after the selection has frozen.
    fmh_original, baseline_sources = fmh_baseline(bundle)
    et_truth = {r['utt_id']: r['label'] for speaker in TEST for r in truths(speaker)}
    et_original = {system: [] for system in ('H3', 'H_ref')}
    for row in rows(ROOT / 'stream_equivalence/predictions.jsonl'):
        for system in et_original:
            et_original[system].append({'utt_id': row['utt_id'], 'truth': et_truth[row['utt_id']], **row['systems'][system]})
    reports = {}
    for cohort, speakers, baseline in [('fmh', FMH, fmh_original), ('et_test', TEST, et_original)]:
        current = [r for r in speech if r['speaker_id'] in speakers]
        reports[cohort] = {}
        for system in ('H3', 'H_ref'):
            original = describe([r['truth'] for r in baseline[system]], baseline[system])
            s2, decisions = truth_evaluation(bundle, current, speakers, system)
            s2['macro_f1_minus_correct_boundaries'] = s2['macro_f1'] - original['macro_f1']
            write_rows(output / f'{cohort}_{system}_s2.jsonl', decisions)
            zero, _ = truth_evaluation(bundle, current, speakers, system, 0.)
            smoothed, smooth_decisions = truth_evaluation(bundle, current, speakers, system, alpha)
            smoothed['macro_f1_minus_same_order_alpha_zero'] = smoothed['macro_f1'] - zero['macro_f1']
            write_rows(output / f'{cohort}_{system}_s5.jsonl', smooth_decisions)
            gate_events = sum(route(r['has_face'], True, r['track_scores'])[1] == 'ambiguous_multi_face' for r in current)
            reports[cohort][system] = {'correct_boundaries': original, 's2': s2, 's5_alpha_zero': zero,
                's5_selected': smoothed, 'speech_event_ambiguous_multi_face_rate': gate_events / max(1, len(current)),
                's2_predictions': spec(output / f'{cohort}_{system}_s2.jsonl'),
                's5_predictions': spec(output / f'{cohort}_{system}_s5.jsonl')}
    finish(output, meta, {'alpha': alpha, 'freeze': spec(output / 'freeze.json'), 'reports': reports,
                         'baseline_sources': baseline_sources, 'segment_source': source, 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
