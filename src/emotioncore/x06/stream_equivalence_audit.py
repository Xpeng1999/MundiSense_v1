"""Report actual-stream disagreements, including an early-stop upper bound."""
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x06.common import *
from emotioncore.x06.decision import apply
from emotioncore.x06.service import Service


def run():
    output = ROOT / 'stream_equivalence_audit'
    meta = start(output)
    summary = read(ROOT / 'stream_equivalence/summary.json')
    inputs = {r['utt_id']: r for r in rows(ROOT / 'stream_oracle_inputs/inputs.jsonl')}
    fresh = rows(ROOT / 'stream_equivalence/predictions.jsonl')
    single = {r['utt_id']: r for r in rows(ROOT / 'equivalence_attempt2/predictions.jsonl')}
    baseline = {s: {r['utt_id']: r for r in rows(R5 / f'deployment/{s}_et_test.jsonl')}
                for s in ('H3', 'H_ref')}
    service = Service.__new__(Service)
    service.bundle = read('configs/calibration_bundle_v1.1.json')
    service.previous = {}
    details = []; counterfactual = {s: {'without_multiface_mismatches': 0, 'without_all_gates_mismatches': 0} for s in baseline}
    for row in fresh:
        item = inputs[row['utt_id']]
        face = item['q_video'] is not None and item['q_video']['status'] == 'ok'
        scores = item['face']['track_scores']
        for system in baseline:
            old = baseline[system][row['utt_id']]
            threshold = service.bundle['systems'][system]['thresholds']['90']
            expected = LABELS[old['label']] if old['label'] >= 0 and old['confidence'] >= threshold else 'reject'
            actual = apply(service.bundle, row['component_probs'], system, has_face=face, scores=scores)
            assert actual['label'] == row['systems'][system]['label']
            no_multi = apply(service.bundle, row['component_probs'], system, has_face=face, scores=[])
            no_gates = service.decide(row['component_probs'], system, use_gate=False)
            counterfactual[system]['without_multiface_mismatches'] += no_multi['label'] != expected
            counterfactual[system]['without_all_gates_mismatches'] += no_gates['label'] != expected
            if actual['label'] != expected:
                if actual['reason'] == 'ambiguous_multi_face' and no_multi['label'] == expected:
                    cause = 'recovered_when_only_multiface_gate_removed'
                elif no_gates['label'] == expected:
                    cause = 'routing_or_gate_difference'
                else:
                    cause = 'changed_stream_input_decision'
                delta = {name: float(np.max(np.abs(np.asarray(p) - single[row['utt_id']]['component_probs'][name])))
                         for name, p in row['component_probs'].items()}
                details.append({'utt_id': row['utt_id'], 'system': system, 'expected_x05_r90': expected,
                    'actual': actual['label'], 'confidence': actual['confidence'], 'route': actual['route'],
                    'reason': actual['reason'], 'diagnostic_category': cause,
                    'without_multiface_only': no_multi['label'], 'without_all_gates': no_gates['label'],
                    'has_face': face, 'track_scores': scores,
                    'second_to_first_score_ratio': scores[1] / scores[0] if len(scores) > 1 and scores[0] else None,
                    'max_absolute_component_change_from_original_clip': delta})
    path = output / 'mismatches.json'
    write_json_atomic(path, details)
    statistics = {}
    for system in baseline:
        current = [r for r in details if r['system'] == system]
        statistics[system] = {'processed_prefix_agreement': 1 - len(current) / len(fresh),
            'mismatches': len(current), 'optimistic_full_population_agreement': 1 - len(current) / summary['required_n'],
            'complete_population_agreement': 1 - len(current) / len(fresh) if len(fresh) == summary['required_n'] else None,
            'diagnostic_categories': {name: sum(r['diagnostic_category'] == name for r in current)
                                     for name in sorted({r['diagnostic_category'] for r in current})},
            **counterfactual[system]}
    finish(output, meta, {'status': summary['status'], 'processed': len(fresh), 'planned': summary['required_n'],
        'systems': statistics, 'details': spec(path), 'source': spec(ROOT / 'stream_equivalence/summary.json'),
        'interpretation': 'Changed-input categories locate decision differences but do not causally separate waveform alignment from video sampling; no test-driven parameter changes',
        'gpu_seconds': 0})


if __name__ == '__main__':
    run()
