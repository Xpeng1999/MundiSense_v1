"""CPU-only explanation of every S6 mismatch, preserving the full denominator."""
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x06.common import *
from emotioncore.x06.decision import apply


def run():
    output = ROOT / 'equivalence_audit'
    meta = start(output)
    bundle = read('configs/calibration_bundle_v1.1.json')
    inputs = {r['utt_id']: r for r in rows(ROOT / 'oracle_inputs/inputs.jsonl')}
    fresh = rows(ROOT / 'equivalence_attempt2/predictions.jsonl')
    baseline = {system: {r['utt_id']: r for r in rows(R5 / f'deployment/{system}_et_test.jsonl')}
                for system in ('H3', 'H_ref')}
    mismatch_details = []; without_multiface = {s: 0 for s in baseline}
    final_implementation_disagreement = {s: 0 for s in baseline}
    for row in fresh:
        item = inputs[row['utt_id']]
        face = item['q_video'] is not None and item['q_video']['status'] == 'ok'
        for system in baseline:
            old = baseline[system][row['utt_id']]
            expected = LABELS[old['label']] if old['label'] >= 0 and old['confidence'] >= bundle['systems'][system]['thresholds']['90'] else 'reject'
            actual = apply(bundle, row['component_probs'], system, has_face=face, scores=item['face']['track_scores'])
            assert actual['label'] == row['systems'][system]['label']
            no_multiface = apply(bundle, row['component_probs'], system, has_face=face, scores=[])
            without_multiface[system] += no_multiface['label'] != expected
            if actual['label'] != expected:
                scores = item['face']['track_scores']
                mismatch_details.append({'utt_id': row['utt_id'], 'system': system, 'expected_x05_r90': expected,
                    'actual_x06': actual['label'], 'reason': actual['reason'], 'has_face': face,
                    'target_score': scores[0], 'second_score': scores[1], 'ratio': scores[1] / scores[0],
                    'without_multiface_only': no_multiface['label']})
    write_json_atomic(output / 'mismatches.json', mismatch_details)
    finish(output, meta, {'n': len(fresh), 'primary': read(ROOT / 'equivalence_attempt2/summary.json')['systems'],
        'without_multiface_only_agreement': {s: 1 - n / len(fresh) for s, n in without_multiface.items()},
        'final_decision_implementation_matches_s6_labels': True,
        'mismatch_reasons': {reason: sum(r['reason'] == reason for r in mismatch_details)
                             for reason in sorted({r['reason'] for r in mismatch_details})},
        'details': spec(output / 'mismatches.json'), 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
