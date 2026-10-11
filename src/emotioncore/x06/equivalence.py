"""Fresh correct-boundary inference; stop before later GPU stages below 95%."""
import json
import os
from emotioncore.eval.metrics import LABELS
from emotioncore.x06.common import *
from emotioncore.x06.service import Service


def run():
    if os.environ.get('X06_GPU_GUARDED') != 'equivalence':
        raise RuntimeError('Use X06 GPU watchdog')
    output = ROOT / 'equivalence'
    meta = start(output)
    inputs = rows(ROOT / 'oracle_inputs/inputs.jsonl')
    assert len(inputs) == 1929
    parameters = read('configs/calibration_bundle_v1.json')['systems']
    references = {name: {r['utt_id']: r for r in rows(R5 / f'deployment/{name}_et_test.jsonl')}
                  for name in ('H3', 'H_ref')}
    service = Service()
    mismatches = {n: [] for n in references}
    raw_mismatches = {n: 0 for n in references}
    count = 0
    with (output / 'predictions.jsonl').open('w') as stream:
        for item in inputs:
            probabilities, has_face, timing = service.components(item, 'H3', compare_both=True)
            record = {'utt_id': item['utt_id'], 'speaker_id': item['speaker_id'],
                      'component_probs': probabilities, 'timing': timing, 'systems': {}}
            for name, reference in references.items():
                old = reference[item['utt_id']]
                expected = LABELS[old['label']] if old['label'] >= 0 and old['confidence'] >= parameters[name]['thresholds']['90'] else 'reject'
                result = service.decide(probabilities, name, has_face=has_face,
                                        scores=item['face']['track_scores'])
                no_gate = service.decide(probabilities, name, use_gate=False)
                raw_mismatches[name] += no_gate['label'] != expected
                if result['label'] != expected:
                    mismatches[name].append({'utt_id': item['utt_id'], 'expected': expected,
                                            'actual': result['label'], 'without_gate': no_gate['label'],
                                            'reason': result['reason'], 'confidence': result['confidence']})
                record['systems'][name] = result
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
            if count % 25 == 0:
                stream.flush(); storage_gate()
                write_json_atomic(output / 'progress.json', {'done': count, 'total': len(inputs),
                    'mismatches': {n: len(v) for n, v in mismatches.items()}})
            # Stop as soon as even all remaining correct cases cannot reach 95%.
            if any(1 - len(v) / len(inputs) < .95 for v in mismatches.values()):
                break
    metrics = {n: {'agreement': (count - len(v)) / count,
                   'mismatches': len(v), 'without_gate_mismatches': raw_mismatches[n]}
               for n, v in mismatches.items()}
    failed = count != len(inputs) or any(m['agreement'] < .95 for m in metrics.values())
    write_json_atomic(output / 'mismatches.json', mismatches)
    finish(output, meta, {'status': 'STOP_equivalence' if failed else 'completed', 'n': count,
                          'required_n': len(inputs), 'systems': metrics,
                          'predictions': spec(output / 'predictions.jsonl'),
                          'mismatch_details': spec(output / 'mismatches.json'),
                          'parameter_bundle': spec('configs/calibration_bundle_v1.1.json')})
    if failed:
        raise RuntimeError('STOP equivalence below 0.95; do not continue')


if __name__ == '__main__':
    run()
