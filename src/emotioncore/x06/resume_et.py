"""Infer only the immutable ET remainder with the unchanged inference service."""
import os
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, authorize
from emotioncore.x06.service import Service


def remaining(inputs, previous):
    ids = [r['utt_id'] for r in previous]
    if (len(inputs) != 1929 or len(previous) != 1184 or len(set(ids)) != 1184
            or ids != [r['utt_id'] for r in inputs[:1184]]):
        raise ValueError('STOP ET immutable prefix mismatch')
    return inputs[1184:]


def run():
    if os.environ.get('X06_GPU_GUARDED') != 'rev2_et':
        raise RuntimeError('Use X06 GPU watchdog')
    output = REV / 'et_remaining'; meta = start(output); authorize()
    source = ROOT / 'stream_equivalence/predictions.jsonl'
    inputs_path = ROOT / 'stream_oracle_inputs/inputs.jsonl'
    todo = remaining(rows(inputs_path), rows(source))
    service = Service()
    with (output / 'new_predictions.jsonl').open('w') as stream:
        for i, item in enumerate(todo):
            probabilities, has_face, timing = service.components(item, 'H3', compare_both=True)
            record = {'utt_id': item['utt_id'], 'speaker_id': item['speaker_id'],
                'component_probs': probabilities, 'timing': timing, 'systems': {
                    system: service.decide(probabilities, system, has_face=has_face,
                        scores=item['face']['track_scores']) for system in ('H3', 'H_ref')}}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            if (i + 1) % 25 == 0:
                stream.flush(); storage_gate()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(todo)})
    # Preserve every existing byte, not merely numerical equivalence.
    combined = source.read_bytes() + (output / 'new_predictions.jsonl').read_bytes()
    (output / 'predictions.jsonl').write_bytes(combined)
    assert combined[:source.stat().st_size] == source.read_bytes()
    assert len(rows(output / 'predictions.jsonl')) == 1929
    finish(output, meta, {'status': 'completed', 'n': 1929, 'inferred_n': 745, 'reused_n': 1184,
        'new_predictions': spec(output / 'new_predictions.jsonl'),
        'predictions': spec(output / 'predictions.jsonl'), 'reused_prefix': spec(source),
        'inputs': spec(inputs_path), 'old_prefix_bytes_identical': True,
        'training': False, 'report_only': True, 'service': spec('src/emotioncore/x06/service.py')})


if __name__ == '__main__':
    run()
