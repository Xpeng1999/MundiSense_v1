"""Frozen-model FMH correct-boundary baseline on the same S1 input adapter."""
import os
from emotioncore.x06.common import *
from emotioncore.x06.service import Service


def run():
    if os.environ.get('X06_GPU_GUARDED') != 'fmh_stream_oracle':
        raise RuntimeError('Use X06 GPU watchdog')
    output = ROOT / 'fmh_stream_oracle'
    meta = start(output)
    source = read(ROOT / 'fmh_stream_oracle_inputs/summary.json')['inputs']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('STOP prepared oracle input SHA changed')
    inputs = rows(source['path'])
    if len(inputs) != 3266 or {r['speaker_id'] for r in inputs} != set(FMH):
        raise ValueError('STOP FMH oracle population mismatch')
    service = Service()
    with (output / 'predictions.jsonl').open('w') as stream:
        for i, item in enumerate(inputs):
            probability, face, timing = service.components(item, 'H3', compare_both=True)
            result = {'utt_id': item['utt_id'], 'speaker_id': item['speaker_id'],
                'component_probs': probability, 'has_face': face, 'track_scores': item['face']['track_scores'],
                'systems': {name: service.decide(probability, name, has_face=face, scores=item['face']['track_scores'])
                            for name in ('H3', 'H_ref')}, 'timing': timing}
            stream.write(json.dumps(result) + '\n')
            if (i + 1) % 25 == 0:
                stream.flush(); storage_gate()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(inputs)})
    finish(output, meta, {'n': len(inputs), 'inputs': source, 'predictions': spec(output / 'predictions.jsonl'),
        'training': False, 'parameters_selected': False, 'input_scope': 'actual S1 streams'})


if __name__ == '__main__':
    run()
