"""Fresh full-speed segment inference; common components serve both full systems."""
import os
from emotioncore.x06.common import *
from emotioncore.x06.service import Service


def run():
    if os.environ.get('X06_GPU_GUARDED') != 'emotion_segments':
        raise RuntimeError('Use X06 GPU watchdog')
    if read(ROOT / 'budget_recheck/summary.json')['status'] != 'passed':
        raise RuntimeError('STOP remaining-budget estimate has not passed')
    output = ROOT / 'emotion_segments'
    meta = start(output)
    source = read(ROOT / 'segment_inputs/summary.json')['inputs']
    if sha256(source['path']) != source['sha256']:
        raise ValueError('STOP prepared input SHA changed')
    inputs = rows(source['path'])
    service = Service()
    with (output / 'predictions.jsonl').open('w') as stream:
        for i, item in enumerate(inputs):
            speech = not item['visual_only']
            p, face, timing = service.components(item, 'H3', has_speech=speech, compare_both=speech)
            decisions = {name: service.decide(p, name, has_face=face, has_speech=speech,
                        scores=item['face']['track_scores']) for name in (('H3', 'H_ref') if speech else ('H_V',))}
            result = {k: item[k] for k in ('segment_id', 'speaker_id', 'cohort', 'start_s', 'end_s', 'decision_s', 'source', 'visual_only')}
            result.update(component_probs=p, has_face=face, track_scores=item['face']['track_scores'],
                          systems=decisions, timing=timing, timestamp=now())
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            if (i + 1) % 25 == 0:
                stream.flush()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(inputs)})
    finish(output, meta, {'n': len(inputs), 'predictions': spec(output / 'predictions.jsonl'),
                          'inputs': source, 'training': False, 'component_sharing': 'H_ref union contains H3; no repeated common forwards'})


if __name__ == '__main__':
    run()
