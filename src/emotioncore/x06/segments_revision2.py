"""Frozen full-speed service for each registered minimum-silence candidate."""
import argparse
import os
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, checked
from emotioncore.x06.service import Service


def run(silence):
    key = '04' if silence == .4 else '08'
    if os.environ.get('X06_GPU_GUARDED') != 'rev2_s2_' + key:
        raise RuntimeError('Use X06 GPU watchdog')
    output = REV / ('emotion_' + key); meta = start(output)
    prepared = REV / 'inputs_04' if silence == .4 else ROOT / 'segment_inputs'
    source = read(prepared / 'summary.json')['inputs']; inputs = rows(checked(source))
    expected = read(ROOT / 'revision_1_budget/summary.json')['counts'][str(silence)]
    if len(inputs) != expected['speech'] + expected['visual']:
        raise ValueError('STOP VAD population mismatch')
    if any(r['speaker_id'] not in FMH | TEST or r['cohort'] != ('fmh' if r['speaker_id'] in FMH else 'et_test') for r in inputs):
        raise ValueError('STOP unauthorized VAD input cohort')
    service = Service()
    with (output / 'predictions.jsonl').open('w') as stream:
        for i, item in enumerate(inputs):
            speech = not item['visual_only']
            p, face, timing = service.components(item, 'H3', has_speech=speech, compare_both=speech)
            decisions = {name: service.decide(p, name, has_face=face, has_speech=speech,
                         scores=item['face']['track_scores']) for name in (('H3', 'H_ref') if speech else ('H_V',))}
            result = {k: item[k] for k in ('segment_id', 'speaker_id', 'cohort', 'start_s', 'end_s', 'decision_s', 'source', 'visual_only')}
            result.update(component_probs=p, has_face=face, track_scores=item['face']['track_scores'],
                          systems=decisions, timing=timing, min_silence=silence, timestamp=now())
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            if (i + 1) % 25 == 0:
                stream.flush(); storage_gate()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(inputs)})
    finish(output, meta, {'status': 'completed', 'n': len(inputs), 'min_silence': silence,
        'predictions': spec(output / 'predictions.jsonl'), 'inputs': source, 'training': False,
        'component_sharing': 'common forwards shared only within H3/H_ref for this configuration'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--silence', type=float, choices=(.4, .8), required=True)
    run(parser.parse_args().silence)
