"""CPU-only 0.4s media preparation from the immutable FSMN posterior cache."""
import concurrent.futures
from functools import partial
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, checked
from emotioncore.x06.vad import segments
from emotioncore.x06.online_vad import LiveEvents, silence_windows
from emotioncore.x06.event_audit import signature
from emotioncore.x06.prepare_segments import one


def cuts():
    import soundfile as sf
    output = REV / 'cuts_04'; meta = start(output)
    frozen = read('configs/x06_vad_freeze.json')['selected']
    assert frozen['model'] == 'fsmn' and frozen['threshold'] == .3
    result = {}
    old = read(ROOT / 'revision_1_budget/summary.json')
    for speaker in sorted(FMH | TEST):
        source = old['posterior_sources'][speaker]
        probability = np.load(checked(source))
        info = sf.info(ROOT / f'streams/{speaker}.wav'); duration = info.frames / info.samplerate
        voice = segments(probability, .3, .4)
        windows = silence_windows(voice, duration)
        records = sorted([dict(v, source='speech_segment', visual_only=False) for v in voice] + windows,
                         key=lambda r: (r['decision_s'], r['start_s']))
        machine = LiveEvents(.3, .4)
        live = [e for p in probability for e in machine.advance(p)] + machine.finish(duration)
        if signature(live) != signature(records):
            raise ValueError('STOP 0.4s online/fullspeed schedule mismatch')
        for i, row in enumerate(records):
            row.update(segment_id=f'{speaker}_{i:05d}', speaker_id=speaker,
                       cohort='fmh' if speaker in FMH else 'et_test')
        write_rows(output / f'{speaker}.jsonl', records)
        result[speaker] = {'speech_segments': len(voice), 'silence_windows': len(windows),
            'segments': spec(output / f'{speaker}.jsonl'), 'posterior': source, 'schedule_identical': True}
    finish(output, meta, {'streams': result, 'min_silence': .4, 'label_reads': False,
                         'model_inference': False, 'gpu_seconds': 0})


def inputs():
    output = REV / 'inputs_04'; meta = start(output)
    selected = [r for source in read(REV / 'cuts_04/summary.json')['streams'].values()
                for r in rows(checked(source['segments']))]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        for i, result in enumerate(pool.map(partial(one, output_name='revision2/inputs_04'), selected)):
            results.append(result)
            if (i + 1) % 25 == 0:
                storage_gate()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(selected)})
    write_rows(output / 'inputs.jsonl', results)
    finish(output, meta, {'n': len(results), 'inputs': spec(output / 'inputs.jsonl'),
        'cuts': spec(REV / 'cuts_04/summary.json'), 'gpu_seconds': 0, 'cpu_workers': 16})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('cuts', 'inputs'))
    {'cuts': cuts, 'inputs': inputs}[parser.parse_args().phase]()
