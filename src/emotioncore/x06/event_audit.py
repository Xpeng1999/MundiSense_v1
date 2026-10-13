"""Verify live endpoint/window schedules against frozen full-speed cuts on CPU."""
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.online_vad import LiveEvents


def signature(records):
    return [(r['source'], round(r['start_s'], 8), round(r['end_s'], 8), round(r['decision_s'], 8)) for r in records]


def run():
    import soundfile as sf
    output = ROOT / 'event_schedule_audit'
    meta = start(output)
    config = read('configs/x06_vad_freeze.json')['selected']
    results = {}
    for speaker in sorted(FMH | TEST):
        machine = LiveEvents(config['threshold'], config['min_silence'])
        path = ROOT / f'cuts/{speaker}_posterior.npy'
        probability = np.load(path)
        info = sf.info(ROOT / f'streams/{speaker}.wav')
        duration = info.frames / info.samplerate
        actual = [event for p in probability for event in machine.advance(p)] + machine.finish(duration)
        expected = rows(ROOT / f'cuts/{speaker}.jsonl')
        if signature(actual) != signature(expected):
            write_json_atomic(output / f'{speaker}_difference.json', {'actual': signature(actual), 'expected': signature(expected)})
            raise AssertionError(f'Live event schedule differs for {speaker}')
        results[speaker] = {'events': len(actual), 'speech': sum(not r['visual_only'] for r in actual),
                            'silence_windows': sum(r['visual_only'] for r in actual),
                            'schedule_identical': True, 'posterior': spec(path)}
    finish(output, meta, {'streams': results, 'gpu_seconds': 0, 'emotion_label_reads': False})


if __name__ == '__main__':
    run()
