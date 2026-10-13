"""Apply the train-frozen VAD to the seven continuous streams without labels."""
import math
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.service import Service
from emotioncore.x06.vad import Posterior, segments
from emotioncore.x06.online_vad import posterior_chunks, Endpoints, silence_windows


def run():
    output = ROOT / 'cuts'
    meta = start(output)
    frozen = read(ROOT / 'aishell_train_parallel/freeze.json')['selected']
    model = Posterior(frozen['model'])
    streams = read(ROOT / 'streams/summary.json')['streams']
    result = {}
    for speaker, source in streams.items():
        signal = Service.audio(source['audio']['path'])
        machine = Endpoints(frozen['threshold'], frozen['min_silence'])
        probabilities = []; voice = []; elapsed = 0.
        for chunk in posterior_chunks(model, signal):
            probabilities.extend(chunk['probabilities'].tolist())
            elapsed += chunk['vad_seconds']
            for p in chunk['probabilities']:
                event = machine.push(p)
                if event:
                    event['available_s'] = max(event['decision_s'], chunk['available_s'])
                    voice.append(event)
        final = machine.finish()
        if final:
            final['available_s'] = len(signal) / 16000
            voice.append(final)
        reference = segments(probabilities, frozen['threshold'], frozen['min_silence'])
        assert [{k: v for k, v in r.items() if k != 'available_s'} for r in voice] == reference
        silence = silence_windows(voice, len(signal) / 16000)
        all_segments = sorted([dict(r, source='speech_segment', visual_only=False) for r in voice] + silence,
                              key=lambda r: (r['decision_s'], r['start_s']))
        for i, event in enumerate(all_segments):
            event.update(segment_id=f'{speaker}_{i:05d}', speaker_id=speaker,
                         cohort='fmh' if speaker in FMH else 'et_test')
        write_rows(output / f'{speaker}.jsonl', all_segments)
        np.save(output / f'{speaker}_posterior.npy', np.asarray(probabilities, dtype=np.float32))
        result[speaker] = {'speech_segments': len(voice), 'silence_windows': len(silence),
                           'vad_cpu_s': elapsed, 'segments': spec(output / f'{speaker}.jsonl')}
        write_json_atomic(output / 'progress.json', result)
    finish(output, meta, {'streams': result, 'vad_freeze': spec(ROOT / 'aishell_train_parallel/freeze.json'),
                          'label_reads': False, 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
