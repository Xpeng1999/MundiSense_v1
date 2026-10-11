"""Correct utterance boundaries on actual S1 waveform and global video grid."""
import concurrent.futures
from functools import partial
from emotioncore.x06.common import *
from emotioncore.x06.prepare_segments import one


def run():
    name = 'stream_oracle_inputs'
    output = ROOT / name
    meta = start(output)
    selected = []
    for speaker in sorted(TEST):
        timeline = read(ROOT / f'preflight/timeline_{speaker}.json')
        selected.extend({'segment_id': entry['utt_id'], 'speaker_id': speaker, 'cohort': 'et_test',
            'start_s': entry['start_s'], 'end_s': entry['end_s'], 'decision_s': entry['end_s'],
            'source': 'speech_segment', 'visual_only': False}
            for entry in timeline['entries'] if entry['kind'] == 'utterance')
    assert len(selected) == 1929
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        for i, result in enumerate(pool.map(partial(one, output_name=name), selected)):
            results.append(result)
            if (i + 1) % 25 == 0:
                storage_gate()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(selected)})
    write_rows(output / 'inputs.jsonl', results)
    finish(output, meta, {'n': len(results), 'inputs': spec(output / 'inputs.jsonl'), 'gpu_seconds': 0,
        'input_scope': 'actual S1 waveform and global 5fps frames, correct original utterance intervals',
        'parameters_selected': False, 'emotion_labels_read_for_selection': False})


if __name__ == '__main__':
    run()
