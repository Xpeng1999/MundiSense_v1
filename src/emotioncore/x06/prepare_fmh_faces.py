"""Recompute FMH track scores with X02 operations; no GPU or fitted parameters."""
import concurrent.futures
from emotioncore.x06.common import *
from emotioncore.x06.face import clip


def one(row):
    directory = ROOT / 'fmh_faces' / row['utt_id']
    result = clip(MEDIA / row['mp4']['path'], row['mp4']['duration_s'], directory)
    return {'utt_id': row['utt_id'], 'speaker_id': row['speaker_id'], 'face': result}


def run():
    output = ROOT / 'fmh_faces'
    meta = start(output)
    selected = [r for r in rows(MANIFEST) if r['speaker_id'] in FMH]
    assert len(selected) == 3266
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(one, selected)):
            results.append(result)
            if (i + 1) % 25 == 0:
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(selected)})
    write_rows(output / 'faces.jsonl', results)
    finish(output, meta, {'n': len(results), 'faces': spec(output / 'faces.jsonl'), 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
