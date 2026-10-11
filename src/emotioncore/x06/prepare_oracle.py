"""CPU-only fresh oracle-boundary face processing before GPU equivalence."""
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from emotioncore.x02.runtime import PersistentJSONWorker, write_json_atomic
from emotioncore.x06.common import *
from emotioncore.x06.face import clip


def one(row):
    path = ROOT / 'oracle_inputs' / row['utt_id']
    face = clip(MEDIA / row['mp4']['path'], row['mp4']['duration_s'], path / 'face')
    face.update(id=row['utt_id'], input_sha256=row['mp4']['sha256'])
    face_path = path / 'face.json'
    write_json_atomic(face_path, face)
    video = None
    if face['status'] == 'ok':
        worker = PersistentJSONWorker([sys.executable, '-u', '-m', 'emotioncore.x03.media', '--worker',
            '--detector', '/home/administrator/projects/emotion/artifacts/perception_setup/weights/face_detection_yunet_2023mar.onnx'],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=Path.cwd())
        try:
            video = worker.request({'face_receipt': str(face_path), 'utt_id': row['utt_id'],
                    'input_sha256': row['mp4']['sha256'], 'duration_s': row['mp4']['duration_s'],
                    'output': str(path / 'q_video.mp4')}, 120)
        finally:
            worker.close()
    result = {'utt_id': row['utt_id'], 'speaker_id': row['speaker_id'],
              'wav': str(MEDIA / row['wav']['path']), 'wav_sha256': row['wav']['sha256'],
              'duration_s': row['mp4']['duration_s'], 'face': face, 'q_video': video}
    write_json_atomic(path / 'input.json', result)
    return result


def run():
    output = ROOT / 'oracle_inputs'
    meta = start(output)
    selected = [r for r in rows(MANIFEST) if r['speaker_id'] in TEST and r['official_split'] == 'test']
    assert len(selected) == 1929
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        receipts = []
        for i, result in enumerate(pool.map(one, selected)):
            receipts.append(result)
            if (i + 1) % 25 == 0:
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(selected)})
    path = output / 'inputs.jsonl'
    write_rows(path, receipts)
    missing = set(read('configs/x03_face_missing_policy.json')['face_missing_utt_ids'])
    changed = [r['utt_id'] for r in receipts
               if (r['q_video'] is None or r['q_video']['status'] != 'ok') != (r['utt_id'] in missing)]
    finish(output, meta, {'n': len(receipts), 'inputs': spec(path), 'availability_differences': changed,
                          'face_pipeline_source': spec('src/emotioncore/x02/face.py'), 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
