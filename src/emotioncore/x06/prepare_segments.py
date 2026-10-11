"""Full-speed CPU media preparation using the exact online 5fps sample grid."""
import concurrent.futures
import hashlib
import math
import os
import sys
import time
from emotioncore.x02.runtime import PersistentJSONWorker
from emotioncore.x06.common import *
from emotioncore.x06.face import Tracker, detector
from emotioncore.x06.streams import Video

DETECTOR = '/home/administrator/projects/emotion/artifacts/perception_setup/weights/face_detection_yunet_2023mar.onnx'


def frame_index(when):
    # VAD boundaries are exact 10ms ticks; remove binary float representation
    # noise at a coincident 200ms video boundary, without shifting any frame.
    return math.ceil(when * 5 - 1e-9)


def q_media(face, directory, segment_id, identity, duration):
    face.update(id=segment_id, input_sha256=identity)
    path = directory / 'face.json'
    write_json_atomic(path, face)
    if face['status'] != 'ok':
        return None
    worker = PersistentJSONWorker([sys.executable, '-u', '-m', 'emotioncore.x03.media', '--worker',
            '--detector', DETECTOR], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=Path.cwd())
    try:
        return worker.request({'face_receipt': str(path), 'utt_id': segment_id,
                'input_sha256': identity, 'duration_s': duration,
                'output': str(directory / 'q_video.mp4')}, 120)
    finally:
        worker.close()


def one(segment, output_name='segment_inputs'):
    import soundfile as sf
    speaker = segment['speaker_id']
    directory = ROOT / output_name / segment['segment_id']
    directory.mkdir(parents=True, exist_ok=False)
    timeline_path = ROOT / f'preflight/timeline_{speaker}.json'
    timeline = read(timeline_path)
    identity = hashlib.sha256((sha256(timeline_path) + json.dumps(segment, sort_keys=True)).encode()).hexdigest()
    video = Video(timeline)
    tracker = Tracker(detector())
    first, last = frame_index(segment['start_s']), frame_index(segment['end_s'])
    decode = 0.
    try:
        for index in range(first, last):
            tic = time.monotonic()
            frame = video.frame(index / 5)
            decode += time.monotonic() - tic
            tracker.add(frame, index - first)
    finally:
        video.close()
    face = tracker.finish(directory / 'face')
    face['decode_seconds'] = decode
    face['source_terminal_frame_clamps'] = video.terminal_frame_clamps
    duration = segment['end_s'] - segment['start_s']
    q_video = q_media(face, directory, segment['segment_id'], identity, duration)
    wav_path = directory / 'segment.wav'
    with sf.SoundFile(ROOT / f'streams/{speaker}.wav') as source:
        source.seek(round(segment['start_s'] * 16000))
        audio = source.read(round(segment['end_s'] * 16000) - round(segment['start_s'] * 16000), dtype='float32')
    sf.write(wav_path, audio, 16000, subtype='FLOAT')
    result = {**segment, 'utt_id': segment['segment_id'], 'duration_s': duration,
              'wav': str(wav_path), 'wav_sha256': sha256(wav_path), 'face': face, 'q_video': q_video,
              'identity': identity, 'source_timeline': spec(timeline_path)}
    write_json_atomic(directory / 'input.json', result)
    return result


def run():
    output = ROOT / 'segment_inputs'
    meta = start(output)
    cuts = read(ROOT / 'cuts/summary.json')
    selected = [r for source in cuts['streams'].values() for r in rows(source['segments']['path'])]
    results = []
    # Offline preparation only; real-time replay retains one causal producer.
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as pool:
        for i, result in enumerate(pool.map(one, selected)):
            results.append(result)
            if (i + 1) % 25 == 0:
                storage_gate()
                write_json_atomic(output / 'progress.json', {'done': i + 1, 'total': len(selected)})
    write_rows(output / 'inputs.jsonl', results)
    finish(output, meta, {'n': len(results), 'inputs': spec(output / 'inputs.jsonl'),
                          'cuts': spec(ROOT / 'cuts/summary.json'), 'gpu_seconds': 0, 'cpu_workers': 16})


if __name__ == '__main__':
    run()
