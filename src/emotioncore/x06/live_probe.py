"""CPU pipeline integration check on one deterministically selected FMH clip."""
import multiprocessing as mp
import queue
from emotioncore.x06.common import *
from emotioncore.x06.live import producer
from emotioncore.x06.streams import population, render_audio, Video
from emotioncore.x06.simulator import build_timelines
from emotioncore.x06.face import Tracker, detector
from emotioncore.x06.prepare_segments import frame_index


def run():
    output = ROOT / 'live_cpu_probe_v2'
    meta = start(output)
    candidates = sorted((r for r in population() if r['speaker_id'] in FMH and
                         r['mp4']['duration_s'] <= 30), key=lambda r: (-r['mp4']['duration_s'], r['utt_id']))
    if not candidates:
        raise RuntimeError('No bounded FMH integration-test clip')
    row = candidates[0]
    selected = [r for r in candidates if r['speaker_id'] == row['speaker_id']][:2]
    timeline = build_timelines(selected)[row['speaker_id']]
    timeline_path = output / 'timeline.json'
    write_json_atomic(timeline_path, timeline)
    audio = render_audio(timeline, output / 'source.wav')
    sources = {row['speaker_id']: {'timeline': spec(timeline_path), 'audio': audio['audio']}}
    context = mp.get_context('spawn')
    messages = context.Queue(maxsize=2); begin = context.Event()
    config = {'model': 'silero', 'threshold': .5, 'min_silence': .2}
    process = context.Process(target=producer, args=(sources, config, output, messages, begin, False))
    process.start()
    verified = []
    try:
        assert messages.get(timeout=300)['kind'] == 'ready'
        begin.set()
        while True:
            try:
                message = messages.get(timeout=10)
            except queue.Empty:
                if not process.is_alive():
                    raise RuntimeError('CPU producer terminated unexpectedly')
                continue
            if message['kind'] == 'error':
                raise RuntimeError(message['traceback'])
            if message['kind'] == 'done':
                break
            if message['kind'] != 'segment':
                continue
            item = message['item']
            tracker = Tracker(detector()); reader = Video(timeline)
            first, last = frame_index(item['start_s']), frame_index(item['end_s'])
            try:
                for index in range(first, last):
                    tracker.add(reader.frame(index / 5), index - first)
            finally:
                reader.close()
            expected = tracker.finish(output / 'independent_faces' / item['segment_id'])
            assert expected['track_scores'] == item['face']['track_scores']
            assert [sha256(p) for p in expected['crop_paths']] == [sha256(p) for p in item['face']['crop_paths']]
            verified.append({'segment_id': item['segment_id'], 'source': item['source'],
                             'face_status': expected['status'], 'crop_bytes_identical': True})
        process.join(timeout=30)
        assert process.exitcode == 0 and verified
    finally:
        if process.is_alive():
            process.terminate(); process.join(timeout=10)
    finish(output, meta, {'source_utt_ids': [r['utt_id'] for r in selected], 'source_speaker': row['speaker_id'],
                          'segments': verified, 'config': config, 'parameter_selection': False,
                          'purpose': 'online cached frames versus independent frozen face operations', 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
