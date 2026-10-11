"""Paced input producer and one resident GPU consumer; never replay predictions."""
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
import numpy as np
from emotioncore.x02 import face as original_face
from emotioncore.x06.common import *
from emotioncore.x06.face import Tracker, detector
from emotioncore.x06.streams import Video
from emotioncore.x06.online_vad import posterior_chunks, LiveEvents
from emotioncore.x06.vad import Posterior
from emotioncore.x06.service import Service
from emotioncore.x06.prepare_segments import q_media, frame_index
from emotioncore.x06.decision import apply


def rss_tree(pid):
    import psutil
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(p.memory_info().rss for p in processes if p.is_running())
    except psutil.NoSuchProcess:
        return 0


def producer(sources, config, output, messages, begin_event, realtime):
    import cv2
    import soundfile as sf
    try:
        model = Posterior(config['model'])
        face_detector = detector()
        messages.put({'kind': 'ready', 'pid': os.getpid()})
        begin_event.wait()
        for speaker, source in sources.items():
            timeline = read(source['timeline']['path'])
            audio_started = time.monotonic()
            signal, sample_rate = sf.read(source['audio']['path'], dtype='float32')
            if sample_rate != 16000 or signal.ndim != 1:
                raise ValueError('STOP generated streams must be mono 16kHz')
            audio_load_seconds = time.monotonic() - audio_started
            video = Video(timeline)
            machine = LiveEvents(config['threshold'], config['min_silence'])
            cache = {}; next_frame = 0; event_index = 0
            started = time.monotonic()
            totals = {'vad': 0., 'decode': 0., 'face_detection': 0.}

            def advance(when):
                nonlocal next_frame
                while next_frame / 5 < min(when, timeline['duration_s']):
                    frame_time = next_frame / 5
                    if realtime:
                        time.sleep(max(0., started + frame_time - time.monotonic()))
                    tic = time.monotonic()
                    frame = video.frame(frame_time)
                    decode = time.monotonic() - tic
                    tic = time.monotonic()
                    height, width = frame.shape[:2]
                    face_detector.setInputSize((width, height))
                    _, faces = face_detector.detect(frame)
                    boxes = [] if faces is None else [tuple(map(float, r[:4])) for r in faces]
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if not ok:
                        raise RuntimeError('Cannot cache input frame')
                    face_s = time.monotonic() - tic
                    cache[next_frame] = (encoded.tobytes(), boxes, decode, face_s)
                    totals['decode'] += decode; totals['face_detection'] += face_s
                    next_frame += 1
                # The maximum speech window is 15s, plus endpoint silence and
                # at most 1s frontend input chunks; retain 20s of frame receipts.
                for index in list(cache):
                    if index / 5 < when - 20:
                        del cache[index]
                if realtime:
                    time.sleep(max(0., started + when - time.monotonic()))

            def prepare(event, endpoint, vad_s):
                nonlocal event_index
                before = time.monotonic()
                identifier = f'{speaker}_{event_index:05d}'
                event_index += 1
                directory = output / 'inputs' / identifier
                directory.mkdir(parents=True, exist_ok=False)
                tracker = Tracker(None)
                first, last = frame_index(event['start_s']), frame_index(event['end_s'])
                decoded = face_s = 0.
                for index in range(first, last):
                    if index not in cache:
                        raise RuntimeError(f'Frame cache gap at {speaker}:{index}')
                    encoded, boxes, decode, detection = cache[index]
                    original_face.add_detections(tracker.tracks, index - first, boxes)
                    tracker.encoded[index - first] = encoded
                    tracker.multiple_faces += len(boxes) >= 2
                    tracker.zero_faces += not boxes
                    decoded += decode; face_s += detection
                tracker.requested = last - first
                tracker.seconds = face_s
                face = tracker.finish(directory / 'face')
                face['decode_seconds'] = decoded
                identity = __import__('hashlib').sha256((source['timeline']['sha256'] + json.dumps(event, sort_keys=True)).encode()).hexdigest()
                duration = event['end_s'] - event['start_s']
                q_video = q_media(face, directory, identifier, identity, duration)
                path = directory / 'segment.wav'
                audio = signal[round(event['start_s'] * 16000):round(event['end_s'] * 16000)]
                sf.write(path, audio, 16000, subtype='FLOAT')
                item = {**event, 'utt_id': identifier, 'segment_id': identifier, 'speaker_id': speaker,
                        'wav': str(path), 'wav_sha256': sha256(path), 'duration_s': duration,
                        'face': face, 'q_video': q_video, 'endpoint_monotonic': endpoint,
                        'input_stream_started_monotonic': started,
                        'timing_input': {'vad': vad_s, 'decode': decoded,
                            'face_detection_tracking': face['face_seconds'],
                            'after_endpoint_preparation': time.monotonic() - before}}
                write_json_atomic(directory / 'input.json', item)
                messages.put({'kind': 'segment', 'item': item})

            try:
                for chunk in posterior_chunks(model, signal, wait_until=advance):
                    totals['vad'] += chunk['vad_seconds']
                    ready = [event for p in chunk['probabilities'] for event in machine.advance(p)]
                    if ready and realtime:
                        ready_time = min(len(signal) / 16000, max(r['decision_s'] for r in ready))
                        time.sleep(max(0., started + ready_time - time.monotonic()))
                    endpoint = time.monotonic()
                    for event in ready:
                        prepare(event, endpoint, chunk['vad_seconds'] / max(1, len(ready)))
                final = machine.finish(len(signal) / 16000)
                endpoint = time.monotonic()
                for event in final:
                    prepare(event, endpoint, 0.)
            finally:
                video.close()
            messages.put({'kind': 'stream_finished', 'speaker_id': speaker,
                          'input_duration_s': timeline['duration_s'], 'events': event_index,
                          'producer_wall_s': time.monotonic() - started, 'totals': totals,
                          'audio_load_seconds': audio_load_seconds})
        messages.put({'kind': 'done', 'resident_rss_before_exit': rss_tree(os.getpid())})
    except BaseException:
        messages.put({'kind': 'error', 'traceback': traceback.format_exc()})
        raise


def quantiles(values):
    return {f'p{q}': float(np.percentile(values, q)) if values else None for q in (50, 95, 99)}


def consume(sources, system, output, event_path, alpha, realtime=True):
    context = mp.get_context('spawn')
    messages = context.Queue(maxsize=2)
    begin_event = context.Event()
    config = read(ROOT / 'aishell_train_parallel/freeze.json')['selected']
    child = context.Process(target=producer, args=(sources, config, output, messages, begin_event, realtime))
    child.start()
    service = None
    try:
        service = Service()
        first = messages.get(timeout=300)
        if first['kind'] != 'ready':
            raise RuntimeError(first)
        baseline = rss_tree(os.getpid()); peak_rss = baseline
        service.torch.cuda.reset_peak_memory_stats()
        state = {}; latencies = []; streams = {}; count = 0
        stages = {}; voice_latencies = []; visual_latencies = []
        begin = time.monotonic(); begin_event.set()
        with event_path.open('w') as events, (output / 'latency_receipts.jsonl').open('w') as receipts:
            while True:
                peak_rss = max(peak_rss, rss_tree(os.getpid()))
                try:
                    message = messages.get(timeout=1)
                except queue.Empty:
                    if not child.is_alive():
                        raise RuntimeError(f'Input producer exited {child.exitcode}')
                    continue
                if message['kind'] == 'done':
                    producer_final_rss = message['resident_rss_before_exit']
                    break
                if message['kind'] == 'error':
                    raise RuntimeError(message['traceback'])
                if message['kind'] == 'stream_finished':
                    streams[message['speaker_id']] = message
                    continue
                item = message['item']
                speech = not item['visual_only']
                component, face, timing = service.components(item, system, has_speech=speech)
                tic = time.monotonic()
                decision = apply(service.bundle, component, system, has_face=face, speech=speech,
                                 scores=item['face']['track_scores'], alpha=alpha, state=state,
                                 stream_id=item['speaker_id'])
                timing['fusion'] = time.monotonic() - tic
                event = {k: item[k] for k in ('segment_id', 'speaker_id', 'source', 'start_s', 'end_s', 'decision_s', 'visual_only')}
                event.update(timestamp=now(), system=system, alpha=alpha, **decision)
                event.update(component_probs=component, has_face=face, track_scores=item['face']['track_scores'],
                             endpoint_monotonic=item['endpoint_monotonic'], timing={**item['timing_input'], **timing})
                events.write(json.dumps(event, ensure_ascii=False) + '\n'); events.flush()
                written = time.monotonic()
                latency = written - item['endpoint_monotonic']
                receipts.write(json.dumps({'segment_id': item['segment_id'], 'endpoint_monotonic': item['endpoint_monotonic'],
                            'after_flush_monotonic': written, 'latency_s': latency}) + '\n'); receipts.flush()
                latencies.append(latency)
                (voice_latencies if speech else visual_latencies).append(latency)
                for name, seconds in event['timing'].items():
                    stages.setdefault(name, []).append(seconds)
                count += 1
                if count % 25 == 0:
                    storage_gate()
                    write_json_atomic(output / 'progress.json', {'events_written': count, 'last_segment': item['segment_id'],
                        'last_latency_s': latency, 'rss_bytes': rss_tree(os.getpid())})
        wall = time.monotonic() - begin
        child.join(timeout=30)
        if child.exitcode != 0:
            raise RuntimeError(f'Producer failed with exit code {child.exitcode}')
        # Do not manufacture a memory reduction by letting the producer exit.
        end_rss = rss_tree(os.getpid()) + producer_final_rss
        peak_rss = max(peak_rss, end_rss)
        duration = sum(r['input_duration_s'] for r in streams.values())
        return {'system': system, 'alpha': alpha, 'realtime': realtime, 'events': count, 'streams': streams,
                'latency_seconds': quantiles(latencies), 'speech_latency_seconds': quantiles(voice_latencies),
                'visual_only_latency_seconds': quantiles(visual_latencies),
                'stage_seconds': {k: {**quantiles(v), 'total': sum(v)} for k, v in stages.items()},
                'stage_note': 'decode/detection describe full segments including work before endpoints; after_endpoint_preparation and GPU/fusion overlap no earlier input work',
                'wall_seconds': wall, 'input_seconds': duration, 'real_time_factor_wall_over_input': wall / duration,
                'rss_baseline_bytes': baseline, 'rss_end_bytes': end_rss, 'rss_sampled_peak_bytes': peak_rss,
                'rss_end_growth': (end_rss - baseline) / baseline, 'rss_peak_growth': (peak_rss - baseline) / baseline,
                'rss_scope': 'consumer and live descendants; baseline after both model loads before input; final includes producer resident RSS immediately before exit; peak sampled per event or one-second queue wait',
                'gpu_peak_allocated_bytes': service.torch.cuda.max_memory_allocated(),
                'gpu_peak_reserved_bytes': service.torch.cuda.max_memory_reserved(),
                'events_file': spec(event_path), 'latency_receipts': spec(output / 'latency_receipts.jsonl')}
    finally:
        if child.is_alive():
            child.terminate(); child.join(timeout=10)
