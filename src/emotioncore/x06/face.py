"""Incremental X02 detection/tracking, retaining the original crop operations."""
import math
import time
from pathlib import Path
from emotioncore.x02 import face as original


class Tracker:
    def __init__(self, detector):
        self.detector = detector
        self.tracks = []
        self.encoded = {}
        self.requested = 0
        self.zero_faces = 0
        self.multiple_faces = 0
        self.seconds = 0.

    def add(self, frame, frame_index):
        import cv2
        tic = time.monotonic()
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        boxes = [] if faces is None else [tuple(map(float, row[:4])) for row in faces]
        original.add_detections(self.tracks, frame_index, boxes)
        self.zero_faces += not boxes
        self.multiple_faces += len(boxes) >= 2
        ok, compressed = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            raise RuntimeError('Frame compression failed')
        self.encoded[frame_index] = compressed.tobytes()
        self.requested = max(self.requested, frame_index + 1)
        self.seconds += time.monotonic() - tic

    def finish(self, directory):
        import cv2
        import numpy as np
        tic = time.monotonic()
        ranked = original.rank_tracks(self.tracks, len(self.encoded))
        result = {'status': 'no_face', 'requested_frames': self.requested,
                  'decoded_frames': len(self.encoded), 'track_scores': [t['target_score'] for t in ranked],
                  'tracks': len(ranked), 'multi_face_frames': self.multiple_faces,
                  'crop_paths': [], 'whole_paths': []}
        if not ranked:
            result['face_seconds'] = self.seconds + time.monotonic() - tic
            return result
        target = ranked[0]
        for index in original.selected_indices(sorted(target['boxes'])):
            frame = cv2.imdecode(np.frombuffer(self.encoded[index], dtype=np.uint8), cv2.IMREAD_COLOR)
            x, y, w, h = target['boxes'][index]
            height, width = frame.shape[:2]
            left, top = max(0, math.floor(x - original.CROP_EXPANSION * w)), max(0, math.floor(y - original.CROP_EXPANSION * h))
            right = min(width, math.ceil(x + (1 + original.CROP_EXPANSION) * w))
            bottom = min(height, math.ceil(y + (1 + original.CROP_EXPANSION) * h))
            if right <= left or bottom <= top:
                continue
            crop = cv2.resize(frame[top:bottom, left:right], (448, 448))
            scale = math.sqrt(50176 / (width * height))
            whole = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            crop_path, whole_path = directory / f'crop_{index:04d}.jpg', directory / f'whole_{index:04d}.jpg'
            original._write_jpg(crop_path, crop, cv2)
            original._write_jpg(whole_path, whole, cv2)
            result['crop_paths'].append(str(crop_path))
            result['whole_paths'].append(str(whole_path))
        result.update(status='ok' if result['crop_paths'] else 'failed',
                      target_track=target['id'], target_score=target['target_score'],
                      target_presence_fraction=target['presence_fraction'],
                      face_seconds=self.seconds + time.monotonic() - tic)
        return result


def detector():
    import cv2
    cv2.setNumThreads(1)
    path = Path('/home/administrator/projects/emotion/artifacts/perception_setup/weights/face_detection_yunet_2023mar.onnx')
    return cv2.FaceDetectorYN.create(str(path), '', (320, 320), .6, .3, 50)


def clip(path, duration, directory, wait_until=None):
    """Correct-boundary path uses the same 5-fps seeks as X02, without a transcode."""
    import cv2
    reader = cv2.VideoCapture(str(path))
    tracker = Tracker(detector())
    decode = 0.
    try:
        requested = max(1, math.ceil(duration * original.FPS))
        for index in range(requested):
            if wait_until:
                wait_until(index / original.FPS)
            tic = time.monotonic()
            reader.set(cv2.CAP_PROP_POS_MSEC, index * 1000. / original.FPS)
            ok, frame = reader.read()
            decode += time.monotonic() - tic
            if ok:
                tracker.add(frame, index)
        tracker.requested = requested
    finally:
        reader.release()
    result = tracker.finish(directory)
    result['decode_seconds'] = decode
    return result
