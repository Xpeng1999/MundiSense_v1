"""Sample-accurate audio and lazy video playback for the seven virtual streams."""
import bisect
import math
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.service import Service


def population():
    selected = [r for r in rows(MANIFEST) if r['speaker_id'] in FMH or
                (r['speaker_id'] in TEST and r['official_split'] == 'test')]
    if len(selected) != 5195 or len({r['utt_id'] for r in selected}) != 5195:
        raise ValueError('STOP unexpected X06 population')
    return selected


def validate_input(path):
    """This experimental entry accepts only the registered FMH/ET-test media."""
    path = Path(path).resolve()
    allowed = {str((MEDIA / r['mp4']['path']).resolve()): r for r in population()}
    if str(path) not in allowed:
        raise ValueError('STOP input outside the X06 FMH/ET-test allowlist')
    row = allowed[str(path)]
    if sha256(path) != row['mp4']['sha256']:
        raise ValueError('STOP source media SHA changed')
    return row


def render_audio(timeline, path):
    import soundfile as sf
    noise_scale = 0.; adjustments = []; gap_levels = []
    with sf.SoundFile(path, 'w', samplerate=16000, channels=1, subtype='FLOAT') as output:
        cursor = 0
        for entry in timeline['entries']:
            expected = round(entry['start_s'] * 16000)
            assert expected == cursor
            count = round(entry['end_s'] * 16000) - cursor
            if entry['kind'] == 'utterance':
                source = MEDIA / entry['audio']['path']
                if sha256(source) != entry['audio']['sha256']:
                    raise ValueError('STOP source audio SHA changed')
                audio = Service.audio(source)
                edge = np.r_[audio[:1600], audio[-1600:]].astype(np.float64)
                noise_scale = float(np.sqrt(np.mean(edge * edge)))
                adjustments.append({'utt_id': entry['utt_id'], 'source_samples_16k': len(audio),
                                    'stream_samples': count, 'padding_or_crop_samples': count - len(audio)})
                signal = np.pad(audio[:count], (0, max(0, count - len(audio))))
            else:
                signal = np.random.default_rng(entry['noise_seed']).normal(0, noise_scale, count).astype(np.float32)
                gap_levels.append({'start_s': entry['start_s'], 'rms_reference': noise_scale,
                                   'noise_seed': entry['noise_seed']})
            output.write(signal)
            cursor += count
    return {'audio': spec(path), 'samples': cursor, 'audio_adjustments': adjustments,
            'gap_noise': gap_levels, 'sample_boundary_tolerance_s': .5 / 16000}


class Video:
    """No pixel re-encoding before X02 processing; gaps hold the last source frame."""
    def __init__(self, timeline):
        self.entries = timeline['entries']
        self.ends = [r['end_s'] for r in self.entries]
        self.reader = None
        self.path = None
        self.duration = timeline['duration_s']

    def frame(self, when):
        import cv2
        index = min(bisect.bisect_right(self.ends, when), len(self.entries) - 1)
        entry = self.entries[index]
        gap = entry['kind'] == 'gap'
        if gap:
            entry = self.entries[index - 1]
        path = str(MEDIA / entry['video']['path'])
        if path != self.path:
            self.close()
            self.reader = cv2.VideoCapture(path)
            self.path = path
        if gap:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        else:
            self.reader.set(cv2.CAP_PROP_POS_MSEC, max(0, when - entry['start_s']) * 1000)
        ok, frame = self.reader.read()
        if not ok:
            raise RuntimeError(f'Cannot decode stream frame: {path} at {when}')
        return frame

    def close(self):
        if self.reader is not None:
            self.reader.release()
        self.reader = None; self.path = None


def run():
    output = ROOT / 'streams'
    meta = start(output)
    population()
    receipts = {}
    for speaker in sorted(FMH | TEST):
        source = ROOT / f'preflight/timeline_{speaker}.json'
        timeline = read(source)
        receipts[speaker] = render_audio(timeline, output / f'{speaker}.wav')
        receipts[speaker]['timeline'] = spec(source)
        write_json_atomic(output / f'receipt_{speaker}.json', receipts[speaker])
        write_json_atomic(output / 'progress.json', {'completed_speakers': list(receipts)})
    finish(output, meta, {'streams': {s: {'receipt': spec(output / f'receipt_{s}.json'),
                'audio': r['audio'], 'timeline': r['timeline']} for s, r in receipts.items()},
            'gpu_seconds': 0, 'video': 'lazy original pixels; last frame held during synthetic gaps'})


if __name__ == '__main__':
    run()
