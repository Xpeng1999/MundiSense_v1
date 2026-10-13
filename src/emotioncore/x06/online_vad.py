"""Causal chunks and incremental endpoints for the registered frozen VADs."""
import math
import time
import numpy as np
from emotioncore.x06.vad import STEP


def posterior_chunks(model, signal, wait_until=None):
    """Yield only computed posteriors; availability includes frontend lookahead."""
    torch = model.torch
    total = math.ceil(len(signal) / 160)
    emitted = 0
    if model.name == 'silero':
        model.model.reset_states()
        for offset in range(0, len(signal), 512):
            if wait_until:
                wait_until(min(len(signal) / 16000, (offset + 512) / 16000))
            before = time.monotonic()
            chunk = signal[offset:offset + 512]
            last = offset + 512 >= len(signal)
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))
            with torch.inference_mode():
                p = float(model.model(torch.from_numpy(chunk), 16000).item())
            end = min(total, math.ceil((offset + 512) / 160))
            values = np.full(end - emitted, p, dtype=np.float32)
            yield {'available_s': min(len(signal) / 16000, (offset + 512) / 16000),
                   'first_frame': emitted, 'probabilities': values,
                   'vad_seconds': time.monotonic() - before, 'last': last}
            emitted = end
    else:
        frontend_cache, encoder_cache = {}, {}
        for offset in range(0, len(signal), 16000):
            if wait_until:
                wait_until(min(len(signal) / 16000, (offset + 16000) / 16000))
            before = time.monotonic()
            chunk = torch.from_numpy(signal[offset:offset + 16000])[None]
            last = offset + 16000 >= len(signal)
            with torch.inference_mode():
                feats, lengths = model.frontend(chunk, torch.tensor([chunk.shape[1]]),
                    cache=frontend_cache, is_final=last)
                values = np.zeros(0, dtype=np.float32)
                if feats.ndim == 3 and feats.shape[1]:
                    scores = model.model.encoder(feats, cache=encoder_cache)
                    values = (1 - scores[0, :, 0]).numpy()
                if last and emitted + len(values) < total:
                    values = np.pad(values, (0, total - emitted - len(values)), mode='edge')
                values = values[:total - emitted]
            yield {'available_s': min(len(signal) / 16000, (offset + 16000) / 16000),
                   'first_frame': emitted, 'probabilities': values,
                   'vad_seconds': time.monotonic() - before, 'last': last}
            emitted += len(values)
    if emitted != total:
        raise ValueError('VAD stream did not cover every 10ms frame')


class Endpoints:
    def __init__(self, threshold, min_silence):
        self.threshold = threshold
        self.silence = round(min_silence / STEP)
        self.start = self.last = None
        self.index = 0

    def emit(self, end, decision):
        if self.start is not None and end - self.start >= 30:
            return {'start_s': self.start * STEP, 'end_s': end * STEP, 'decision_s': decision * STEP}

    def push(self, probability):
        i = self.index
        self.index += 1
        if probability >= self.threshold:
            if self.start is None:
                self.start = i
            self.last = i
        event = None
        if self.start is not None:
            if i + 1 - self.start >= 1500:
                event = self.emit(i + 1, i + 1)
                self.start = self.last = None
            elif i - self.last >= self.silence:
                event = self.emit(self.last + 1, i + 1)
                self.start = self.last = None
        return event

    def finish(self):
        return self.emit(self.last + 1 if self.last is not None else self.index, self.index)


class LiveEvents:
    def __init__(self, threshold, min_silence):
        self.voice = Endpoints(threshold, min_silence)
        self.next_silence = 2.

    def advance(self, p):
        result = []
        voice = self.voice.push(p)
        if voice:
            result.append(dict(voice, source='speech_segment', visual_only=False))
            self.next_silence = voice['end_s'] + 2.
        known_silent_until = (self.voice.start if self.voice.start is not None else self.voice.index) * STEP
        while self.next_silence <= known_silent_until + 1e-9:
            result.append({'start_s': self.next_silence - 2., 'end_s': self.next_silence,
                           'decision_s': self.next_silence, 'source': 'silence_window', 'visual_only': True})
            self.next_silence += 2.
        return result

    def finish(self, duration):
        voice = self.voice.finish()
        result = []
        if voice:
            result.append(dict(voice, source='speech_segment', visual_only=False))
            self.next_silence = voice['end_s'] + 2.
        while self.next_silence <= duration + 1e-9:
            result.append({'start_s': self.next_silence - 2., 'end_s': self.next_silence,
                           'decision_s': self.next_silence, 'source': 'silence_window', 'visual_only': True})
            self.next_silence += 2.
        return result


def silence_windows(voice, duration):
    """Disjoint two-second windows wholly in VAD silence, after two silent seconds."""
    previous = 0.
    result = []
    for segment in [*voice, {'start_s': duration, 'end_s': duration}]:
        end = previous + 2.
        while end <= segment['start_s'] + 1e-9:
            result.append({'start_s': end - 2., 'end_s': end, 'decision_s': end,
                           'source': 'silence_window', 'visual_only': True})
            end += 2.
        previous = segment['end_s']
    return result
