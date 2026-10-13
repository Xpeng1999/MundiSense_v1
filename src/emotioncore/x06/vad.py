"""Frozen VAD posteriors and shared registered segmentation parameters."""
import math
import numpy as np
from emotioncore.x06.common import ROOT, read

THRESHOLDS = (.3, .5, .7)
SILENCES = (.2, .4, .8)
STEP = .01


def segments(probability, threshold, min_silence, valid=None):
    valid = np.ones(len(probability), dtype=bool) if valid is None else np.asarray(valid)
    minimum = 30; maximum = 1500; silence = round(min_silence / STEP)
    start = last = None
    out = []
    def emit(end, decision):
        if start is not None and end - start >= minimum:
            out.append({'start_s': start * STEP, 'end_s': end * STEP, 'decision_s': decision * STEP})
    for i, value in enumerate(probability):
        if not valid[i]:
            emit(last + 1 if last is not None else i, i)
            start = last = None
            continue
        if value >= threshold:
            if start is None:
                start = i
            last = i
        if start is not None:
            if i + 1 - start >= maximum:
                emit(i + 1, i + 1)
                start = last = None
            elif i - last >= silence:
                emit(last + 1, i + 1)
                start = last = None
    emit(last + 1 if last is not None else len(probability), len(probability))
    return out


def mask_segments(intervals, n):
    mask = np.zeros(n, dtype=bool)
    for interval in intervals:
        a = max(0, round(interval['start_s'] / STEP))
        b = min(n, round(interval['end_s'] / STEP))
        mask[a:b] = True
    return mask


def intervals_from_mask(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [{'start_s': a * STEP, 'end_s': b * STEP}
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def boundary_counts(predicted, truth, tolerance=.25):
    tp = fp = fn = 0
    for kind in ('start_s', 'end_s'):
        p = sorted(r[kind] for r in predicted)
        t = sorted(r[kind] for r in truth)
        i = j = matched = 0
        while i < len(p) and j < len(t):
            if abs(p[i] - t[j]) <= tolerance:
                matched += 1; i += 1; j += 1
            elif p[i] < t[j]:
                i += 1
            else:
                j += 1
        tp += matched; fp += len(p) - matched; fn += len(t) - matched
    return np.asarray([tp, fp, fn], dtype=np.int64)


def prf(counts):
    tp, fp, fn = map(int, counts)
    return {'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn),
            'f1': 2 * tp / max(1, 2 * tp + fp + fn), 'tp': tp, 'fp': fp, 'fn': fn}


class Posterior:
    def __init__(self, name):
        import torch
        self.torch = torch
        self.name = name
        torch.set_num_threads(1)
        paths = read(ROOT / 'vad_models/summary.json')
        if name == 'silero':
            self.model = torch.jit.load(paths['silero'], map_location='cpu').eval()
        elif name == 'fsmn':
            import yaml
            from pathlib import Path
            from funasr.models.fsmn_vad_streaming.encoder import FSMN
            from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming
            from funasr.frontends.wav_frontend import WavFrontendOnline
            root = Path(paths['fsmn'])
            config = yaml.safe_load((root / 'config.yaml').read_text())
            self.frontend = WavFrontendOnline(cmvn_file=str(root / 'am.mvn'), **config['frontend_conf'])
            self.model = FsmnVADStreaming(encoder=config['encoder'], encoder_conf=config['encoder_conf'],
                                          **config['model_conf']).eval()
            state = torch.load(root / 'model.pt', map_location='cpu', weights_only=True)
            self.model.load_state_dict(state)
        else:
            raise ValueError('Only registered VAD candidates are allowed')

    def score(self, signal, valid=None):
        torch = self.torch
        n = math.ceil(len(signal) / 160)
        valid = np.ones(n, dtype=bool) if valid is None else valid
        result = np.zeros(n, dtype=np.float32)
        for region in intervals_from_mask(valid):
            a, b = round(region['start_s'] * 16000), min(len(signal), round(region['end_s'] * 16000))
            audio = signal[a:b]
            if len(audio) < 400:
                continue
            values = []
            with torch.inference_mode():
                if self.name == 'silero':
                    self.model.reset_states()
                    for offset in range(0, len(audio), 512):
                        chunk = audio[offset:offset + 512]
                        if len(chunk) < 512:
                            chunk = np.pad(chunk, (0, 512 - len(chunk)))
                        values.append(float(self.model(torch.from_numpy(chunk), 16000).item()))
                    positions = np.minimum((np.arange(math.ceil(len(audio) / 160)) * 160 / 512).astype(int), len(values) - 1)
                    scored = np.asarray(values, dtype=np.float32)[positions]
                else:
                    frontend_cache, encoder_cache = {}, {}
                    for offset in range(0, len(audio), 16000):
                        chunk = torch.from_numpy(audio[offset:offset + 16000])[None]
                        feats, lengths = self.frontend(chunk, torch.tensor([chunk.shape[1]]),
                                                       cache=frontend_cache, is_final=offset + 16000 >= len(audio))
                        if feats.ndim == 3 and feats.shape[1]:
                            scores = self.model.encoder(feats, cache=encoder_cache)
                            values.extend((1 - scores[0, :, 0]).numpy().tolist())
                    if not values:
                        continue
                    scored = np.asarray(values, dtype=np.float32)
                    size = math.ceil(len(audio) / 160)
                    scored = np.pad(scored, (0, max(0, size - len(scored))), mode='edge')[:size]
            first = round(region['start_s'] / STEP)
            result[first:first + len(scored)] = scored[:n - first]
        if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
            raise ValueError('STOP invalid VAD posterior')
        return result
