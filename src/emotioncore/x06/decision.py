"""Frozen calibration, optional EMA, then unchanged neutral bias/rejection."""
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x05.calibration import base_scores, predictions
from emotioncore.x06.gate import route


def from_probability(p, params, speech=True, ambiguous=False, routing='full', reason='both_available'):
    decision = np.log(np.clip(p, 1e-12, 1))
    decision[0] += params['neutral_bias']
    label = int(decision.argmax())
    confidence = float(p[label])
    point = '90' if speech else '80'
    accepted = confidence >= params['thresholds'][point] and not ambiguous
    return {'label': LABELS[label] if accepted else 'reject', 'unrejected_label': LABELS[label],
            'confidence': confidence, 'route': 'reject' if ambiguous else routing,
            'reason': 'ambiguous_multi_face' if ambiguous else reason,
            'rejection_point': point, 'calibrated_probabilities': np.asarray(p).tolist()}


def apply(bundle, probabilities, system, *, has_face=True, speech=True, scores=(), alpha=0., state=None, stream_id=''):
    routing, reason = route(has_face, speech, scores)
    if reason == 'no_modalities':
        return {'label': 'reject', 'confidence': 0., 'route': routing, 'reason': reason}
    # A gate rejection still advances the calibrated speech sequence; the gate
    # is applied after EMA and never changes any fitted parameter.
    name = system if has_face and speech else 'H_A' if speech else 'H_V'
    params = bundle['systems'][name]
    logp = np.log(np.clip(np.asarray([probabilities[c] for c in params['components']])[None], 1e-12, 1))
    p = predictions(base_scores(logp, params), params['neutral_bias'])[0][0]
    if speech and state is not None:
        key = (stream_id, system)
        if key in state:
            p = (1 - alpha) * p + alpha * state[key]
        state[key] = p.copy()
    return from_probability(p, params, speech, reason == 'ambiguous_multi_face', name, reason)
