"""Duration-weighted truth evaluation, retaining uncovered utterances as rejects."""
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x06.simulator import overlap_seconds


def overlaps(truth, segments):
    return [(i, overlap_seconds(truth['start_s'], truth['end_s'], s['start_s'], s['end_s']))
            for i, s in enumerate(segments)
            if s['start_s'] < truth['end_s'] and s['end_s'] > truth['start_s']]


def components_for_truth(truth, segments):
    matched = overlaps(truth, segments)
    if not matched:
        return None
    keys = set(segments[matched[0][0]]['component_probs'])
    if any(set(segments[i]['component_probs']) != keys for i, _ in matched):
        raise ValueError('Cannot aggregate different component definitions')
    total = sum(weight for _, weight in matched)
    return {key: (sum(weight * np.asarray(segments[i]['component_probs'][key])
                     for i, weight in matched) / total).tolist() for key in sorted(keys)}


def truth_decision(truth, segments, decide):
    probabilities = components_for_truth(truth, segments)
    if probabilities is None:
        return {'label': 'reject', 'confidence': 0., 'reason': 'uncovered_by_vad'}
    return decide(probabilities)


def metrics(truth, predicted):
    """Seven-class macro-F1; reject contributes FN and remains in the denominator."""
    if len(truth) != len(predicted):
        raise ValueError('Truth and prediction populations differ')
    details = {}
    for label in LABELS:
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        details[label] = 2 * tp / max(1, 2 * tp + fp + fn)
    accepted = sum(p != 'reject' for p in predicted)
    neutral = sum(t == 'neutral' for t in truth)
    return {'n': len(truth), 'macro_f1': float(np.mean(list(details.values()))),
            'class_f1': details, 'accepted': accepted, 'coverage': accepted / max(1, len(truth)),
            'neutral_false_positive_rate': sum(t == 'neutral' and p not in ('neutral', 'reject')
                for t, p in zip(truth, predicted)) / max(1, neutral)}
