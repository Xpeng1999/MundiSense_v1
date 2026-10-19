"""Fixed X06a track-score gate; preserve every other byte of bundle v1."""
import json
from pathlib import Path

RULES = {
    'multiple_faces': 'reject:ambiguous_multi_face if second_score >= 0.5 * target_score',
    'second_to_target_score_threshold': 0.5,
    'neither_modality': 'reject:no_modalities',
    'no_face': 'H_A',
    'no_speech': 'H_V',
    'both': 'requested full system',
    'face_quality_threshold': None,
    'default_full_system': 'H3',
    'visual_only_rejection_point': '80',
}


def gate_value_span(raw):
    """Locate the JSON value rather than re-encoding the surrounding document."""
    decoder = json.JSONDecoder()
    i = raw.index('{') + 1
    while True:
        while raw[i].isspace() or raw[i] == ',':
            i += 1
        if raw[i] == '}':
            raise ValueError('gate_rules missing')
        key, i = decoder.raw_decode(raw, i)
        while raw[i].isspace() or raw[i] == ':':
            i += 1
        _, end = decoder.raw_decode(raw, i)
        if key == 'gate_rules':
            return i, end
        i = end


def upgraded_bytes(raw):
    text = raw.decode('utf-8')
    start, end = gate_value_span(text)
    replacement = json.dumps(RULES, ensure_ascii=False, indent=2).replace('\n', '\n  ')
    return (text[:start] + replacement + text[end:]).encode('utf-8')


def verify_only_gate_changed(original, upgraded):
    left, right = original.decode(), upgraded.decode()
    a, b = gate_value_span(left)
    c, d = gate_value_span(right)
    if left[:a].encode() != right[:c].encode() or left[b:].encode() != right[d:].encode():
        raise ValueError('STOP bytes outside gate_rules changed')
    if json.loads(right[c:d]) != RULES:
        raise ValueError('STOP unregistered gate rules')


def route(has_face, has_speech, scores):
    scores = sorted(scores, reverse=True)
    if len(scores) > 1 and scores[0] > 0 and scores[1] >= .5 * scores[0]:
        return 'reject', 'ambiguous_multi_face'
    if not has_face and not has_speech:
        return 'reject', 'no_modalities'
    if not has_face:
        return 'H_A', 'no_face'
    if not has_speech:
        return 'H_V', 'visual_only'
    return 'full', 'both_available'


if __name__ == '__main__':
    source = Path('configs/calibration_bundle_v1.json').read_bytes()
    target = upgraded_bytes(source)
    verify_only_gate_changed(source, target)
    Path('configs/calibration_bundle_v1.1.json').write_bytes(target)
