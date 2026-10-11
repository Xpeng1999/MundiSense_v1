import json
from pathlib import Path
import pytest
from emotioncore.x06.common import assert_tuning_source
from emotioncore.x06.gate import route, upgraded_bytes, verify_only_gate_changed
from emotioncore.x06.simulator import build_timelines


def test_only_training_sources_can_tune():
    assert_tuning_source('fmh', ['06', '10', '11'])
    assert_tuning_source('aishell_train')
    for source, speakers in [('et_test', ['02']), ('fmh', ['06', '02']),
                             ('aishell_test', None), ('sims_valid', None)]:
        with pytest.raises(ValueError):
            assert_tuning_source(source, speakers)


def test_bundle_changes_only_gate_bytes():
    original = Path('configs/calibration_bundle_v1.json').read_bytes()
    revised = upgraded_bytes(original)
    verify_only_gate_changed(original, revised)
    a, b = json.loads(original), json.loads(revised)
    del a['gate_rules'], b['gate_rules']
    assert a == b
    with pytest.raises(ValueError):
        verify_only_gate_changed(original, revised.replace(b'"version": 1', b'"version": 2'))


@pytest.mark.parametrize('face,speech,scores,expected', [
    (True, True, [10], ('full', 'both_available')),
    (True, True, [10, 4.9], ('full', 'both_available')),
    (True, True, [10, 5], ('reject', 'ambiguous_multi_face')),
    (False, True, [], ('H_A', 'no_face')),
    (False, False, [], ('reject', 'no_modalities')),
    (True, False, [10], ('H_V', 'visual_only')),
])
def test_gate_boundary(face, speech, scores, expected):
    assert route(face, speech, scores) == expected


def test_timelines_retain_original_durations_and_sorting():
    records = [{'utt_id': uid, 'speaker_id': '06', 'group': 'G01', 'dialogue': '01',
                'mp4': {'duration_s': duration}, 'wav': {'duration_s': duration - .1},
                'labels': {'M': 'neutral'}}
               for uid, duration in [('G01_01_06_002', 1.234567), ('G01_01_06_001', 2.345678)]]
    timeline = build_timelines(records)['06']
    assert timeline == build_timelines(records)['06']
    utterances = [e for e in timeline['entries'] if e['kind'] == 'utterance']
    assert [e['utt_id'] for e in utterances] == sorted(r['utt_id'] for r in records)
    durations = {r['utt_id']: r['mp4']['duration_s'] for r in records}
    for e in utterances:
        assert e['end_s'] - e['start_s'] == pytest.approx(durations[e['utt_id']], abs=1e-12)
    gap = timeline['entries'][1]
    assert gap['synthetic'] and .3 <= gap['end_s'] - gap['start_s'] <= 1.5


def test_service_decisions_preserve_frozen_bundle_formula():
    from emotioncore.x06.service import Service
    from emotioncore.x05.bundle import Bundle
    service = Service.__new__(Service)
    service.bundle = json.loads(Path('configs/calibration_bundle_v1.1.json').read_text())
    service.previous = {}
    baseline = Bundle.load()
    probabilities = {c: [.1, .3, .05, .2, .05, .1, .2]
                     for p in service.bundle['systems'].values() for c in p['components']}
    for system in ('H3', 'H_ref'):
        old = baseline.apply(probabilities, dict(has_face=True, face_count=1, has_speech=True,
                                                full_system=system, rejection_point='90'))
        new = service.decide(probabilities, system, scores=[1.])
        assert (new['label'], new['route']) == (old[0], old[2])
        assert new['confidence'] == pytest.approx(old[1], abs=1e-12)
    visual = service.decide(probabilities, 'H3', has_speech=False, scores=[1.])
    assert visual['rejection_point'] == '80' and visual['reason'] == 'visual_only'


def test_vad_silence_minimum_maximum_and_overlap_exclusion():
    import numpy as np
    from emotioncore.x06.vad import segments, mask_segments
    p = np.r_[np.zeros(10), np.ones(40), np.zeros(20), np.ones(29), np.zeros(20)]
    result = segments(p, .5, .2)
    assert result == [{'start_s': .1, 'end_s': .5, 'decision_s': .7}]
    forced = segments(np.ones(3100), .5, .2)
    assert [round(x['end_s'] - x['start_s'], 5) for x in forced] == [15, 15, 1]
    valid = np.ones(100, dtype=bool); valid[40:60] = False
    detected = segments(np.ones(100), .5, .2, valid)
    assert not mask_segments(detected, 100)[~valid].any()
