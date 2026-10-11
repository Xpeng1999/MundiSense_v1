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
