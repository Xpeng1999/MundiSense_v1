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
    assert len(result) == 1
    assert result[0] == pytest.approx({'start_s': .1, 'end_s': .5, 'decision_s': .7}, abs=1e-12)
    forced = segments(np.ones(3100), .5, .2)
    assert [round(x['end_s'] - x['start_s'], 5) for x in forced] == [15, 15, 1]
    valid = np.ones(100, dtype=bool); valid[40:60] = False
    detected = segments(np.ones(100), .5, .2, valid)
    assert not mask_segments(detected, 100)[~valid].any()


def test_uncovered_truth_is_reject_and_counts_as_false_negative():
    from emotioncore.x06.aggregate import truth_decision, metrics, components_for_truth
    truth = {'start_s': 2., 'end_s': 4.}
    actual = truth_decision(truth, [], lambda _: pytest.fail('No covered component exists'))
    assert actual['label'] == 'reject'
    assert metrics(['neutral'], [actual['label']])['n'] == 1
    assert metrics(['neutral'], [actual['label']])['class_f1']['neutral'] == 0
    segments = [{'start_s': 1., 'end_s': 3., 'component_probs': {'A': [1., 0.]}},
                {'start_s': 3., 'end_s': 5., 'component_probs': {'A': [0., 1.]}}]
    assert components_for_truth(truth, segments)['A'] == [.5, .5]


def test_inference_input_allowlist_excludes_sealed_or_unregistered_media(monkeypatch):
    from emotioncore.x06 import streams
    monkeypatch.setattr(streams, 'population', lambda: [])
    for path in ('/data/CH-SIMS/test/sample.mp4', '/data/MER/test/sample.mp4',
                 '/tmp/unregistered.mp4'):
        with pytest.raises(ValueError, match='outside the X06'):
            streams.validate_input(path)


def test_online_endpoints_match_fullspeed_and_silence_is_disjoint():
    import numpy as np
    from emotioncore.x06.online_vad import Endpoints, LiveEvents, silence_windows
    from emotioncore.x06.vad import segments
    p = np.r_[np.zeros(250), np.ones(45), np.zeros(320), np.ones(1700), np.zeros(21)]
    online = Endpoints(.5, .2)
    events = []
    for value in p:
        event = online.push(value)
        if event:
            events.append(event)
    if online.finish():
        events.append(online.finish())
    assert events == segments(p, .5, .2)
    silent = silence_windows(events, len(p) / 100)
    for window in silent:
        assert window['end_s'] - window['start_s'] == pytest.approx(2.)
        assert all(window['end_s'] <= v['start_s'] or window['start_s'] >= v['end_s'] for v in events)
    causal = LiveEvents(.5, .2)
    emitted = [event for value in p for event in causal.advance(value)] + causal.finish(len(p) / 100)
    assert [{k: e[k] for k in ('start_s', 'end_s')} for e in emitted if e['visual_only']] == [
        {k: e[k] for k in ('start_s', 'end_s')} for e in silent]


def test_streaming_ema_zero_matches_frozen_service_and_gates_update_state():
    from emotioncore.x06.decision import apply
    from emotioncore.x06.service import Service
    service = Service.__new__(Service)
    service.bundle = json.loads(Path('configs/calibration_bundle_v1.1.json').read_text())
    service.previous = {}
    probabilities = {c: [.1, .3, .05, .2, .05, .1, .2]
                     for p in service.bundle['systems'].values() for c in p['components']}
    for name in ('H3', 'H_ref'):
        for face, speech, scores in [(True, True, [1]), (True, True, [1, .5]),
                                     (False, True, []), (True, False, [1]), (False, False, [])]:
            a = service.decide(probabilities, name, has_face=face, has_speech=speech, scores=scores)
            state = {}
            b = apply(service.bundle, probabilities, name, has_face=face, speech=speech, scores=scores, state=state)
            assert (a['label'], a['route'], a['reason']) == (b['label'], b['route'], b['reason'])
            if a['route'] != 'reject':
                assert a['confidence'] == pytest.approx(b['confidence'], abs=1e-12)
            if speech:
                assert ('', name) in state


def test_ema_selection_cannot_access_et_labels(monkeypatch):
    from emotioncore.x06 import analysis
    from emotioncore.x06.common import FMH
    bundle = json.loads(Path('configs/calibration_bundle_v1.1.json').read_text())
    components = {c: [.1, .3, .05, .2, .05, .1, .2] for p in bundle['systems'].values() for c in p['components']}
    seen = []
    def label_reader(speaker):
        assert speaker in FMH, 'ET-test labels accessed during tuning'
        seen.append(speaker)
        return [{'utt_id': speaker + '_truth', 'start_s': 0., 'end_s': 1., 'label': 'neutral'}]
    monkeypatch.setattr(analysis, 'truths', label_reader)
    records = [{'cohort': 'fmh', 'speaker_id': speaker, 'start_s': 0., 'end_s': 1.,
                'component_probs': components, 'has_face': True, 'track_scores': [1.]} for speaker in FMH]
    alpha, _ = analysis.choose_alpha(bundle, records)
    assert alpha == 0. and set(seen) == set(FMH)
    with pytest.raises(ValueError):
        analysis.choose_alpha(bundle, [*records, dict(records[0], cohort='et_test', speaker_id='02')])


def test_video_sampling_excludes_exact_end_even_with_float_noise():
    from emotioncore.x06.prepare_segments import frame_index
    assert frame_index(21.6) == frame_index(2160 * .01) == 108
    assert frame_index(21.61) == 109
