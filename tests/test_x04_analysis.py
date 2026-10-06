import numpy as np
import pytest
from emotioncore.x03 import fusion
from emotioncore.x04.analysis import extend_mer, native_metrics, paired_speakers


def test_eighth_class_extension_preserves_frozen_seven_class_calibration():
    rng = np.random.default_rng(20261006)
    raw = rng.dirichlet(np.ones(8), size=(11, 3))
    parameters = {"fusion": [.2, .5, .8, .1, -.2, .3, .5, -.1, .2, -.3],
                  "calibration": [.4, .1, -.2, .3, .2, -.4, .1, .4], "neutral_bias": .7}
    a, n = extend_mer(raw, parameters)
    p7 = raw[..., :7]/raw[..., :7].sum(2, keepdims=True)
    logits = fusion.calibrated_logits(fusion.fusion_logits(np.log(p7), np.array(parameters["fusion"])), np.array(parameters["calibration"]))
    expected_a = fusion.softmax(logits)
    logits[:, 0] += parameters["neutral_bias"]
    expected_n = fusion.softmax(logits)
    assert np.allclose(a[:, :7]/a[:, :7].sum(1, keepdims=True), expected_a, atol=1e-12)
    assert np.allclose(n[:, :7]/n[:, :7].sum(1, keepdims=True), expected_n, atol=1e-12)
    a, n = extend_mer(raw, parameters, unsupported_worried=True)
    assert (a[:, 7] == 0).all() and (n[:, 7] == 0).all()


def test_native_uniform_failures_stay_in_denominator_and_shared_truth_removed():
    records = [{"truth_M": k} for k in ("neutral", "worried", "joy")]
    p = np.full((3, 8), 1/8)
    invalid = np.array([True, False, False])
    six = native_metrics(p, records, invalid)
    five = native_metrics(p, records, invalid, shared=True)
    assert six["n"] == 3 and six["failures"] == 1
    assert six["confusion_with_invalid_column"][0][-1] == 1
    assert five["n"] == 2 and "worried" not in five["classes"]


def test_paired_bootstrap_uses_identical_speaker_draws_and_requires_alignment():
    rows = [{"id": str(i), "speaker_id": str(i//7), "truth": k, "pred": k}
            for i, k in enumerate(tuple(("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"))*3)]
    result = paired_speakers(rows, rows)
    assert result["seed"] == 20261006 and result["replicates"] == 2000
    assert result["left_minus_right_95"] == [0., 0.]
    with pytest.raises(ValueError): paired_speakers(rows, list(reversed(rows)))
