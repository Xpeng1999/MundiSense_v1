import numpy as np

from emotioncore.x03.fusion import apply_face_missing, crossfit, fit_neutral_bias, softmax


def toy_data():
    rng = np.random.default_rng(13)
    folds = np.repeat(np.arange(5), 35)
    y = np.tile(np.arange(7), 25)
    p = rng.dirichlet(np.ones(7), size=(175, 2))
    logp = np.log(p)
    return logp, y, folds


def test_heldout_fold_labels_do_not_affect_its_parameters_or_predictions():
    logp, y, folds = toy_data()
    a, n, parameters = crossfit(logp, y, folds)
    changed = y.copy()
    changed[folds == 2] = (changed[folds == 2] + 1) % 7
    a2, n2, parameters2 = crossfit(logp, changed, folds)
    assert parameters[2] == parameters2[2]
    np.testing.assert_array_equal(a[folds == 2], a2[folds == 2])
    np.testing.assert_array_equal(n[folds == 2], n2[folds == 2])
    assert np.isfinite(a).all() and np.isfinite(n).all()
    np.testing.assert_allclose(a.sum(axis=1), 1)


def test_neutral_working_point_uses_only_fit_rows():
    logp, y, folds = toy_data()
    logits = logp[:, 0, :]
    bias = fit_neutral_bias(logits[folds != 0], y[folds != 0])
    changed = y.copy()
    changed[folds == 0] = 0
    assert bias == fit_neutral_bias(logits[folds != 0], changed[folds != 0])
    adjusted = logits[folds != 0].copy()
    adjusted[:, 0] += bias
    neutral = y[folds != 0] == 0
    miss_rate = np.mean(np.argmax(adjusted[neutral], axis=1) != 0)
    assert abs(miss_rate - 0.15) <= 1 / neutral.sum()


def test_face_missing_only_uniforms_visual_expert():
    probabilities = np.zeros((2, 2, 7))
    probabilities[:, :, 0] = 1
    changed = apply_face_missing(probabilities, ["missing", "present"],
                                 ("A3_h2", "V2_h2"), {"missing"})
    np.testing.assert_allclose(changed[0, 1], np.ones(7) / 7)
    np.testing.assert_allclose(changed[0, 0], probabilities[0, 0])
    np.testing.assert_allclose(changed[1], probabilities[1])
