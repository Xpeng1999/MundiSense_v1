import numpy as np

from emotioncore.x03.q_calibrate import apply_single, crossfit_single, fit_neutral_bias_with_invalid


def test_q_crossfit_does_not_use_held_out_labels():
    rng = np.random.default_rng(20260929)
    folds = np.repeat(np.arange(5), 14)
    y = np.tile(np.repeat(np.arange(7), 2), 5)
    logp = np.log(rng.dirichlet(np.ones(7), size=len(y)))
    invalid = np.zeros(len(y), dtype=bool)
    invalid[0] = True
    first_a, first_n, first_params = crossfit_single(logp, y, folds, invalid)
    changed = y.copy()
    changed[folds == 0] = (changed[folds == 0] + 1) % 7
    second_a, second_n, second_params = crossfit_single(logp, changed, folds, invalid)
    np.testing.assert_allclose(first_a[folds == 0], second_a[folds == 0])
    np.testing.assert_allclose(first_n[folds == 0], second_n[folds == 0])
    assert first_params[0] == second_params[0]
    assert np.all(first_n[invalid] == 1 / 7)


def test_q_missing_face_remains_uniform_and_counts_in_neutral_denominator():
    logits = np.array([[0.0, 1.0, 0, 0, 0, 0, 0],
                       [0.0, 2.0, 0, 0, 0, 0, 0],
                       [0.0, 3.0, 0, 0, 0, 0, 0]])
    y = np.zeros(3, dtype=int)
    invalid = np.array([False, False, True])
    bias = fit_neutral_bias_with_invalid(logits, y, invalid, target=1 / 3)
    assert np.isclose(bias, 1.0) or np.isclose(bias, 2.0)
    a, n = apply_single(np.log(np.full((3, 7), 1 / 7)), invalid,
                        np.zeros(8), bias)
    np.testing.assert_allclose(a[invalid], np.full((1, 7), 1 / 7))
    np.testing.assert_allclose(n[invalid], np.full((1, 7), 1 / 7))
