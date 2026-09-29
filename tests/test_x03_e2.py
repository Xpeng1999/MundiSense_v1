import numpy as np
import torch

from emotioncore.x03.e2 import E2Model, predict, sample_keep
from emotioncore.x03.e2_metrics import calibrated_oof, fit_neutral_bias_with_invalid, score


def test_modality_dropout_keeps_one_available_modality():
    available = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0]],
                             dtype=torch.bool)
    keep = sample_keep(available, probability=1.0,
                       generator=torch.Generator().manual_seed(20260929))
    assert keep.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_visual_only_missing_face_is_uniform_and_kept_in_output():
    rng = np.random.default_rng(3)
    data = {"audio": rng.normal(size=(2, 4, 3, 1024)).astype(np.float16),
            "audio_lengths": np.array([3, 3]),
            "visual": rng.normal(size=(2, 2, 768)).astype(np.float16),
            "visual_lengths": np.array([2, 0]),
            "text": rng.normal(size=(2, 1024)).astype(np.float16),
            "text_available": np.array([True, True])}
    model = E2Model.build(0.2)
    probabilities = predict(model, data, np.array([0, 1]), "V", "cpu", 2)
    assert probabilities.shape == (2, 7)
    assert np.allclose(probabilities[1], 1 / 7, atol=0)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)


def test_uniform_missing_visual_counts_as_error_even_for_neutral():
    rows = [{"utt_id": "missing", "speaker_id": "01", "truth_M": "neutral",
             "status": "face_missing"}]
    result = score(rows, np.full((1, 7), 1 / 7), "V")
    assert result["n"] == 1
    assert result["failures"] == 1
    assert result["wa"] == 0
    assert result["face_missing_forced_error"] == 1


def test_e2_fold_calibration_never_reads_held_fold_labels():
    rng = np.random.default_rng(5)
    from emotioncore.eval.metrics import LABELS

    rows = []
    for fold in range(5):
        for repeat in range(2):
            for label in LABELS:
                rows.append({"fold": fold, "truth_M": label,
                             "status": "ok",
                             "probabilities": rng.dirichlet(np.ones(7)).tolist()})
    a, n, parameters = calibrated_oof(rows, "AVT")
    changed = [dict(row) for row in rows]
    for row in changed:
        if row["fold"] == 0:
            row["truth_M"] = LABELS[(LABELS.index(row["truth_M"]) + 1) % 7]
    a_changed, n_changed, parameters_changed = calibrated_oof(changed, "AVT")
    assert parameters[0] == parameters_changed[0]
    assert np.array_equal(a[:14], a_changed[:14])
    assert np.array_equal(n[:14], n_changed[:14])


def test_neutral_workpoint_retains_invalid_neutral_in_denominator():
    logits = np.zeros((4, 7))
    logits[1:, 1] = [0.1, 0.2, 0.3]
    invalid = np.array([True, False, False, False])
    bias = fit_neutral_bias_with_invalid(logits, np.zeros(4, dtype=int),
                                         invalid, target=0.25)
    assert np.isclose(bias, 0.2)
    predicted = np.argmax(logits + np.array([bias, 0, 0, 0, 0, 0, 0]), axis=1)
    assert int(np.count_nonzero((predicted[~invalid] != 0))) == 1


def test_calibrated_missing_visual_stays_uniform():
    from emotioncore.eval.metrics import LABELS

    rng = np.random.default_rng(17)
    rows = [{"fold": fold, "truth_M": label,
             "status": "face_missing" if fold == 0 and label == "neutral" else "ok",
             "probabilities": (np.full(7, 1 / 7) if fold == 0 and label == "neutral"
                               else rng.dirichlet(np.ones(7))).tolist()}
            for fold in range(5) for label in LABELS]
    op_a, op_n, _ = calibrated_oof(rows, "V")
    assert np.array_equal(op_a[0], np.full(7, 1 / 7))
    assert np.array_equal(op_n[0], np.full(7, 1 / 7))
