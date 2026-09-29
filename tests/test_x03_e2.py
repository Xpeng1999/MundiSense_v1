import numpy as np
import torch

from emotioncore.x03.e2 import E2Model, predict, sample_keep


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
