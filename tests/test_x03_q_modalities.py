import pytest

from emotioncore.x03.q_smoke import assert_modality_receipt


@pytest.mark.parametrize("condition,keys", [
    ("A", {"input_features"}),
    ("V", {"pixel_values_videos"}),
    ("AV", {"input_features", "pixel_values_videos"}),
])
def test_expected_modalities(condition, keys):
    assert_modality_receipt(condition, keys)


@pytest.mark.parametrize("condition,keys", [
    ("A", {"input_features", "pixel_values_videos"}),
    ("V", {"input_features", "pixel_values_videos"}),
    ("AV", {"input_features"}),
])
def test_forbidden_or_missing_modalities(condition, keys):
    with pytest.raises(RuntimeError):
        assert_modality_receipt(condition, keys)
