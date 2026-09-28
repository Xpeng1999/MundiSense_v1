import pytest

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.paired_oof import compare


def rows():
    result = []
    for i in range(15):
        probabilities = [0.0] * 7
        probabilities[i % 7] = 1.0
        result.append({"utt_id": f"u{i}", "speaker_id": f"{i:02d}",
                       "truth_M": LABELS[i % 7], "probabilities": probabilities,
                       "status": "ok"})
    return result


def test_identical_oof_has_zero_paired_speaker_interval():
    data = rows()
    result = compare(data, data, repetitions=40)
    assert result["n"] == 15
    assert result["left_minus_right"]["macro_f1"]["difference"] == 0
    assert result["left_minus_right"]["macro_f1"]["lower_95"] == 0
    assert result["left_minus_right"]["macro_f1"]["upper_95"] == 0


def test_paired_oof_rejects_id_mismatch():
    data = rows()
    changed = [*data]
    changed[0] = {**changed[0], "utt_id": "other"}
    with pytest.raises(ValueError, match="same unique ordered IDs"):
        compare(data, changed, repetitions=1)
