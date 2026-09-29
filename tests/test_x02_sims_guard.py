import pytest

from emotioncore.x02.sims_prepare import validate_adapter


def test_sims_valid_adapter_rejects_lockbox_rows():
    rows = [{"utt_id": f"v{i}", "official_split": "sims_valid",
             "speaker_id": "sims_valid", "dataset": "CH-SIMS-v2"}
            for i in range(647)]
    validate_adapter(rows)
    rows[0] = {**rows[0], "official_split": "test"}
    with pytest.raises(ValueError, match="valid-only"):
        validate_adapter(rows)
