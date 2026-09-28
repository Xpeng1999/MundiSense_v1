import json
from pathlib import Path

import pytest

from emotioncore.tools.leakage_check import check


def test_four_overlap_levels(tmp_path):
    media = tmp_path / "G00002_01_07_001.wav"
    media.write_bytes(b"test audio")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"audios": [str(media)]}) + "\n", encoding="utf-8")
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(json.dumps([{"id": "et_G00002_01_07_001", "source": str(media)},
                                    {"id": "et_G00003_01_07_002"}]), encoding="utf-8")
    result = check([train], eval_file)
    assert result["overlap_counts"] == {"utterance": 1, "media_sha256": 1,
                                         "speaker": 2, "dialogue": 1}


def test_legacy_chinese_overlap():
    root = Path("/home/administrator/projects/emotion")
    train = [root / "data/manifests_t0/swift/train_e3_core_av.jsonl",
             root / "data/manifests_t0/swift/train_e3_core_audio.jsonl"]
    selected = root / "av_attribution_v16/artifacts/selected_chinese.json"
    if not all(path.is_file() for path in [*train, selected]):
        pytest.skip("Read-only legacy server files are unavailable locally")
    result = check(train, selected, hash_media=False)
    assert result["overlap_counts"]["utterance"] == 161
    assert result["overlap_counts"]["speaker"] == 168
