import hashlib
import json
from pathlib import Path

import pytest

from emotioncore.x02.folds import EXCLUDED_SPEAKERS, validate_oof


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/X01_benchmark/results"


def records(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def test_frozen_folds_have_no_speaker_or_media_leakage():
    folds_file = ROOT / "configs/folds_v1.json"
    frozen = json.loads(folds_file.read_text(encoding="utf-8"))
    expected_sha = (ROOT / "configs/folds_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(folds_file.read_bytes()).hexdigest() == expected_sha
    manifest = RESULTS / "et_utterances.jsonl"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == frozen["source_manifest_sha256"]
    held = set(EXCLUDED_SPEAKERS)
    pool = set(frozen["cv_pool_speakers"])
    assert len(pool) == 15 and not pool.intersection(held)
    outer_groups = [set(f["outer_holdout_speakers"]) for f in frozen["folds"]]
    assert len(outer_groups) == 5
    assert all(len(group) == 3 for group in outer_groups)
    assert set.union(*outer_groups) == pool
    assert sum(map(len, outer_groups)) == len(pool)
    for fold, outer in zip(frozen["folds"], outer_groups):
        inner = set(fold["inner_validation_speakers"])
        train = set(fold["inner_train_speakers"])
        assert len(inner) == 3 and len(train) == 9
        assert not outer.intersection(inner | train)
        assert not inner.intersection(train)
        assert outer | inner | train == pool

    pool_ids = set()
    test_ids = set()
    et_hashes = set()
    for row in records(manifest):
        if row["speaker_id"] in pool:
            pool_ids.add(row["utt_id"])
        if row["official_split"] == "test":
            assert row["speaker_id"] in held
            test_ids.add(row["utt_id"])
        for modality in ("mp4", "wav"):
            if row.get(modality, {}).get("sha256"):
                et_hashes.add(row[modality]["sha256"])
    assert len(pool_ids) == frozen["cv_pool_count"] == 16366
    assert len(test_ids) == frozen["et_test_count"] == 1929
    assert not pool_ids.intersection(test_ids)

    # CH-SIMS supplies no speaker IDs. Cross-corpus checks use IDs and media hashes.
    lockbox = list(records(RESULTS / "sims_test_lockbox.jsonl"))
    lockbox_ids = {f"{row['video_id']}__{row['clip_id']}" for row in lockbox}
    lockbox_hashes = {row["media"]["sha256"] for row in lockbox
                      if row["media"].get("sha256")}
    assert not lockbox_ids.intersection(pool_ids | test_ids)
    assert not lockbox_hashes.intersection(et_hashes)


def test_oof_validator_rejects_missing_or_duplicate_rows():
    expected = {"u1", "u2", "u3"}
    validate_oof(expected, [{"utt_id": id_} for id_ in ("u1", "u2", "u3")])
    with pytest.raises(ValueError):
        validate_oof(expected, [{"utt_id": id_} for id_ in ("u1", "u2", "u2")])
    with pytest.raises(ValueError):
        validate_oof(expected, [{"utt_id": id_} for id_ in ("u1", "u2")])
