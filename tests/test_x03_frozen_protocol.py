import hashlib
import json
from pathlib import Path

from emotioncore.eval.metrics import LABELS


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "configs" / name
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    return json.loads(path.read_text())


def test_inner_validation_subsets_are_frozen_and_speaker_disjoint():
    subset = load("x03_inner_val_subsets.json")
    folds = load("folds_v1.json") if (ROOT / "configs/folds_v1.json.sha256").exists() else json.loads(
        (ROOT / "configs/folds_v1.json").read_text())
    rows = [json.loads(line) for line in
            (ROOT / "experiments/X01_benchmark/results/et_utterances.jsonl").open()]
    by_id = {row["utt_id"]: row for row in rows}
    assert subset["seed"] == 20260929
    assert len(subset["folds"]) == 5
    for item, frozen in zip(subset["folds"], folds["folds"]):
        ids = item["utt_ids"]
        assert item["fold"] == frozen["fold"]
        assert len(ids) == len(set(ids)) == 700
        assert set(by_id[uid]["speaker_id"] for uid in ids) <= set(frozen["inner_validation_speakers"])
        assert not (set(frozen["inner_validation_speakers"]) & set(frozen["outer_holdout_speakers"]))
        assert {label: sum(by_id[uid]["labels"]["M"] == label for uid in ids)
                for label in LABELS} == item["label_counts"]


def test_face_missing_policy_is_frozen_and_complete():
    policy = load("x03_face_missing_policy.json")
    ids = policy["face_missing_utt_ids"]
    assert len(ids) == len(set(ids)) == 43
    assert policy["training"]["assigned_V"] == "A"
    assert policy["training"]["assigned_AV"] == "A"
    assert policy["evaluation"]["V"] == "uniform_7_class_counted_in_denominator"
    assert policy["evaluation"]["AV"] == "audio_only_forward_mark_face_missing_fallback"
