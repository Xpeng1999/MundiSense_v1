import json

from emotioncore.x03.final_analysis import prediction_rows, score


def test_visual_reference_missing_face_is_wrong_with_full_denominator(tmp_path):
    path = tmp_path / "oof.jsonl"
    rows = [
        {"id": "a", "speaker_id": "s1", "fold": 0, "truth_M": "neutral",
         "status": "face_missing", "op_n": [1 / 7] * 7},
        {"id": "b", "speaker_id": "s2", "fold": 1, "truth_M": "joy",
         "status": "ok", "op_n": [0, 1, 0, 0, 0, 0, 0]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scored = prediction_rows(path, {"a", "b"}, "op_n", force_face_error=True)
    assert scored[0]["pred"] is None
    result = score(scored)
    assert result["n"] == 2
    assert result["failures"] == 1
    assert result["wa"] == .5
    assert result["per_class"]["neutral"]["recall"] == 0
