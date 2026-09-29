from emotioncore.x02.asr_cer import edit_distance, normalize


def test_cer_edit_distance_covers_insert_delete_substitute_and_empty():
    assert edit_distance("", "") == 0
    assert edit_distance("情绪", "情绪啊") == 1
    assert edit_distance("情绪啊", "情绪") == 1
    assert edit_distance("开心", "伤心") == 1
    assert edit_distance("ABC", "XYZ") == 3


def test_cer_normalization_uses_nfc_and_outer_trim_only():
    assert normalize("  e\u0301 好  ") == "é 好"
