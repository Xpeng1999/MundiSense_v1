from emotioncore.x02.quality_addendum import measure, normalize_nfkc_alnum


def test_nfkc_lowercase_and_remove_punctuation_without_removing_hanzi():
    assert normalize_nfkc_alnum("Ａ，Ｂ。１２　中文!") == "ab12中文"


def test_truncation_rule_uses_strict_fixed_thresholds():
    assert not measure("123456789", "123456789", 1.0)["truncation_suspect"]
    assert measure("1234567890", "1234567890", 1.0)["truncation_suspect"]
    assert not measure("1234567890", "12345", 2.0)["truncation_suspect"]
    assert measure("1234567890", "1234", 2.0)["truncation_suspect"]


def test_new_cer_ignores_punctuation_and_old_cer_keeps_it():
    result = measure("你好，WORLD!", "你好world", 2.0)
    assert result["cer_nfkc_alnum"] == 0
    assert result["old_edit_distance"] > 0
