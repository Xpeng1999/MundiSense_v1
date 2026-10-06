from emotioncore.x04.dedup import match, normalize, candidates


def test_normalize_without_extra_casefold():
    assert normalize("ＡＢＣ，汉 字１２!ab") == "ABC汉字12ab"


def test_fixed_short_rule_needs_duration():
    assert match("你好", "你好") is None
    assert match("你好", "你好", 2., 2.3) == 1
    assert match("你好", "你好", 2., 2.301) is None
    assert match("", "", 1., 1.) is None


def test_jaccard_and_exact():
    assert match("今天心情很好", "今天心情很好") == 1
    assert match("今天心情很好", "今天心情很差") is None
    assert match("今天心情很好啊", "今天心情很好吧") >= .7


def test_candidates_emit_ids_not_texts():
    a = [{"utt_id": "mer_1", "normalized": "今天心情很好"}]
    b = [{"utt_id": "sims_2", "normalized": "今天心情很好"}]
    found = candidates(a, b, lambda _: None)
    assert found == [{"source_id": "mer_1", "target_id": "sims_2", "similarity": 1., "short_exact": False}]
