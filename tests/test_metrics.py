from emotioncore.eval.metrics import classification, sims_metrics


def test_invalid_predictions_stay_in_denominator():
    rows = [{"truth": "neutral", "pred": "joy"},
            {"truth": "neutral", "pred": None},
            {"truth": "joy", "pred": "joy"}]
    result = classification(rows)
    assert result["n"] == 3
    assert result["valid"] == 2
    assert result["wa"] == 1 / 3
    assert result["neutral_false_positive_rate"] == 1 / 2
    assert result["invalid_by_true"]["neutral"] == 1


def test_sims_excludes_zero_label_and_counts_failure_as_error():
    rows = [{"label": 1.0, "s": 0.2},
            {"label": -1.0, "s": -0.4},
            {"label": 0.0, "s": 0.9},
            {"label": 1.0, "s": None}]
    result = sims_metrics(rows)
    assert result["nonzero_denominator"] == 3
    assert result["nonzero_correct"] == 2
    assert result["acc_2_nonzero"] == 2 / 3
    assert result["correlation_denominator"] == 3
