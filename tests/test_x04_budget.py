import numpy as np
from emotioncore.x04.budget import conservative_curve, RATIOS


def test_cost_curve_does_not_discount_slow_probe_and_flags_extrapolation():
    samples = [{"input_token_length": n, "train_step_s": s}
               for n, s in ((100, 2.), (200, 1.), (300, 3.))]
    seconds, receipt = conservative_curve(samples, [50, 100, 200, 250, 300, 600], "train_step_s")
    assert np.allclose(seconds, [2, 2, 2, 2.5, 3, 12])
    assert receipt["extrapolated_count"] == 1
    assert RATIOS == {"AV": .6, "A": .2, "V": .2}
