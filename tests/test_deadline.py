import time

import pytest

from emotioncore.eval.run_baselines import sample_deadline


def test_sample_deadline_interrupts_stalled_preprocessing():
    with pytest.raises(TimeoutError, match="exceeded"):
        with sample_deadline(0.1):
            time.sleep(1)
    with sample_deadline(1):
        time.sleep(0.01)
