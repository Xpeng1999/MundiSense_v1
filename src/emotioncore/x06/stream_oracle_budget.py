"""Charge supplementary true-stream oracle checks before further inference."""
from emotioncore.x06.common import *


def run():
    output = ROOT / 'stream_oracle_budget'
    meta = start(output)
    old = read(ROOT / 'budget_recheck/summary.json')
    # Reserve fresh correct-boundary inference on both ET-test and FMH if the
    # actual-stream ET-test equivalence stop gate permits continuation.
    extra = 5195 * old['both_system_seconds'] * 1.2 + 60
    total = old['estimated_total_gpu_hours'] * 3600 + extra
    passed = total <= MAX_GPU_SECONDS
    finish(output, meta, {'status': 'passed' if passed else 'STOP_budget_estimate',
        'estimated_total_gpu_hours': total / 3600, 'additional_gpu_hours': extra / 3600,
        'previous_estimate': spec(ROOT / 'budget_recheck/summary.json'),
        'additional_correct_boundary_sentences': 5195, 'limit_gpu_hours': 15,
        'reason': 'Original-clip service equivalence did not cover S1 waveform alignment and global video sampling',
        'gpu_seconds': 0})
    if not passed:
        raise RuntimeError('STOP supplementary actual-stream checks exceed GPU estimate limit')


if __name__ == '__main__':
    run()
