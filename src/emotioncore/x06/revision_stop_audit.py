"""Deliver revision-1 budget STOP evidence without starting any model job."""
import fcntl
import os
import subprocess
import psutil
from emotioncore.x06.common import *
from emotioncore.x06.revision_budget import HISTORICAL, check_s6a
from emotioncore.x06.gate import verify_only_gate_changed


def run():
    output = ROOT / 'revision_1_stop_audit'
    meta = start(output)
    budget = read(ROOT / 'revision_1_budget/summary.json')
    assert budget['status'] == 'STOP_budget_estimate'
    assert budget['estimated_total_gpu_hours'] > MAX_GPU_SECONDS / 3600
    evidence = {}
    for name in ('revision_1_budget', 'unit_tests_revision_1_budget',
                 'unit_tests_revision_1_budget_attempt2', 'stop_delivery_audit_v2'):
        path = ROOT / name
        provenance = read(path / 'run_meta.json')
        assert provenance['ended_at'] and not provenance['git']['dirty']
        assert provenance['summary'] == spec(path / 'summary.json')
        evidence[name] = {'summary': spec(path / 'summary.json'), 'run_meta': spec(path / 'run_meta.json')}
    tests = read(ROOT / 'unit_tests_revision_1_budget_attempt2/summary.json')
    assert tests['returncode'] == 0 and tests['totals']['failures'] == tests['totals']['errors'] == 0
    for path, expected in HISTORICAL.items():
        assert sha256(ROOT / path) == expected
    check_s6a(read(ROOT / 'equivalence_attempt2/summary.json'))
    ledger = [{'source': spec(p), **read(p)} for p in sorted((ROOT / 'gpu_jobs').glob('*/summary.json'))]
    assert [r['source'] for r in ledger] == budget['ledger']
    assert sum(r['charged_gpu_s'] for r in ledger) == budget['used_gpu_s']
    downstream = ('fmh_stream_oracle', 'emotion_segments', 'smoothing', 'realtime_H3', 'realtime_H_ref')
    assert not any((ROOT / name).exists() for name in downstream)
    active = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        command = p.info['cmdline'] or []
        if p.pid != os.getpid() and '-m' in command:
            module = command[command.index('-m') + 1]
            if module.startswith('emotioncore.x06.'):
                active.append({'pid': p.pid, 'command': command})
    assert not active
    with (ROOT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_only_gate_changed(Path('configs/calibration_bundle_v1.json').read_bytes(),
                             Path('configs/calibration_bundle_v1.1.json').read_bytes())
    frozen_paths = ['src/emotioncore/x02', 'src/emotioncore/x03', 'src/emotioncore/x04',
                    'src/emotioncore/x05', 'src/emotioncore/x06/service.py',
                    'src/emotioncore/x06/equivalence.py', 'configs/calibration_bundle_v1.json',
                    'configs/calibration_bundle_v1.1.json', 'configs/x06_vad_freeze.json']
    unchanged = subprocess.check_output(['git', 'diff', '--name-only', 'b75a691', 'HEAD', '--', *frozen_paths], text=True)
    assert not unchanged.strip()
    finish(output, meta, {'status': 'STOP_budget_estimate', 'evidence': evidence,
        'estimated_total_gpu_hours': budget['estimated_total_gpu_hours'], 'limit_gpu_hours': 18,
        'total_charged_gpu_hours': budget['used_gpu_hours'], 'new_gpu_seconds': 0,
        'gpu_ledger': ledger, 'active_x06_processes': active, 'gpu_lock_available': True,
        'downstream_runs_absent': list(downstream), 'historical_artifacts_unchanged': True,
        'bundle_only_gate_bytes_changed': True, 'frozen_paths_unchanged_from_b75a691': frozen_paths,
        'tests': tests['totals'], 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
