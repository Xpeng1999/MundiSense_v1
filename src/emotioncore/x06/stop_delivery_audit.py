"""Preserve complete evidence for a mandatory S6 stop; no model execution."""
import fcntl
import os
import subprocess
import psutil
from emotioncore.x06.common import *
from emotioncore.x06.gate import verify_only_gate_changed


def run(name='stop_delivery_audit', tests_name='unit_tests_final'):
    output = ROOT / name
    meta = start(output)
    trigger = read(ROOT / 'stream_equivalence/summary.json')
    if trigger['status'] != 'STOP_equivalence':
        raise ValueError('No S6 stop to document')
    downstream = ('fmh_stream_oracle', 'emotion_segments', 'smoothing', 'realtime_H3', 'realtime_H_ref')
    if any((ROOT / name).exists() for name in downstream):
        raise ValueError('Unexpected downstream run after failed S6 gate')
    active = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and '-m' in command:
            module = command[command.index('-m') + 1]
            if module.startswith('emotioncore.x06.'):
                active.append({'pid': process.pid, 'command': command})
    if active:
        raise ValueError(f'X06 processes still active: {active}')
    with (ROOT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_only_gate_changed(Path('configs/calibration_bundle_v1.json').read_bytes(),
                             Path('configs/calibration_bundle_v1.1.json').read_bytes())
    changed = subprocess.check_output(['git', 'diff', '--name-only',
        '801320bc75f87aae736ae529c47180e8289fba30', 'HEAD'], text=True).splitlines()
    permitted = ('src/emotioncore/x06/', 'experiments/X06a_stream/', 'tests/test_x06_protocol.py',
        'configs/calibration_bundle_v1.1.json', 'configs/x06_vad_freeze.json', 'docs/X06B_HANDOFF.md')
    if any(not path.startswith(permitted) for path in changed):
        raise ValueError('Changed file outside X06 scope')
    names = ('preflight', 'streams', 'oracle_inputs', 'equivalence_attempt2', 'equivalence_audit_v2',
        'stream_oracle_inputs', 'stream_equivalence', 'stream_equivalence_audit', 'stop_after_equivalence',
        'fmh_faces', 'segment_inputs', 'cuts', 'aishell_train_parallel', 'aishell_test', 'budget_recheck',
        'stream_oracle_budget', 'event_schedule_audit', 'live_cpu_probe_selected_attempt2', tests_name)
    evidence = {}
    for name in names:
        path = ROOT / name
        run_meta = read(path / 'run_meta.json')
        summary = spec(path / 'summary.json')
        if not run_meta.get('ended_at') or run_meta['git']['dirty'] or run_meta['summary']['sha256'] != summary['sha256']:
            raise ValueError(f'Incomplete or changed provenance: {name}')
        evidence[name] = {'summary': summary, 'run_meta': spec(path / 'run_meta.json')}
    ledger = [{'source': spec(p), **read(p)} for p in sorted((ROOT / 'gpu_jobs').glob('*/summary.json'))]
    total = sum(row['charged_gpu_s'] for row in ledger)
    if total > MAX_GPU_SECONDS:
        raise ValueError('GPU budget exceeded')
    finish(output, meta, {'status': 'STOP_equivalence', 'evidence': evidence, 'gpu_ledger': ledger,
        'total_charged_gpu_seconds': total, 'total_charged_gpu_hours': total / 3600,
        'budget_hours': 15, 'active_x06_processes': active, 'downstream_runs_absent': list(downstream),
        'bundle_only_gate_bytes_changed': True, 'frozen_code_changed': False, 'changed_paths': changed,
        'face_source': spec('src/emotioncore/x02/face.py'), 'gpu_seconds': 0})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='stop_delivery_audit')
    parser.add_argument('--tests-name', default='unit_tests_final')
    args = parser.parse_args()
    run(args.name, args.tests_name)
