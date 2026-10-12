"""Read-only final/stop provenance, replay comparison and budget accounting."""
import argparse
import fcntl
import os
import subprocess
import numpy as np
import psutil
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, authorize, checked, ledger, done
from emotioncore.x06.gate import verify_only_gate_changed
from emotioncore.x06.live import quantiles


def latency_subset(system, speaker=None):
    root = ROOT / ('realtime_' + system)
    events = {r['segment_id']: r for r in rows(root / 'events.jsonl')}
    receipts = rows(root / 'latency_receipts.jsonl')
    selected = [r for r in receipts if not events[r['segment_id']]['visual_only']
                and (speaker is None or events[r['segment_id']]['speaker_id'] == speaker)]
    return {'n': len(selected), 'latency_seconds': quantiles([r['latency_s'] for r in selected])}, {
        r['segment_id']: r['latency_s'] for r in selected}


def run(name, stopped=False):
    output = REV / name; meta = start(output); authorize()
    active = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        command = p.info['cmdline'] or []
        if p.pid != os.getpid() and '-m' in command:
            module = command[command.index('-m') + 1]
            if module.startswith('emotioncore.x06.'):
                active.append({'pid': p.pid, 'command': command})
    if active:
        raise RuntimeError('STOP delivery requires all model and driver processes to finish')
    with (ROOT / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_only_gate_changed(Path('configs/calibration_bundle_v1.json').read_bytes(),
                             Path('configs/calibration_bundle_v1.1.json').read_bytes())
    frozen = ['src/emotioncore/x02', 'src/emotioncore/x03', 'src/emotioncore/x04', 'src/emotioncore/x05',
              'src/emotioncore/x06/service.py', 'src/emotioncore/x06/face.py',
              'src/emotioncore/x06/prepare_segments.py', 'src/emotioncore/x06/streams.py',
              'configs/calibration_bundle_v1.json', 'configs/calibration_bundle_v1.1.json',
              'configs/x06_vad_freeze.json']
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'b75a691', 'HEAD', '--', *frozen], text=True)
    if changed.strip():
        raise RuntimeError('STOP frozen model/input code changed: ' + changed)
    paths = sorted(REV.glob('*/run_meta.json'))
    paths += [p for task in ('fmh_stream_oracle_inputs', 'fmh_stream_oracle', 'realtime_H3', 'realtime_H_ref')
              if (p := ROOT / task / 'run_meta.json').exists()]
    evidence = {}
    for path in paths:
        if path.parent == output:
            continue
        record = read(path)
        if not record.get('ended_at') or record['git']['dirty']:
            raise RuntimeError('STOP incomplete provenance: ' + str(path))
        checked(record['summary'])
        evidence[str(path.parent)] = {'summary': record['summary'], 'run_meta': spec(path)}
    jobs = ledger(); total = sum(r['charged_gpu_s'] for r in jobs)
    assert total <= MAX_GPU_SECONDS
    if done('rev2_et'):
        old = (ROOT / 'stream_equivalence/predictions.jsonl').read_bytes()
        combined = (REV / 'et_remaining/predictions.jsonl').read_bytes()
        assert combined[:len(old)] == old and len(rows(REV / 'et_remaining/predictions.jsonl')) == 1929
    comparison = None
    if not stopped:
        for task in ('rev2_et', 'fmh_stream_oracle', 'rev2_s2_04', 'rev2_s2_08', 'realtime_H3', 'realtime_H_ref'):
            assert done(task), task
        for server, committed in [('s2/freeze.json', 'x06_silence_freeze.json'), ('s5/freeze.json', 'x06_smoothing_freeze.json')]:
            assert sha256(REV / server) == sha256(Path('configs') / committed)
        h3 = read(ROOT / 'realtime_H3/summary.json'); ref = read(ROOT / 'realtime_H_ref/summary.json')
        assert set(h3['replay_speakers']) == set(TEST) and ref['replay_speakers'] == ['21']
        h3_21, a = latency_subset('H3', '21'); ref_21, b = latency_subset('H_ref', '21')
        assert set(a) == set(b)
        comparison = {'speaker_id': '21', 'speech_segments': len(a), 'H3': h3_21, 'H_ref': ref_21,
            'H_ref_minus_H3_quantiles': {k: ref_21['latency_seconds'][k] - h3_21['latency_seconds'][k]
                                       for k in ('p50', 'p95', 'p99')},
            'paired_segment_delta_seconds': quantiles([b[k] - a[k] for k in sorted(a)]),
            'interpretation': 'one speaker only; P95/P99 descriptive, not a four-stream H_ref estimate'}
    finish(output, meta, {'status': 'stopped' if stopped else 'complete_delivery', 'evidence': evidence,
        'gpu_ledger': jobs, 'total_charged_gpu_s': total, 'total_charged_gpu_hours': total / 3600,
        'limit_gpu_hours': 18, 'active_x06_processes': active, 'gpu_lock_available': True,
        'frozen_paths_unchanged_from_b75a691': frozen, 'bundle_only_gate_bytes_changed': True,
        'replay_comparison': comparison, 'last_budget': read(REV / 'budget_current.json'),
        'original_stops_retained': True, 'gpu_seconds': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='delivery_audit')
    parser.add_argument('--stopped', action='store_true'); args = parser.parse_args()
    run(args.name, args.stopped)
