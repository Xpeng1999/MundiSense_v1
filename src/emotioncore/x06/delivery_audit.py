"""CPU-only delivery evidence; compare live output with fresh full-speed inference."""
import subprocess
import numpy as np
from emotioncore.x06.common import *
from emotioncore.x06.decision import apply
from emotioncore.x06.gate import verify_only_gate_changed

REQUIRED = (
    'preflight', 'oracle_inputs', 'equivalence_attempt2', 'equivalence_audit_v2',
    'aishell_train_parallel', 'aishell_test', 'streams', 'cuts', 'budget_recheck',
    'fmh_faces', 'segment_inputs', 'emotion_segments', 'smoothing',
    'stream_oracle_budget', 'stream_oracle_inputs', 'stream_equivalence',
    'fmh_stream_oracle_inputs', 'fmh_stream_oracle',
    'event_schedule_audit', 'live_cpu_probe_selected_attempt2',
    'realtime_H3', 'realtime_H_ref',
)


def live_comparison(system, fullspeed, alpha, bundle):
    live = rows(ROOT / f'realtime_{system}/events.jsonl')
    state = {}; differences = []; maximum = {}
    for event in live:
        source = fullspeed[event['segment_id']]
        speech = not source['visual_only']
        expected = apply(bundle, source['component_probs'], system,
                         has_face=source['has_face'], speech=speech,
                         scores=source['track_scores'], alpha=alpha, state=state,
                         stream_id=source['speaker_id'])
        for name, probability in event['component_probs'].items():
            delta = float(np.max(np.abs(np.asarray(probability) - source['component_probs'][name])))
            maximum[name] = max(maximum.get(name, 0.), delta)
        if event['label'] != expected['label']:
            differences.append({'segment_id': event['segment_id'], 'live': event['label'],
                                'fullspeed': expected['label'], 'live_reason': event['reason'],
                                'fullspeed_reason': expected['reason']})
    return {'n': len(live), 'label_agreement': 1 - len(differences) / len(live),
            'mismatches': len(differences), 'max_absolute_component_difference': maximum}, differences


def run():
    output = ROOT / 'delivery_audit'
    meta = start(output)
    evidence = {}
    for name in REQUIRED:
        directory = ROOT / name
        run_meta = read(directory / 'run_meta.json')
        if not run_meta.get('ended_at') or run_meta['git']['dirty']:
            raise ValueError(f'Unfinished or dirty formal run: {name}')
        actual = spec(directory / 'summary.json')
        if run_meta['summary']['sha256'] != actual['sha256']:
            raise ValueError(f'Summary provenance changed: {name}')
        evidence[name] = {'summary': actual, 'run_meta': spec(directory / 'run_meta.json')}
    verify_only_gate_changed(Path('configs/calibration_bundle_v1.json').read_bytes(),
                             Path('configs/calibration_bundle_v1.1.json').read_bytes())
    changed = subprocess.check_output(['git', 'diff', '--name-only',
                '801320bc75f87aae736ae529c47180e8289fba30', 'HEAD'], text=True).splitlines()
    permitted = ('src/emotioncore/x06/', 'experiments/X06a_stream/',
                 'tests/test_x06_protocol.py', 'configs/calibration_bundle_v1.1.json',
                 'configs/x06_vad_freeze.json', 'configs/x06_smoothing_freeze.json',
                 'docs/X06B_HANDOFF.md')
    unexpected = [path for path in changed if not path.startswith(permitted)]
    if unexpected:
        raise ValueError(f'Frozen-code changes outside X06 scope: {unexpected}')
    ledger = []
    for path in sorted((ROOT / 'gpu_jobs').glob('*/summary.json')):
        ledger.append({'summary': spec(path), **read(path)})
    total = sum(row['charged_gpu_s'] for row in ledger)
    if total > MAX_GPU_SECONDS:
        raise ValueError('STOP GPU budget exceeded')
    unexpected_failures = [row for row in ledger if row['status'] != 'completed'
                          and Path(row['summary']['path']).parent.name != 'equivalence']
    if unexpected_failures:
        raise ValueError('Unresolved GPU failure')
    fullspeed = {r['segment_id']: r for r in rows(ROOT / 'emotion_segments/predictions.jsonl')}
    bundle = read('configs/calibration_bundle_v1.1.json')
    alpha = read(ROOT / 'smoothing/freeze.json')['alpha']
    comparisons = {}
    for system in ('H3', 'H_ref'):
        comparison, differences = live_comparison(system, fullspeed, alpha, bundle)
        path = output / f'{system}_live_fullspeed_differences.json'
        write_json_atomic(path, differences)
        comparisons[system] = {**comparison, 'details': spec(path)}
    finish(output, meta, {'evidence': evidence, 'gpu_ledger': ledger,
        'total_charged_gpu_seconds': total, 'total_charged_gpu_hours': total / 3600,
        'gpu_budget_hours': 15, 'live_vs_fullspeed_diagnostic': comparisons,
        'bundle_only_gate_bytes_changed': True, 'changed_paths': changed,
        'frozen_code_changed': False, 'gpu_seconds': 0})


if __name__ == '__main__':
    run()
