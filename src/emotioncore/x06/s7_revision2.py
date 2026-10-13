"""Detached sequential replay with an audit and fresh budget between systems."""
import argparse
import os
import subprocess
import sys
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, current_budget, entry_gate, authorize, done


def steps():
    yield 'budget_before_s7', 'budget_revision2', ['--stage', 'before_s7']
    for system in ('H3', 'H_ref'):
        yield 'realtime_' + system, 'gpu_guard', ['realtime_' + system]
        yield 'audit_' + system, 'live_audit_revision2', [system]
        yield 'budget_after_' + system, 'budget_revision2', ['--stage', 'after_' + system]


def execute(output, resume_audit=False):
    records = []
    plan = list(steps())
    if resume_audit:
        plan = plan[2:]
        plan[0] = ('audit_H3_retry', 'live_audit_revision2', ['H3', '--retry'])
    for name, module, args in plan:
        command = [sys.executable, '-u', '-m', 'emotioncore.x06.' + module, *args]
        record = {'stage': name, 'command': command, 'started_at': now()}
        write_json_atomic(output / 'progress.json', {'current': record, 'completed': records})
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0' if module == 'gpu_guard' else '',
               'PYTHONDONTWRITEBYTECODE': '1'}
        with (output / (name + '.log')).open('w') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        record.update(ended_at=now(), returncode=child.returncode, log=spec(output / (name + '.log')))
        records.append(record)
        if child.returncode:
            return records, name
    return records, None


def resume_gate():
    authorize()
    assert done('realtime_H3')
    assert not (REV / 's6a_runtime_stop.json').exists()
    previous = read(REV / 's7_driver/summary.json')
    assert previous['status'] == 'stopped' and previous['problem_stage'] == 'audit_H3'


def run(resume_audit=False):
    if resume_audit:
        resume_gate()
    else:
        current_budget()
        entry_gate('realtime_H3')
    output = REV / ('s7_driver_retry' if resume_audit else 's7_driver'); meta = start(output)
    records, problem = execute(output, resume_audit)
    finish(output, meta, {'status': 'stopped' if problem else 'completed',
        'problem_stage': problem, 'stages': records,
        'next_step': 'final provenance audit after this driver exits; collect results and report'})
    if problem:
        raise RuntimeError('STOP sequential S7 stage failed: ' + problem)


def launch(resume_audit=False):
    if resume_audit:
        resume_gate()
    else:
        current_budget()
        entry_gate('realtime_H3')
    output = REV / ('s7_launch_retry' if resume_audit else 's7_launch'); meta = start(output)
    command = [sys.executable, '-u', '-m', 'emotioncore.x06.s7_revision2']
    if resume_audit:
        command.append('--resume-audit')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HOME': str(ROOT / 'cache/hf'), 'XDG_CACHE_HOME': str(ROOT / 'cache'),
           'MODELSCOPE_CACHE': str(ROOT / 'cache/modelscope'), 'TORCH_HOME': str(ROOT / 'cache/torch'),
           'TRITON_CACHE_DIR': str(ROOT / 'cache/triton')}
    with (output / 'driver.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    finish(output, meta, {'pid': child.pid, 'command': command, 'status': 'dispatched'})
    print(json.dumps(read(output / 'summary.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--launch', action='store_true')
    parser.add_argument('--resume-audit', action='store_true'); args = parser.parse_args()
    launch(args.resume_audit) if args.launch else run(args.resume_audit)
