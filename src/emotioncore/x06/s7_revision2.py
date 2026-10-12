"""Detached sequential replay with an audit and fresh budget between systems."""
import argparse
import os
import subprocess
import sys
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, current_budget, entry_gate


def steps():
    yield 'budget_before_s7', 'budget_revision2', ['--stage', 'before_s7']
    for system in ('H3', 'H_ref'):
        yield 'realtime_' + system, 'gpu_guard', ['realtime_' + system]
        yield 'audit_' + system, 'live_audit_revision2', [system]
        yield 'budget_after_' + system, 'budget_revision2', ['--stage', 'after_' + system]


def execute(output):
    records = []
    for name, module, args in steps():
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


def run():
    current_budget()
    entry_gate('realtime_H3')
    output = REV / 's7_driver'; meta = start(output)
    records, problem = execute(output)
    finish(output, meta, {'status': 'stopped' if problem else 'completed',
        'problem_stage': problem, 'stages': records,
        'next_step': 'final provenance audit after this driver exits; collect results and report'})
    if problem:
        raise RuntimeError('STOP sequential S7 stage failed: ' + problem)


def launch():
    current_budget()
    entry_gate('realtime_H3')
    output = REV / 's7_launch'; meta = start(output)
    command = [sys.executable, '-u', '-m', 'emotioncore.x06.s7_revision2']
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
    launch() if parser.parse_args().launch else run()
