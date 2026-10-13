"""Sequential revision-2 execution through S5; every stage remains fail-closed."""
import argparse
import os
import subprocess
import sys
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, done, current_budget


def steps():
    phases = [
        ('fmh_inputs', 'prepare_stream_oracle', ['--fmh']),
        ('fmh', 'gpu_guard', ['fmh_stream_oracle']),
        ('s6b_fmh', 's6b', ['fmh']),
        ('s6b_et', 's6b', ['et_test']),
        ('cuts_04', 'prepare_revision2', ['cuts']),
        ('inputs_04', 'prepare_revision2', ['inputs']),
        ('s2_04', 'gpu_guard', ['rev2_s2_04']),
        ('s2_08', 'gpu_guard', ['rev2_s2_08']),
        ('s2', 'analyze_revision2', ['s2']),
        ('s3', 'analyze_revision2', ['s3']),
        ('s5', 'analyze_revision2', ['s5'])]
    for name, module, args in phases:
        yield name, module, args
        yield 'budget_after_' + name, 'budget_revision2', ['--stage', 'after_' + name]


def run():
    if not done('rev2_et'):
        raise RuntimeError('STOP ET remainder must be completed before this sequence')
    current_budget()
    output = REV / 'pre_s7_driver'; meta = start(output)
    records = []; status = 'completed'; problem = None
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
            status = 'stopped'; problem = name; break
    finish(output, meta, {'status': status, 'problem_stage': problem, 'stages': records,
        'replay_started': False, 'next_step': 'commit FMH-only silence/EMA freezes, verify, then budget-gated S7'})
    if problem:
        raise RuntimeError('STOP sequential stage failed: ' + problem)


def launch():
    output = REV / 'pre_s7_launch'; meta = start(output)
    command = [sys.executable, '-u', '-m', 'emotioncore.x06.pre_s7']
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
