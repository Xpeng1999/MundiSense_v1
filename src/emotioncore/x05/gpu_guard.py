"""Single-GPU process watchdog; all X05 GPU work shares a five-hour ledger."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x05.common import ROOT, start, read, now, storage_gate, finish

TASKS = {
    'reproduce_a3': ['emotioncore.x05.heads', '--encoder', 'A3', '--reproduce'],
    'fmhx_a3': ['emotioncore.x05.heads', '--encoder', 'A3'],
    'fmh_q_et': ['emotioncore.x05.q_fmh', '--version', 'et'],
    'fmh_q_mix': ['emotioncore.x05.q_fmh', '--version', 'mix'],
}


def run(task):
    output = ROOT / 'gpu_jobs' / task
    meta = start(output)
    lock = (ROOT / 'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = list((ROOT / 'gpu_jobs').glob('*/summary.json'))
    used = sum(read(p)['charged_gpu_s'] for p in prior)
    if any(read(p)['status'] != 'completed' for p in prior):
        raise RuntimeError('STOP previous GPU task failed; reviewer/user action required')
    if used >= 18000:
        raise RuntimeError('STOP five-hour GPU budget exhausted')
    command = [sys.executable, '-u', '-m', *TASKS[task]]
    meta.update(prior_charged_gpu_s=used, command_child=command, remaining_gpu_s=18000-used)
    write_json_atomic(output / 'run_meta.json', meta)
    begin = time.monotonic()
    env = {**os.environ, 'X05_GPU_GUARDED': task, 'PYTHONDONTWRITEBYTECODE': '1'}
    problem = None
    with (output / 'process.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            while child.poll() is None:
                elapsed = time.monotonic()-begin
                if elapsed + used >= 18000:
                    raise RuntimeError('STOP five-hour GPU deadline')
                storage_gate()
                write_json_atomic(output / 'progress.json', {'at': now(), 'pid': child.pid,
                    'charged_gpu_s': elapsed, 'total_charged_gpu_s': used+elapsed})
                try:
                    child.wait(timeout=min(10, 18000-used-elapsed))
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                problem = f'child_exit_{child.returncode}'
        except BaseException as exc:
            problem = f'{type(exc).__name__}: {exc}'
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        finally:
            charged = time.monotonic()-begin
            finish(output, meta, {'status': 'stopped' if problem else 'completed',
                'problem': problem, 'returncode': child.returncode,
                'charged_gpu_s': charged, 'total_charged_gpu_s': used+charged,
                'accounting': 'conservative full child wall time including load/CPU preparation'})
    if problem:
        raise RuntimeError(problem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=TASKS)
    run(parser.parse_args().task)
