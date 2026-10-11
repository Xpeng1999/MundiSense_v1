"""Fifteen-hour shared full-process wall-time budget for X06a inference only."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from emotioncore.x06.common import *

TASKS = {'equivalence': ['emotioncore.x06.equivalence']}


def run(task):
    output = ROOT / 'gpu_jobs' / task
    meta = start(output)
    lock = (ROOT / 'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = [read(p) for p in (ROOT / 'gpu_jobs').glob('*/summary.json')]
    used = sum(r['charged_gpu_s'] for r in previous)
    if any(r['status'] != 'completed' for r in previous):
        raise RuntimeError('STOP prior GPU job failed')
    if read(ROOT / 'preflight/summary.json')['estimated_gpu_hours'] > 15:
        raise RuntimeError('STOP startup budget estimate')
    command = [sys.executable, '-u', '-m', *TASKS[task]]
    meta.update(command_child=command, prior_charged_gpu_s=used)
    write_json_atomic(output / 'run_meta.json', meta)
    begin = time.monotonic(); problem = None
    env = {**os.environ, 'X06_GPU_GUARDED': task, 'PYTHONDONTWRITEBYTECODE': '1'}
    with (output / 'process.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - begin
                if elapsed + used >= MAX_GPU_SECONDS:
                    raise RuntimeError('STOP fifteen-hour GPU deadline')
                storage_gate()
                write_json_atomic(output / 'progress.json', {'at': now(), 'pid': child.pid,
                                  'charged_gpu_s': elapsed, 'total_charged_gpu_s': used + elapsed})
                try:
                    child.wait(timeout=min(10, MAX_GPU_SECONDS - used - elapsed))
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                problem = f'child_exit_{child.returncode}'
        except BaseException as exc:
            problem = f'{type(exc).__name__}: {exc}'
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        finally:
            charged = time.monotonic() - begin
            finish(output, meta, {'status': 'stopped' if problem else 'completed', 'problem': problem,
                'returncode': child.returncode, 'charged_gpu_s': charged, 'total_charged_gpu_s': used + charged,
                'accounting': 'full child wall time, including initialization, CPU work and playback waits'})
    if problem:
        raise RuntimeError(problem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('task', choices=TASKS)
    run(parser.parse_args().task)
