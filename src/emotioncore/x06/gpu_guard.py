"""Eighteen-hour shared full-process wall-time budget for X06a inference only."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from emotioncore.x06.common import *

TASKS = {'rev2_et': ['emotioncore.x06.resume_et'],
         'rev2_s2_04': ['emotioncore.x06.segments_revision2', '--silence', '0.4'],
         'rev2_s2_08': ['emotioncore.x06.segments_revision2', '--silence', '0.8'],
         'equivalence': ['emotioncore.x06.equivalence'],
         'stream_equivalence': ['emotioncore.x06.equivalence', '--stream'],
         'fmh_stream_oracle': ['emotioncore.x06.fmh_stream_oracle'],
         'emotion_segments': ['emotioncore.x06.emotion_segments'],
         'realtime_H3': ['emotioncore.x06.run', '--batch', '--system', 'H3'],
         'realtime_H_ref': ['emotioncore.x06.run', '--batch', '--system', 'H_ref'],
         'offline': ['emotioncore.x06.run']}


def run(task, attempt=1, child_args=()):
    revised = (ROOT / 'revision2/budget_current.json').exists()
    if revised:
        from emotioncore.x06.revision2 import entry_gate
        entry_gate(task)
    else:
        revision_budget = ROOT / 'revision_1_budget/summary.json'
        if revision_budget.exists() and read(revision_budget)['status'] != 'passed':
            raise RuntimeError('STOP revision-1 startup budget; no further GPU work')
        stopped = ROOT / 'stream_equivalence/summary.json'
        if stopped.exists() and read(stopped)['status'] == 'STOP_equivalence':
            raise RuntimeError('STOP historical S6; no further GPU work without authorized revision')
        raise RuntimeError('STOP revision-2 startup budget required; no further GPU work')
    suffix = '' if attempt == 1 else f'_attempt{attempt}'
    output = ROOT / 'gpu_jobs' / (task + suffix)
    meta = start(output)
    lock = (ROOT / 'gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = list((ROOT / 'gpu_jobs').glob('*/summary.json'))
    previous = [read(p) for p in paths]
    used = sum(r['charged_gpu_s'] for r in previous)
    for path, result in zip(paths, previous):
        if result['status'] != 'completed':
            from emotioncore.x06.revision_budget import HISTORICAL
            if (path == ROOT / 'gpu_jobs/stream_equivalence/summary.json'
                    and sha256(path) == HISTORICAL['gpu_jobs/stream_equivalence/summary.json']):
                continue  # authorized by revisions 1/2; historical failure remains charged
            repair = read(ROOT / 'dependency_repair/summary.json')
            if (repair['status'] != 'repaired_before_inference' or not repair['cpu_load_passed']
                    or repair['failed_job'] != spec(path)
                    or repair['failed_log'] != spec(path.parent / 'process.log')):
                raise RuntimeError('STOP prior GPU job failed without a verified pre-inference repair')
    if used >= MAX_GPU_SECONDS:
        raise RuntimeError('STOP eighteen-hour GPU deadline')
    if child_args and task != 'offline':
        raise ValueError('Only the bounded offline video entry takes input arguments')
    command = [sys.executable, '-u', '-m', *TASKS[task], *child_args]
    meta.update(command_child=command, prior_charged_gpu_s=used)
    write_json_atomic(output / 'run_meta.json', meta)
    begin = time.monotonic(); problem = None
    env = {**os.environ, 'X06_GPU_GUARDED': task, 'X06_RUN_SUFFIX': suffix, 'PYTHONDONTWRITEBYTECODE': '1'}
    with (output / 'process.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - begin
                if elapsed + used >= MAX_GPU_SECONDS:
                    raise RuntimeError('STOP eighteen-hour GPU deadline')
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
    parser.add_argument('--attempt', type=int, choices=range(1, 10), default=1)
    args = parser.parse_args()
    run(args.task, args.attempt)
