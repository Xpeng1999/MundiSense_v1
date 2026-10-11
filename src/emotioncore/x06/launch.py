"""Detach one bounded GPU watchdog so an SSH interruption cannot remove it."""
import argparse
import os
import subprocess
import sys
from emotioncore.x06.common import *
from emotioncore.x06.gpu_guard import TASKS


def run(task):
    output = ROOT / 'launches' / task
    meta = start(output)
    command = [sys.executable, '-u', '-m', 'emotioncore.x06.gpu_guard', task]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HOME': str(ROOT / 'cache/hf'), 'XDG_CACHE_HOME': str(ROOT / 'cache'),
           'MODELSCOPE_CACHE': str(ROOT / 'cache/modelscope'), 'TORCH_HOME': str(ROOT / 'cache/torch'),
           'TRITON_CACHE_DIR': str(ROOT / 'cache/triton')}
    with (output / 'watchdog.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'status': 'dispatched', 'pid': child.pid, 'command': command,
               'gpu_job': str(ROOT / 'gpu_jobs' / task), 'watchdog_log': str(output / 'watchdog.log')}
    finish(output, meta, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=('emotion_segments', 'realtime_H3', 'realtime_H_ref'))
    run(parser.parse_args().task)
