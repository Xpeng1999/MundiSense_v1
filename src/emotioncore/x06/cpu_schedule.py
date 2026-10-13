"""Reversible scheduling of the already-running CPU segment preparation only."""
import argparse
import psutil
from emotioncore.x06.common import *


def run(resume=False):
    output = ROOT / ('resume_segment_preparation' if resume else 'pause_segment_preparation')
    meta = start(output)
    if resume:
        records = read(ROOT / 'pause_segment_preparation/summary.json')['processes']
        resumed = []
        for record in reversed(records):
            try:
                process = psutil.Process(record['pid'])
                if process.create_time() != record['created_at']:
                    raise RuntimeError('PID was reused; refusing to signal another process')
                process.resume(); resumed.append(record['pid'])
            except psutil.NoSuchProcess:
                continue
        finish(output, meta, {'resumed': resumed, 'gpu_seconds': 0})
        return
    matches = []
    for process in psutil.process_iter(['pid', 'ppid', 'cmdline']):
        if process.info['cmdline'] and process.info['cmdline'][-1] == 'emotioncore.x06.prepare_segments':
            matches.append(process)
    ids = {p.pid for p in matches}
    roots = [p for p in matches if p.ppid() not in ids]
    if len(roots) != 1:
        raise RuntimeError(f'Expected one CPU preparation parent, found {len(roots)}')
    root = roots[0]
    root.suspend()
    children = root.children(recursive=True)
    records = []
    for process in [root, *children]:
        try:
            process.suspend()
            records.append({'pid': process.pid, 'created_at': process.create_time(), 'command': process.cmdline()})
        except psutil.NoSuchProcess:
            continue
    finish(output, meta, {'processes': records, 'reason': 'Prioritize actual-stream S6 media preparation before dependent work',
                         'model_parameters_changed': False, 'gpu_seconds': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', action='store_true')
    run(parser.parse_args().resume)
