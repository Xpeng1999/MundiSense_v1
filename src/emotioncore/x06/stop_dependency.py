"""Close dependent CPU work after the mandatory actual-stream equivalence stop."""
import psutil
from emotioncore.x06.common import *


def run():
    trigger = ROOT / 'stream_equivalence/summary.json'
    if read(trigger)['status'] != 'STOP_equivalence':
        raise RuntimeError('No actual-stream equivalence stop has occurred')
    output = ROOT / 'stop_after_equivalence'
    meta = start(output)
    task = ROOT / 'segment_inputs'
    terminated = []
    if not (task / 'summary.json').exists():
        identity = read(ROOT / 'pause_segment_preparation/summary.json')['processes'][0]
        try:
            parent = psutil.Process(identity['pid'])
            if parent.create_time() != identity['created_at']:
                raise RuntimeError('PID reused; refusing to terminate another process')
            if parent.cmdline()[-1] != 'emotioncore.x06.prepare_segments':
                raise RuntimeError('Dependent process identity changed')
            processes = [parent, *parent.children(recursive=True)]
            # Suspend the parent first so it cannot submit additional work.
            parent.suspend()
            for process in reversed(processes):
                try:
                    process.kill(); terminated.append(process.pid)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(processes, timeout=5)
        except psutil.NoSuchProcess:
            pass
        previous = read(task / 'run_meta.json')
        progress = read(task / 'progress.json')
        finish(task, previous, {'status': 'STOP_dependency_equivalence', 'trigger': spec(trigger),
            'last_reported_completed_media': progress['done'], 'planned_media': progress['total'],
            'partial_directory': str(task), 'aggregate_manifest_not_created': True, 'gpu_seconds': 0})
    finish(output, meta, {'trigger': spec(trigger), 'terminated_dependent_pids': terminated,
        'dependent_summary': spec(task / 'summary.json'), 'gpu_seconds': 0,
        'continued_emotion_inference': False, 'smoothing_selected': False, 'realtime_runs_started': False})


if __name__ == '__main__':
    run()
