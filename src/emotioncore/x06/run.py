"""Offline video entry and ET-test real-time replay under the shared GPU budget."""
import argparse
import os
from emotioncore.x06.common import *
from emotioncore.x06.streams import validate_input, render_audio
from emotioncore.x06.live import consume


def event_destination(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / 'offline_runs' / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('STOP outputs must remain inside the new X06a run root')
    return path


def run(args):
    row = None if args.batch else validate_input(args.input)
    if args.batch:
        task = f'realtime_{args.system}'
        output = ROOT / task
        event_path = output / 'events.jsonl'
    else:
        task = 'offline'
        event_path = event_destination(args.out)
        output = event_path.parent / (event_path.stem + '.run')
    if not os.environ.get('X06_GPU_GUARDED'):
        from emotioncore.x06.gpu_guard import run as guarded
        if args.batch:
            return guarded(task)
        child = ['--input', str(Path(args.input).resolve()), '--system', args.system, '--out', str(event_path)]
        if args.realtime:
            child.append('--realtime')
        attempt = 1 + len(list((ROOT / 'gpu_jobs').glob('offline*/run_meta.json')))
        return guarded(task, attempt, child)
    if os.environ['X06_GPU_GUARDED'] != task:
        raise RuntimeError('STOP mismatched GPU task')
    if read(ROOT / 'budget_recheck/summary.json')['status'] != 'passed':
        raise RuntimeError('STOP updated budget estimate has not passed')
    frozen = read(ROOT / 'smoothing/freeze.json')
    alpha = frozen['alpha']
    meta = start(output)
    if event_path.exists():
        raise FileExistsError('Event output already exists')
    event_path.parent.mkdir(parents=True, exist_ok=True)
    if args.batch:
        sources = {s: v for s, v in read(ROOT / 'streams/summary.json')['streams'].items() if s in TEST}
        if set(sources) != set(TEST):
            raise ValueError('STOP all four ET-test streams are required')
    else:
        from emotioncore.x06.simulator import build_timelines
        timeline = build_timelines([row])[row['speaker_id']]
        timeline_path = output / 'timeline.json'
        write_json_atomic(timeline_path, timeline)
        receipt = render_audio(timeline, output / 'audio.wav')
        sources = {row['speaker_id']: {'timeline': spec(timeline_path), 'audio': receipt['audio']}}
    for source in sources.values():
        for kind in ('timeline', 'audio'):
            if sha256(source[kind]['path']) != source[kind]['sha256']:
                raise ValueError('STOP stream provenance changed')
    result = consume(sources, args.system, output, event_path, alpha, realtime=args.batch or args.realtime)
    result.update(smoothing_freeze=spec(ROOT / 'smoothing/freeze.json'), training=False,
                  inference_replayed=False, bundle=spec('configs/calibration_bundle_v1.1.json'))
    finish(output, meta, result)
    print(json.dumps({'events': str(event_path), 'summary': str(output / 'summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path)
    parser.add_argument('--system', choices=('H3', 'H_ref'), default='H3')
    parser.add_argument('--out', default='events.jsonl')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--batch', action='store_true', help='formal four-stream ET-test real-time replay')
    args = parser.parse_args()
    if not args.batch and args.input is None:
        parser.error('--input is required')
    run(args)
