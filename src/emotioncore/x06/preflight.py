"""Read-only input inventory and a budget estimate before any GPU launch."""
import datetime as dt
import statistics
from emotioncore.x06.common import *
from emotioncore.x06.simulator import build_timelines
from emotioncore.x06.gate import verify_only_gate_changed


def run():
    output = ROOT / 'preflight'
    meta = start(output)
    verify_only_gate_changed(Path('configs/calibration_bundle_v1.json').read_bytes(),
                             Path('configs/calibration_bundle_v1.1.json').read_bytes())
    original = rows(MANIFEST)
    selected = [r for r in original if r['speaker_id'] in FMH | TEST]
    assert len(selected) == 5195
    assert all((r['speaker_id'] in TEST) == (r['official_split'] == 'test') for r in selected)
    timelines = build_timelines(selected)
    for speaker, timeline in timelines.items():
        write_json_atomic(output / f'timeline_{speaker}.json', timeline)
    durations = {s: {'hours': t['duration_s'] / 3600, 'utterances': t['utterances']}
                 for s, t in timelines.items()}
    rates = {}
    for form in ('A', 'V', 'AV'):
        source = R5 / f'Q/mix_fmh/{form}.jsonl'
        receipts = rows(source)
        rates[form] = {'seconds_per_sentence': statistics.mean(r['latency_s'] for r in receipts),
                       'source': spec(source), 'n': len(receipts)}
    feature_rates = {}
    sample = sorted((r for r in selected if r['speaker_id'] in FMH),
                    key=lambda r: r['utt_id'])[::50]
    for encoder in ('A3', 'V2', 'V1'):
        values = [read(R2 / f'features/{encoder}/receipts/{r["utt_id"]}.json') for r in sample]
        elapsed = [r['model_seconds'] for r in values if 'model_seconds' in r]
        feature_rates[encoder] = {'seconds_per_sentence': statistics.mean(elapsed), 'n': len(elapsed)}
    face = read(R2 / 'face/run_meta.json')
    face_s = ((dt.datetime.fromisoformat(face['ended_at']) -
               dt.datetime.fromisoformat(face['started_at'])).total_seconds()
              * face['workers'] / face['completed'])
    q_both = rates['A']['seconds_per_sentence'] + rates['AV']['seconds_per_sentence']
    expert = sum(r['seconds_per_sentence'] for r in feature_rates.values())
    # Full-speed stages prepare raw-media face inputs on CPU before loading Q.
    # Shared raw component predictions support both systems; no output replay.
    fullspeed_s = (1929 + 5195 * 1.5) * (q_both + expert)
    realtime_seconds = 2 * sum(t['duration_s'] for s, t in timelines.items() if s in TEST)
    serial_service = 1929 * (2 * face_s + rates['AV']['seconds_per_sentence'] + q_both + 2 * expert)
    realtime_estimate = max(realtime_seconds, serial_service)
    base = fullspeed_s + realtime_estimate
    estimate = base * 1.2 + 600
    aishell = {}
    for split in ('train_S', 'train_M', 'train_L', 'test'):
        annotations = sorted((AISHELL / split / 'TextGrid').glob('*.rttm'))
        missing = []
        for annotation in annotations:
            path = AISHELL / split / 'wav' / (annotation.stem + '.flac')
            if not path.is_file() or path.stat().st_size < 1024:
                missing.append(annotation.stem)
        aishell[split] = {'rttm_count': len(annotations), 'missing_audio': missing}
    summary = {'status': 'passed' if estimate <= MAX_GPU_SECONDS else 'STOP_budget_estimate',
               'gpu_hours_used': 0, 'estimated_gpu_hours': estimate / 3600,
               'limit_gpu_hours': 15, 'streams': durations, 'q_rates': rates,
               'feature_rates': feature_rates, 'x02_serial_face_seconds_per_clip': face_s,
               'fullspeed_estimate_hours': fullspeed_s / 3600,
               'two_realtime_input_hours': realtime_seconds / 3600,
               'two_realtime_service_estimate_hours': realtime_estimate / 3600,
               'reserve': '20 percent plus ten minutes; VAD segment count assumed 1.5 per original sentence',
               'budget_recheck': 'Recompute against actual VAD segment counts before GPU launch',
               'aishell': aishell, 'source_manifest': spec(MANIFEST),
               'face_run_meta': spec(R2 / 'face/run_meta.json'),
               'bundle_v1_1': spec('configs/calibration_bundle_v1.1.json')}
    finish(output, meta, summary)
    print(__import__('json').dumps(summary, ensure_ascii=False, indent=2))
    if summary['status'] != 'passed':
        raise RuntimeError('STOP startup estimate exceeds 15 GPU hours')
    if any(s['missing_audio'] for s in aishell.values()):
        raise RuntimeError('AISHELL-4 annotations have missing audio; do not silently subset')


if __name__ == '__main__':
    run()
