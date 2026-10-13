"""Revision authorization, immutable receipts, and stage order."""
from emotioncore.x06.common import *
from emotioncore.x06.revision_budget import HISTORICAL, check_s6a

REV = ROOT / 'revision2'
TASK_OUTPUTS = {'rev2_et': REV / 'et_remaining', 'fmh_stream_oracle': ROOT / 'fmh_stream_oracle',
                'rev2_s2_04': REV / 'emotion_04', 'rev2_s2_08': REV / 'emotion_08',
                'realtime_H3': ROOT / 'realtime_H3', 'realtime_H_ref': ROOT / 'realtime_H_ref'}


def checked(source):
    if sha256(source['path']) != source['sha256']:
        raise ValueError('STOP artifact SHA changed: ' + source['path'])
    return source['path']


def authorize():
    if sha256('experiments/X06a_stream/protocol/X06A_REVISION_2.md') != REVISION_2_SHA:
        raise RuntimeError('STOP unauthorized revision 2')
    for name, expected in HISTORICAL.items():
        if sha256(ROOT / name) != expected:
            raise ValueError('STOP historical receipt changed: ' + name)
    check_s6a(read(ROOT / 'equivalence_attempt2/summary.json'))
    # Both prior STOPs remain immutable; only their protocol meaning changes.
    receipt = read('experiments/X06a_stream/results/revision_1_budget.json')['source']
    checked(receipt)


def ledger():
    return [{'source': spec(p), **read(p)} for p in sorted((ROOT / 'gpu_jobs').glob('*/summary.json'))]


def done(task):
    path = ROOT / 'gpu_jobs' / task / 'summary.json'
    return path.exists() and read(path)['status'] == 'completed'


def live_audit_summary(system):
    retry = REV / ('live_audit_' + system + '_retry')
    directory = retry if retry.exists() else REV / ('live_audit_' + system)
    return read(checked(read(directory / 'run_meta.json')['summary']))


def s6b_gate(summary):
    if summary['cohort'] != 'fmh' or summary['n'] != 3266:
        raise RuntimeError('STOP S6b requires all FMH utterances')
    if summary['systems']['H3']['op_n']['delta'] < -.05:
        raise RuntimeError('STOP FMH S6b point estimate below -0.05')


def current_budget():
    pointer = REV / 'budget_current.json'
    if not pointer.exists():
        raise RuntimeError('STOP no revision-2 budget; no further GPU work')
    summary = read(checked(read(pointer)))
    if summary['status'] != 'passed':
        raise RuntimeError('STOP revision-2 estimated budget; no further GPU work')
    if summary['ledger'] != [r['source'] for r in ledger()]:
        raise RuntimeError('STOP update budget after the last completed stage')
    return summary


def entry_gate(task):
    authorize()
    if (REV / 's6a_runtime_stop.json').exists():
        raise RuntimeError('STOP S6a identical-input runtime discrepancy')
    current_budget()
    if task in ('equivalence', 'stream_equivalence', 'emotion_segments'):
        raise RuntimeError('Historical entry is immutable; use revision-2 entries')
    if task != 'rev2_et' and not done('rev2_et'):
        raise RuntimeError('STOP ET remainder must finish first')
    if task not in ('rev2_et', 'fmh_stream_oracle'):
        s6b_gate(read(REV / 's6b_fmh/summary.json'))
    if task == 'rev2_s2_08' and not done('rev2_s2_04'):
        raise RuntimeError('STOP first S2 configuration must finish')
    if task in ('realtime_H3', 'realtime_H_ref', 'offline'):
        if not all(done(t) for t in ('rev2_s2_04', 'rev2_s2_08')):
            raise RuntimeError('STOP both S2 configurations required')
        for server, committed in [('s2/freeze.json', 'x06_silence_freeze.json'),
                                  ('s5/freeze.json', 'x06_smoothing_freeze.json')]:
            if sha256(REV / server) != sha256(Path('configs') / committed):
                raise RuntimeError('STOP selection freeze has not been committed unchanged')
        if not (REV / 's3/summary.json').exists():
            raise RuntimeError('STOP S3 report must precede replay')
    if task == 'realtime_H_ref' and not done('realtime_H3'):
        raise RuntimeError('STOP all four H3 streams must precede H_ref speaker 21')
    if task == 'realtime_H_ref' and live_audit_summary('H3')['status'] != 'completed':
        raise RuntimeError('STOP H3 live audit has not completed')


def selected_vad():
    config = dict(read('configs/x06_vad_freeze.json')['selected'])
    frozen = read(REV / 's2/freeze.json')
    if frozen['model'] != config['model'] or frozen['threshold'] != config['threshold']:
        raise RuntimeError('STOP frozen VAD model/threshold changed')
    config['min_silence'] = frozen['min_silence']
    return config


def selected_cuts():
    return ROOT / 'cuts' if selected_vad()['min_silence'] == .8 else REV / 'cuts_04'
