"""Register the authorized label-isolation revision without altering v1."""
import copy
from emotioncore.x07.common import *

V1 = Path('configs/x07_lockbox_protocol.json')
V1_SHA = '8ca5430c3c009450d30daa7c4cdf04bb162819d70df21857225069c37eca09c2'
REVISION = Path('experiments/X07_lockbox/protocol/X07_REVISION_1.md')


def build(v1, code, code_hashes, registered_at, provenance):
    protocol = copy.deepcopy(v1)
    protocol.update(version=2, registered_at=registered_at, code_commit=code,
                    code_sha256=code_hashes, revision=provenance)
    protocol['execution'].update(
        root=str(ROOT), protocol=str(PROTOCOL),
        labels='L2 and pre-metric-lock checks read identity columns only, verify the frozen meta SHA, '
               'exact split IDs and one physical line per record; both receipts must match. '
               'After complete.json and O_EXCL metric lock, parse only that split label rows; '
               'all other split rows are skipped before label parsing.')
    return protocol


def run():
    if git('status','--porcelain','--untracked-files=normal'):
        raise RuntimeError('Commit code and preflight evidence before creating protocol v2')
    if sha(V1) != V1_SHA:
        raise RuntimeError('STOP v1 protocol was changed')
    v1, receipt = committed_protocol(V1)
    code = git('rev-parse','HEAD')
    files = [p for p in git('ls-files','src','tests').splitlines() if p.endswith('.py')]
    provenance = {
        'authorization': spec(REVISION),
        'supersedes': {'protocol': spec(V1), 'commit': receipt['commit']},
        'reason': 'X07_REVISION_1.md authorizes isolating split label access and moving identity '
                  'and physical-line checks to L2 and before the one-shot metric lock. '
                  'v1 stopped before GPU inference or any metrics; no test media were read.',
        'allowed_changes': ['label-access code', 'code SHA', 'protocol provenance', 'execution paths'],
        'preserved_execution_root': '/mnt/h/emotionAI/emotioncore_runs/X07',
        'preservation': 'v1 protocol, execution directory, STOP and all records remain unchanged',
        'prior_gpu_seconds': 0,
        'native_decode_failures': 'The seven v1 valid failures keep the frozen unavailable-media rule; '
                                  'no decoder or parameter change is permitted.'}
    protocol = build(v1, code, {p:sha(p) for p in files}, now(), provenance)
    write_new(PROTOCOL, protocol)
    print(json.dumps({'protocol':spec(PROTOCOL),'code_commit':code}))


if __name__ == '__main__':
    run()
