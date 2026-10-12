"""X06a immutable inputs and formal-run provenance."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import sys
import importlib.metadata
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.prepare import storage_gate, require_h

ROOT = Path('/mnt/h/emotionAI/emotioncore_runs/X06a')
R2 = Path('/home/administrator/emotion-data/emotioncore_runs/X02')
R3 = Path('/home/administrator/emotion-data/emotioncore_runs/X03')
R4B = Path('/mnt/h/emotionAI/emotioncore_runs/X04b')
R5 = Path('/mnt/h/emotionAI/emotioncore_runs/X05')
MEDIA = Path('/home/administrator/emotion-data/emotionAI_processed/phase1_v1/emotiontalk')
AISHELL = Path('/home/administrator/emotion-data/emotionAI_dataset/AISHELL-4')
MANIFEST = Path('experiments/X01_benchmark/results/et_utterances.jsonl')
FMH = frozenset(('06', '10', '11'))
TEST = frozenset(('02', '13', '20', '21'))
BUNDLE_SHA = '94a0805b9775676209306187a3880595b7e7592af70aaec8d50f3b2a7968979a'
MAX_GPU_SECONDS = 18 * 3600  # X06A_REVISION_1, including every historical GPU job
REVISION_SHA = '0f0cfbe879b04be7ac750e51228214b5923f5db64924b834441a7771a2540fc1'
REVISION_2_SHA = '58287378b00854c4246d5523920022c5842f5eeb84fe9ab6f4c4785047935820'
SEED = 20261011


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def spec(path):
    return {'path': str(path), 'sha256': sha256(path)}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def start(output):
    git = git_provenance(Path.cwd())
    storage = storage_gate()
    require_h(output)
    if sha256('configs/calibration_bundle_v1.json') != BUNDLE_SHA:
        raise ValueError('STOP X05 fitted parameters changed')
    from emotioncore.x03.q_manifest import FOLD_SHA
    from emotioncore.x03.budget_v2 import FACE_SHA, SUBSET_SHA
    for name, expected in (('folds_v1.json', FOLD_SHA),
                           ('x03_face_missing_policy.json', FACE_SHA),
                           ('x03_inner_val_subsets.json', SUBSET_SHA)):
        if sha256(Path('configs') / name) != expected:
            raise ValueError('STOP frozen split/policy changed')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_meta.json').exists():
        raise FileExistsError('Do not overwrite an existing X06a run')
    meta = {'git': git, 'command': sys.argv, 'started_at': now(), 'storage': storage,
            'protocol': spec('experiments/X06a_stream/protocol/X06A_PROMPT.md'),
            'review': spec('experiments/X06a_stream/protocol/X05_REVIEW.md'),
            'bundle_v1': spec('configs/calibration_bundle_v1.json')}
    revision = Path('experiments/X06a_stream/protocol/X06A_REVISION_1.md')
    if sha256(revision) != REVISION_SHA:
        raise ValueError('STOP X06a reviewer revision changed')
    meta['revision_1'] = spec(revision)
    revision2 = Path('experiments/X06a_stream/protocol/X06A_REVISION_2.md')
    if sha256(revision2) != REVISION_2_SHA:
        raise ValueError('STOP X06a reviewer revision 2 changed')
    meta['revision_2'] = spec(revision2)
    frozen_vad = Path('configs/x06_vad_freeze.json')
    if frozen_vad.exists():
        if sha256(frozen_vad) != sha256(ROOT / 'aishell_train_parallel/freeze.json'):
            raise ValueError('STOP VAD parameters differ from the committed train-only freeze')
        meta['vad_freeze'] = spec(frozen_vad)
    meta['python'] = sys.version.split()[0]
    meta['packages'] = {}
    for name in ('numpy', 'torch', 'transformers', 'opencv-python', 'opencv-python-headless', 'timm', 'silero-vad', 'funasr'):
        try:
            meta['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json_atomic(output / 'run_meta.json', meta)
    return meta


def finish(output, meta, summary):
    write_json_atomic(output / 'summary.json', summary)
    meta.update(ended_at=now(), summary=spec(output / 'summary.json'))
    write_json_atomic(output / 'run_meta.json', meta)


def assert_tuning_source(cohort, speakers=None):
    if cohort == 'fmh' and speakers and set(speakers) <= FMH:
        return
    if cohort == 'aishell_train' and speakers is None:
        return
    raise ValueError('STOP tuning data must be FMH or AISHELL-4 train')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
