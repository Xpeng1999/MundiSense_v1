"""Provenance, immutable inputs and X05 gates."""
import datetime as dt
import json
import sys
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import sha256, read_jsonl, FOLD_SHA
from emotioncore.x03.budget_v2 import FACE_SHA, SUBSET_SHA
from emotioncore.x04.prepare import require_h, storage_gate

R2 = Path('/home/administrator/emotion-data/emotioncore_runs/X02')
R3 = Path('/home/administrator/emotion-data/emotioncore_runs/X03')
R4 = Path('/mnt/h/emotionAI/emotioncore_runs/X04a')
R4B = Path('/mnt/h/emotionAI/emotioncore_runs/X04b')
ROOT = Path('/mnt/h/emotionAI/emotioncore_runs/X05')
DEDUP = Path('/home/administrator/emotion-data/emotioncore_runs/X04a/preflight_v3/dedup/dedup_report.json')
FMH = frozenset(('06', '10', '11'))
MANIFEST = Path('experiments/X01_benchmark/results/et_utterances.jsonl')


def read(path):
    return json.loads(Path(path).read_text())


def spec(path):
    return {'path': str(path), 'sha256': sha256(Path(path))}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def start(output):
    git = git_provenance(Path.cwd())
    storage = storage_gate()
    require_h(output)
    for name, expected in [('folds_v1.json', FOLD_SHA), ('x03_face_missing_policy.json', FACE_SHA),
                           ('x03_inner_val_subsets.json', SUBSET_SHA)]:
        if sha256(Path('configs') / name) != expected:
            raise ValueError('STOP frozen protocol SHA changed')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_meta.json').exists():
        raise FileExistsError('Do not overwrite an existing X05 run')
    meta = {'git': git, 'command': sys.argv, 'started_at': now(), 'storage': storage,
            'protocol': spec(Path('experiments/X05_calib/protocol/X05_PROMPT.md')),
            'revision': spec(Path('experiments/X05_calib/protocol/X05_REVISION_1.md'))}
    write_json_atomic(output / 'run_meta.json', meta)
    return meta


def finish(output, meta, result):
    write_json_atomic(output / 'summary.json', result)
    meta.update(ended_at=now(), summary=spec(output / 'summary.json'))
    write_json_atomic(output / 'run_meta.json', meta)
