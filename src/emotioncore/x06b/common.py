"""Immutable X06b authority and isolated H-drive provenance."""
from pathlib import Path
import json, datetime, hashlib, sys, importlib.metadata
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.prepare import storage_gate, require_h
from emotioncore.x06.common import read, rows, sha256, spec, now, FMH, TEST, MEDIA, MANIFEST

ROOT = Path('/mnt/h/emotionAI/emotioncore_runs/X06b')
OLD = Path('/mnt/h/emotionAI/emotioncore_runs/X06a')
REG = Path('configs/x06b_registration.json')
LIMIT = 14 * 3600


def authority():
    registered = read(REG)
    for path, digest in registered['frozen_sha256'].items():
        if sha256(path) != digest:
            raise RuntimeError('STOP frozen source changed: ' + path)
    assert read('configs/x06_silence_freeze.json')['min_silence'] == .4
    assert read('configs/x06_smoothing_freeze.json')['alpha'] == 0
    return registered


def start(name):
    output = ROOT / name
    require_h(output); storage = storage_gate(); registered = authority()
    git = git_provenance(Path.cwd())
    if git['dirty']: raise RuntimeError('STOP dirty workspace')
    output.mkdir(parents=True, exist_ok=False)
    meta = {'started_at': now(), 'git': git, 'command': sys.argv, 'storage': storage,
            'registration': spec(REG), 'frozen_sha256': registered['frozen_sha256'],
            'python': sys.version, 'training': False}
    meta['packages']={}
    for name in ('numpy','torch','transformers','opencv-python','opencv-python-headless','timm','funasr','soundfile'):
        try:meta['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    write_json_atomic(output / 'run_meta.json', meta)
    return output, meta


def finish(output, meta, summary):
    write_json_atomic(output / 'summary.json', summary)
    meta.update(ended_at=now(), summary=spec(output / 'summary.json'))
    write_json_atomic(output / 'run_meta.json', meta)


def checked(receipt):
    path = Path(receipt['path'])
    if sha256(path) != receipt['sha256']: raise RuntimeError('STOP source SHA changed: '+str(path))
    return path


def no_stop():
    if list((ROOT/'stops').glob('*.json')):
        raise RuntimeError('STOP newer receipt present; no further experiments')
    if (ROOT/'STOP.json').exists():
        from emotioncore.x06b.recovery import validated_authorization
        validated_authorization()


def stop(stage, reason, evidence):
    path=ROOT/'STOP.json'
    if path.exists():
        import uuid
        directory=ROOT/'stops';directory.mkdir(exist_ok=True)
        path=directory/(uuid.uuid4().hex+'.json')
    write_json_atomic(path, {'status':'stopped','stage':stage,'reason':reason,'at':now(),'evidence':evidence})


def source_timeline(speaker):
    if speaker not in FMH | TEST: raise RuntimeError('STOP unregistered population')
    receipt = read('experiments/X06a_stream/results/streams.json')
    checked(receipt['source'])
    return read(checked(receipt['data']['streams'][speaker]['timeline']))
