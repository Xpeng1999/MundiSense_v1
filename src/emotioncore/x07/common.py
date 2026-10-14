"""Fail-closed, committed authority and append-only X07 receipts."""
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from emotioncore.x07.inventory import sha, spec, ids_sha

ROOT = Path('/mnt/h/emotionAI/emotioncore_runs/X07')
PROTOCOL = Path('configs/x07_lockbox_protocol.json')
SYSTEMS = ('H3', 'H_A', 'H_V')

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL preserves failed/previous attempts rather than silently replacing them.
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

def atomic(path, obj):
    from emotioncore.x02.runtime import write_json_atomic
    write_json_atomic(Path(path), obj)

def checked(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise RuntimeError('STOP SHA mismatch: ' + str(path))
    return path

def git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()

def committed_protocol(path=PROTOCOL, cwd=None):
    path = Path(path)
    repo = Path(cwd or git('rev-parse','--show-toplevel'))
    rel = path.as_posix() if not path.is_absolute() else path.relative_to(repo).as_posix()
    actual = (repo / rel).read_bytes()
    blob = subprocess.check_output(['git','show','HEAD:'+rel], cwd=repo)
    if blob != actual:
        raise RuntimeError('STOP protocol is not committed')
    commits = git('log','--format=%H','--',rel,cwd=repo).splitlines()
    if len(commits) != 1:
        raise RuntimeError('STOP protocol must be introduced exactly once and never modified')
    protocol = json.loads(actual)
    subprocess.run(['git','merge-base','--is-ancestor',protocol['code_commit'],commits[0]],cwd=repo,check=True)
    timestamp = int(git('show','-s','--format=%ct',commits[0],cwd=repo))
    return protocol, {'commit':commits[0], 'commit_unix':timestamp, 'sha256':sha(repo/rel)}

def verify_precedes(receipt, started_at):
    if dt.datetime.fromisoformat(started_at).timestamp() <= receipt['commit_unix']:
        raise RuntimeError('STOP protocol was not committed before inference')

def no_stop(root=ROOT):
    if (root / 'STOP.json').exists():
        raise RuntimeError('STOP receipt present; reviewer action required')

def stop(stage, reason, evidence=None, root=ROOT):
    if not (root/'STOP.json').exists():
        write_new(root/'STOP.json', {'stage':stage,'reason':str(reason),'at':now(),'evidence':evidence})

def authority(assets=False):
    no_stop()
    from emotioncore.x04.prepare import storage_gate
    storage_gate()
    protocol, receipt = committed_protocol()
    if git('status','--porcelain','--untracked-files=normal'):
        raise RuntimeError('STOP dirty execution checkout')
    for path, digest in protocol['code_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('STOP frozen code changed: ' + path)
    for item in protocol['configs'].values():
        checked(item)
    if assets:
        for group in protocol['inventory']['assets'].values():
            for item in group:
                checked(item)
        checked(protocol['inventory']['meta_csv'])
    return protocol, receipt

def stage_gate(split, root=ROOT):
    no_stop(root)
    leak = read(root/'leakage.json')
    if leak['status'] != 'passed' or leak['protocol_sha256'] != sha(PROTOCOL):
        raise RuntimeError('STOP L2 leakage gate not satisfied')
    if split == 'test':
        metric = read(root/'valid/metrics.json')
        if metric['protocol_sha256'] != sha(PROTOCOL):
            raise RuntimeError('STOP development receipt uses different protocol')
        value = metric['systems']['H3']['pearson_available']
        if value is None or value < .78:
            raise RuntimeError('STOP L3 Pearson below 0.78; lockbox remains closed')

def load_completed(directory, expected_ids):
    directory = Path(directory)
    if not (directory/'complete.json').exists():
        raise RuntimeError('Inference must complete before any metrics')
    done = read(directory/'complete.json')
    if done['ids_sha256'] != ids_sha(expected_ids) or done['n'] != len(expected_ids):
        raise RuntimeError('STOP incomplete inference population')
    if set(done['predictions']) != set(expected_ids):
        raise RuntimeError('STOP completion IDs mismatch')
    result=[]
    for uid in expected_ids:
        row = read(checked(done['predictions'][uid]))
        if row['id'] != uid:
            raise RuntimeError('STOP prediction identity mismatch')
        result.append(row)
    return done,result

def claim_metrics(directory):
    write_new(Path(directory)/'metrics.started.json', {'at':now(),'policy':'one attempt, never rerun or overwrite'})
