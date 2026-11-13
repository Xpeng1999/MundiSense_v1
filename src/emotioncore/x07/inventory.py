"""L1 metadata-only asset inventory; does not open SIMS media or labels."""
import hashlib
import json
from pathlib import Path
import pandas as pd

RUN = Path('/mnt/h/emotionAI/emotioncore_runs')
X02 = Path('/home/administrator/emotion-data/emotioncore_runs/X02')
META = Path('/home/administrator/emotion-data/emotionAI_dataset/CH-SIMSV2/meta.csv')
DEDUP = Path('/home/administrator/emotion-data/emotioncore_runs/X04a/preflight_v3/dedup/dedup_report.json')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def spec(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}

def ids_sha(ids):
    return hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()

def collect():
    assets = {}
    roots = {
        'Q_base': Path('/home/administrator/projects/emotion/omni_models/Qwen2.5-Omni-7B'),
        'Q_adapter': RUN / 'X04b/Q/final/best_adapter_eval8',
        'A3_encoder': X02 / 'models/A3', 'V2_encoder': X02 / 'models/V2',
    }
    for name, root in roots.items():
        assets[name] = [spec(p) for p in sorted(root.iterdir()) if p.is_file()]
    assets['V1_encoder'] = [spec('/home/administrator/emotion-data/open_resources_v1/models/emotiefflib/enet_b0_8_best_afew.pt')]
    assets['YuNet'] = [spec('/home/administrator/projects/emotion/artifacts/perception_setup/weights/face_detection_yunet_2023mar.onnx')]
    for name in ('A3', 'V2', 'V1'):
        assets[name + '_head'] = [spec(RUN / f'X05/heads/D1_fmhx/{name}/final_head.pt')]
    vad = RUN / 'X06a/vad_models/summary.json'
    v = json.loads(vad.read_text())
    assets['FSMN'] = [spec(p['path']) for p in v['artifacts'] if p['path'].startswith(v['fsmn'])]
    assets['vad_index'] = [spec(vad)]
    sources = {f'Q_epoch{e}': spec(RUN / f'X04b/manifests/final_epoch{e}.jsonl') for e in range(2)}
    sources.update({f'D1_{n}': spec(RUN / f'X05/heads/D1_fmhx/{n}/training_ids.jsonl') for n in ('A3','V2','V1')})
    sources['calibration'] = spec(RUN / 'X05/deployment/fitting_ids.jsonl')
    sources['MER_dedup'] = spec(RUN / 'X04a/manifests/mer_train_dedup.jsonl')
    df = pd.read_csv(META, usecols=['video_id','clip_id','mode'], dtype=str, keep_default_na=False)
    splits = {}
    for split, n in [('valid',647),('test',1034)]:
        selected = df[df['mode'] == split]
        ids = sorted(f'{v}__{c}' for v,c in zip(selected.video_id,selected.clip_id))
        if len(ids) != n or len(set(ids)) != n:
            raise RuntimeError('STOP split identity/count mismatch')
        splits[split] = {'ids': ids, 'count': n, 'ids_sha256': ids_sha(ids)}
    return {'assets': assets, 'leakage_sources': sources, 'dedup_report': spec(DEDUP),
            'meta_csv': spec(META), 'splits': splits, 'labels_read': False,
            'media_opened': False, 'id_hash_encoding': 'sorted IDs joined with LF plus trailing LF, UTF-8'}

if __name__ == '__main__':
    import sys
    result = collect()
    with Path(sys.argv[1]).open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
