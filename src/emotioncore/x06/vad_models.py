"""Fetch only the two registered pretrained VAD candidates, with file hashes."""
from pathlib import Path
from emotioncore.x06.common import ROOT, start, finish, spec


def run():
    output = ROOT / 'vad_models'
    meta = start(output)
    from modelscope import snapshot_download
    path = Path(snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                                 cache_dir=str(ROOT / 'cache/modelscope')))
    silero = ROOT / 'deps/silero_vad/data/silero_vad.jit'
    assert silero.is_file()
    finish(output, meta, {'fsmn': str(path), 'silero': str(silero),
                          'artifacts': [spec(p) for p in sorted(path.rglob('*')) if p.is_file()] + [spec(silero)],
                          'training': False, 'device': 'cpu'})


if __name__ == '__main__':
    run()
