"""Install VAD-only extras under the new run root; never modify X02's venv."""
import os
import subprocess
import sys
from emotioncore.x06.common import ROOT, start, finish, spec


def run():
    output = ROOT / 'vad_setup'
    meta = start(output)
    deps = ROOT / 'deps'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               HF_HOME=str(ROOT / 'cache/hf'), MODELSCOPE_CACHE=str(ROOT / 'cache/modelscope'),
               XDG_CACHE_HOME=str(ROOT / 'cache'), PIP_CACHE_DIR=str(ROOT / 'cache/pip'))
    packages = ['silero-vad', 'funasr', 'kaldiio', 'kaldi-native-fbank', 'hydra-core',
                'omegaconf', 'antlr4-python3-runtime==4.9.3', 'editdistance', 'jaconv',
                'jamo', 'torch-complex']
    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--target', str(deps),
               '--report', str(output / 'pip_report.json'), *packages]
    with (output / 'pip.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finish(output, meta, {'status': 'installed', 'dependencies': str(deps),
                         'command': command, 'pip_report': spec(output / 'pip_report.json')})


if __name__ == '__main__':
    run()
