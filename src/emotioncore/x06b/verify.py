import subprocess,sys,os,argparse
from emotioncore.x06b.common import *
def run(name):
    out,meta=start(name)
    command=[sys.executable,'-m','pytest','-q','tests/test_x06b_protocol.py','tests/test_x06_protocol.py']
    with (out/'pytest.txt').open('w') as log: result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    finish(out,meta,{'returncode':result.returncode,'stdout':spec(out/'pytest.txt'),'gpu_seconds':0})
    if result.returncode:raise RuntimeError('CPU tests failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='unit_tests_startup');run(p.parse_args().name)
