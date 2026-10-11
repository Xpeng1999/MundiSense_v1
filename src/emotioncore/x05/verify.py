"""CPU-only required tests with clean-worktree and machine-readable provenance."""
import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from emotioncore.x05.common import ROOT,start,finish,spec


def run(name):
    output=ROOT/name;meta=start(output)
    command=[sys.executable,'-m','pytest','-q','tests/test_x05_heads.py','tests/test_x05_methods.py',
        'tests/test_x05_bundle.py','tests/test_x04_heads.py','tests/test_x04_targets.py',
        'tests/test_x03_fusion.py','tests/test_metrics.py',f'--junitxml={output}/pytest.xml']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
    started=time.monotonic()
    result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/'pytest.txt').write_text(result.stdout)
    suites=list(ET.parse(output/'pytest.xml').getroot())
    totals={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    receipt={'command':command,'returncode':result.returncode,'totals':totals,
        'elapsed_s':time.monotonic()-started,'cuda_visible_devices':'','xml':spec(output/'pytest.xml'),
        'stdout':spec(output/'pytest.txt')}
    finish(output,meta,receipt);print(result.stdout)
    if result.returncode:raise RuntimeError('X05 validation failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=('validation_predeploy','final_tests'),required=True)
    run(p.parse_args().name)
