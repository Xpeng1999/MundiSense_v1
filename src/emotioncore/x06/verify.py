"""CPU-only checks with the same formal provenance requirements."""
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from emotioncore.x06.common import ROOT, start, finish, spec


def run(name):
    output = ROOT / name
    meta = start(output)
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_x06_protocol.py',
               '--junitxml=' + str(output / 'pytest.xml')]
    with (output / 'pytest.txt').open('w') as log:
        result = subprocess.run(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                stdout=log, stderr=subprocess.STDOUT)
    totals = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
    if (output / 'pytest.xml').exists():
        for suite in ET.parse(output / 'pytest.xml').getroot().iter('testsuite'):
            for key in totals:
                totals[key] += int(suite.get(key, 0))
    finish(output, meta, {'command': command, 'returncode': result.returncode, 'totals': totals,
                          'stdout': spec(output / 'pytest.txt')})
    if result.returncode:
        raise RuntimeError('X06 protocol tests failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--name', default='unit_tests')
    run(parser.parse_args().name)
