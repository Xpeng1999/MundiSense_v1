"""Recover the pre-inference missing-timm error without changing frozen inputs."""
import shutil
from pathlib import Path
from emotioncore.x06.common import *


def run():
    output = ROOT / 'dependency_repair'
    meta = start(output)
    failure = ROOT / 'gpu_jobs/equivalence/summary.json'
    log = failure.parent / 'process.log'
    assert read(failure)['problem'] == 'child_exit_1'
    assert log.read_text().rstrip().endswith("ModuleNotFoundError: No module named 'timm'")
    assert not (ROOT / 'equivalence/predictions.jsonl').exists()
    source = Path('/home/administrator/emotion-data/open_resources_v1/envs/visual/lib/python3.11/site-packages')
    inventory = []
    for name in ('timm', 'timm-0.9.16.dist-info'):
        destination = ROOT / 'deps' / name
        shutil.copytree(source / name, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                original = source / name / path.relative_to(destination)
                assert sha256(original) == sha256(path)
                inventory.append({'original': spec(original), 'copy': spec(path)})
    import torch
    model = torch.load('/home/administrator/emotion-data/open_resources_v1/models/emotiefflib/enet_b0_8_best_afew.pt',
                       map_location='cpu', weights_only=False)
    assert tuple(model.classifier[0].weight.shape) == (8, 1280)
    write_json_atomic(output / 'files.json', inventory)
    finish(output, meta, {'status': 'repaired_before_inference', 'failed_job': spec(failure),
        'failed_log': spec(log), 'cpu_load_passed': True, 'inventory': spec(output / 'files.json'),
        'charged_gpu_s_retained': read(failure)['charged_gpu_s'], 'frozen_environment_modified': False})


if __name__ == '__main__':
    run()
