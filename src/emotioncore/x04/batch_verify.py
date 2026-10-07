"""Exact batch-fetch verification against the frozen X02 ET A3 mmap."""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import json
import numpy as np
from emotioncore.x02 import head_h2
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.face import sha256
from emotioncore.x04.heads import Frames
from emotioncore.x04.prepare import storage_gate


def run(args):
    import torch
    git = git_provenance(Path.cwd()); storage_gate()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    meta = json.loads((args.pack / 'pack_meta.json').read_text())
    frames = np.load(meta['frame_file'], mmap_mode='r')
    lengths = np.load(meta['lengths_file'])
    indices = np.random.default_rng(20260928).permutation(len(frames))[:128]
    data = Frames([frames], [lengths])
    x, n = data.batch(indices, 'cuda:0')
    old_x, old_n = head_h2.tensor_batch(frames, lengths, indices, 'cuda:0')
    equal_x = torch.equal(x, old_x); equal_n = torch.equal(n, old_n)
    receipt = {'git':git,'started_at':started,'ended_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'pack_meta_sha256':sha256(args.pack/'pack_meta.json'),'indices':indices.tolist(),
        'seed':20260928,'device':'cuda:0','features_torch_equal':equal_x,'lengths_torch_equal':equal_n,
        'source_dtype':str(frames.dtype),'source_pinned':any(t.is_pinned() for t in data.source_tensors),
        'pinned_buffer_bytes':data.pinned_buffer.numel()*data.pinned_buffer.element_size(),
        'destination_dtype':str(x.dtype),'passed':equal_x and equal_n and not any(t.is_pinned() for t in data.source_tensors)}
    write_json_atomic(args.output,receipt)
    if not receipt['passed']:raise RuntimeError('STOP GPU batch fetch differs from X02')


def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X02/packs/A3'))
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())


if __name__=='__main__':main()
