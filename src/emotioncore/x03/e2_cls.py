"""Freeze leakage-free T1 base [CLS] vectors from X02 token caches."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


def extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel

    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    token_meta = json.loads(args.tokens_meta.read_text(encoding="utf-8"))
    if sha256(args.tokens) != token_meta["tokens_sha256"]:
        raise ValueError("T1 token cache SHA differs from X02 receipt")
    if sha256(args.model_dir / "pytorch_model.bin") != token_meta["model_sha256"]:
        raise ValueError("T1 base checkpoint SHA differs from X02 receipt")
    with np.load(args.tokens) as archive:
        tokens = {key: archive[key] for key in archive.files}
    ids = tokens["input_ids"]
    masks = tokens["attention_mask"]
    failed = tokens["failed"]
    utt_ids = [str(uid) for uid in tokens["utt_ids"]]
    if ids.shape != masks.shape or ids.ndim != 2 or len(set(utt_ids)) != len(utt_ids):
        raise ValueError("T1 token cache shape or IDs invalid")
    if len(failed) != len(utt_ids):
        raise ValueError("T1 token failure flags invalid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"T1_{token_meta['source']}_{token_meta['split']}_cls.npy"
    if output.exists():
        raise FileExistsError(f"Frozen [CLS] cache already exists: {output}")
    temporary = output.with_name("." + output.name + ".pending.npy")
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).to("cpu")
    model.eval()
    width = int(model.config.hidden_size)
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                          shape=(len(ids), width))
    features[:] = 0
    try:
        with torch.inference_mode():
            for offset in range(0, len(ids), args.batch_size):
                stop = min(offset + args.batch_size, len(ids))
                good = np.flatnonzero(failed[offset:stop] == 0) + offset
                if len(good):
                    maximum = int(masks[good].sum(axis=1).max())
                    batch_ids = torch.from_numpy(ids[good, :maximum].astype(np.int64))
                    batch_mask = torch.from_numpy(masks[good, :maximum].astype(np.int64))
                    cls = model(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state[:, 0]
                    values = cls.float().numpy()
                    if values.shape != (len(good), width) or not np.isfinite(values).all():
                        raise ValueError("Nonfinite or malformed T1 [CLS] vector")
                    features[good] = values.astype(np.float16)
                if stop % 512 < args.batch_size or stop == len(ids):
                    features.flush()
                    write_json_atomic(args.output_dir / "cls_progress.json",
                                      {"done": stop, "total": len(ids)})
        features.flush()
        del features
        temporary.replace(output)
    except BaseException:
        del features
        temporary.unlink(missing_ok=True)
        raise
    meta = {"git": provenance, "command": sys.argv,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - started,
            "source": token_meta["source"], "split": token_meta["split"],
            "token_meta_sha256": sha256(args.tokens_meta),
            "tokens_sha256": token_meta["tokens_sha256"],
            "model_sha256": token_meta["model_sha256"],
            "checkpoint_policy": "X02 unfine-tuned T1 base; final_model.pt is supervised on all CV speakers and prohibited for outer-fold feature generation",
            "pooling": "last_hidden_state[:,0]", "dtype": "float16",
            "sample_count": len(ids), "failed_count": int(np.count_nonzero(failed)),
            "utt_ids_sha256": __import__("hashlib").sha256("\n".join(utt_ids).encode()).hexdigest(),
            "feature_path": str(output), "feature_sha256": sha256(output),
            "shape": [len(ids), width], "batch_size": args.batch_size,
            "threads": args.threads}
    write_json_atomic(args.output_dir / "cls_run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tokens-meta", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size and threads must be positive")
    extract(args)


if __name__ == "__main__":
    main()
