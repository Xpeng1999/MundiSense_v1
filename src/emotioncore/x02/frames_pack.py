"""Pack cached frame features for repeated nested-CV attention-head training."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from emotioncore.x02.head_h1 import load_rows, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


SPECS = {"A1": (128, 1024), "A3": (128, 1024),
         "V1": (8, 1288), "V2": (8, 768)}


def read_feature(path: Path, encoder: str) -> np.ndarray:
    with np.load(path) as data:
        if encoder == "A1":
            frames = data["frames"]
        elif encoder == "A3":
            frames = data["layers"]
        elif encoder == "V1":
            frames = np.concatenate((data["frames"], data["native_logits"]), axis=-1)
        elif encoder == "V2":
            frames = data["frames"]
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
    return np.asarray(frames, dtype=np.float16)


def pack(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if sha256(args.manifest) != folds["source_manifest_sha256"]:
        raise ValueError("Manifest differs from frozen folds")
    rows = load_rows(args.manifest, folds)
    max_frames, dimension = SPECS[args.encoder]
    shape = ((len(rows), 4, max_frames, dimension) if args.encoder == "A3" else
             (len(rows), max_frames, dimension))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "frames.npy"
    if output.exists():
        raise FileExistsError("Packed feature file already exists; use a new output directory")
    temporary = output.with_name(".frames.pending.npy")
    packed = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    lengths = np.zeros(len(rows), dtype=np.int16)
    status = {"ok": 0, "failed": 0, "no_face": 0}
    sources = hashlib.sha256()
    try:
        for index, row in enumerate(rows):
            receipt_path = args.receipts_root / f"{row['utt_id']}.json"
            if not receipt_path.is_file():
                raise ValueError(f"Missing feature receipt: {row['utt_id']}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            expected = row["wav"]["sha256"] if args.encoder.startswith("A") else row["mp4"]["sha256"]
            if receipt["id"] != row["utt_id"] or receipt["input_sha256"] != expected:
                raise ValueError(f"Input provenance mismatch: {row['utt_id']}")
            state = receipt["status"]
            if state not in status:
                raise ValueError(f"Unexpected feature status: {state}")
            status[state] += 1
            sources.update(f"{row['utt_id']}:{state}:{receipt.get('feature_sha256', '')}\n".encode())
            if state == "ok":
                feature_path = Path(receipt["feature_path"])
                if sha256(feature_path) != receipt["feature_sha256"]:
                    raise ValueError(f"Feature sha256 mismatch: {row['utt_id']}")
                frames = read_feature(feature_path, args.encoder)
                count = frames.shape[-2]
                if count == 0 or frames.shape[-1] != dimension:
                    raise ValueError(f"Unexpected feature shape: {row['utt_id']}: {frames.shape}")
                if args.encoder == "A3" and frames.shape[0] != 4:
                    raise ValueError(f"Expected four A3 layers: {row['utt_id']}")
                if not np.isfinite(frames).all():
                    raise ValueError(f"Non-finite feature: {row['utt_id']}")
                indices = np.linspace(0, count - 1, min(count, max_frames)).round().astype(int)
                selected = frames[..., indices, :]
                packed[index, ..., :len(indices), :] = selected
                lengths[index] = len(indices)
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                write_json_atomic(args.output_dir / "progress.json",
                                  {"done": index + 1, "total": len(rows), "status": status})
        packed.flush()
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        del packed
    lengths_path = args.output_dir / "lengths.npy"
    np.save(lengths_path, lengths)
    meta = {"git": provenance, "command": sys.argv,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "encoder": args.encoder, "manifest_sha256": sha256(args.manifest),
            "folds_sha256": sha256(args.folds), "source_receipts_root": str(args.receipts_root),
            "source_receipt_aggregate_sha256": sources.hexdigest(),
            "frame_file": str(output), "frame_sha256": sha256(output),
            "lengths_file": str(lengths_path), "lengths_sha256": sha256(lengths_path),
            "shape": shape, "dtype": "float16", "max_frames": max_frames,
            "subsample": "evenly spaced including first and last frame",
            "status": status}
    write_json_atomic(args.output_dir / "pack_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--encoder", choices=tuple(SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    pack(parser.parse_args())


if __name__ == "__main__":
    main()
