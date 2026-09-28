"""Paired Qwen2.5-Omni vision tower features for whole and 448px face frames."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


ENCODER = "V3_qwen2.5_omni_vision_tower"
PREPROCESS_VERSION = "1"
EXCLUDED_SPEAKERS = frozenset({"02", "13", "20", "21"})
WHOLE_MAX_PIXELS = 50176
FACE_PIXELS = 448 * 448


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visual_shards(model_dir: Path) -> list[Path]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    names = {value for key, value in index["weight_map"].items()
             if key.startswith("thinker.visual.")}
    if not names:
        raise ValueError("No Qwen visual tower tensors in local model index")
    return [model_dir / name for name in sorted(names)]


def load_visual_tower(model_dir: Path):
    import torch
    from safetensors import safe_open
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniVisionEncoder

    config = Qwen2_5OmniConfig.from_pretrained(str(model_dir), local_files_only=True)
    tower = Qwen2_5OmniVisionEncoder(config.thinker_config.vision_config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    visual = {key: shard for key, shard in index.items()
              if key.startswith("thinker.visual.")}
    state = {}
    for shard in sorted(set(visual.values())):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as stream:
            for key, source in visual.items():
                if source == shard:
                    state[key.removeprefix("thinker.visual.")] = stream.get_tensor(key)
    tower.load_state_dict(state, strict=True)
    del state
    tower = tower.to(device="cuda:0", dtype=torch.bfloat16).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
    return tower, processor.image_processor


def worker_main(model_dir: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    tower, processor = load_visual_tower(model_dir)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            started = time.monotonic()
            paths_a = [Path(value) for value in task["whole_paths"]]
            paths_b = [Path(value) for value in task["crop_paths"]]
            if not paths_a or len(paths_a) != len(paths_b):
                raise ValueError("Whole/crop frame counts differ or are empty")
            for whole, crop in zip(paths_a, paths_b):
                if whole.stem.removeprefix("whole_") != crop.stem.removeprefix("crop_"):
                    raise ValueError("Whole/crop frame indices are not paired")
            results = {}
            for variant, paths, maximum in (("a", paths_a, WHOLE_MAX_PIXELS),
                                            ("b", paths_b, FACE_PIXELS)):
                features = []
                hashes = []
                grid_shapes = []
                elapsed = 0.0
                for path in paths:
                    hashes.append(sha256(path))
                    with Image.open(path) as source:
                        image = source.convert("RGB")
                        processed = processor(images=[image],
                                              size={"shortest_edge": 3136 if variant == "a" else maximum,
                                                    "longest_edge": maximum},
                                              return_tensors="pt")
                    grid = processed["image_grid_thw"].to("cuda:0")
                    pixels = processed["pixel_values"].to("cuda:0", dtype=torch.bfloat16)
                    torch.cuda.synchronize()
                    frame_started = time.monotonic()
                    with torch.inference_mode():
                        patches = tower(pixels, grid_thw=grid).pooler_output
                        pooled = patches.float().mean(dim=0)
                    torch.cuda.synchronize()
                    elapsed += time.monotonic() - frame_started
                    features.append(pooled.cpu().numpy())
                    grid_shapes.append(processed["image_grid_thw"][0].tolist())
                frames = np.stack(features).astype(np.float32)
                if frames.shape != (len(paths), 3584) or not np.isfinite(frames).all():
                    raise ValueError(f"Unexpected Qwen vision frame shape: {frames.shape}")
                output = Path(task[f"output_{variant}"])
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f".{output.stem}.{os.getpid()}.npz")
                np.savez(temporary, frames=frames.astype(np.float16))
                os.replace(temporary, output)
                results[variant] = {"status": "ok", "feature_path": str(output),
                                    "feature_sha256": sha256(output),
                                    "frame_shape": list(frames.shape), "dtype": "float16",
                                    "source_frame_sha256": hashes, "grid_thw": grid_shapes,
                                    "model_seconds": elapsed}
            response = {"status": "ok", "variants": results,
                        "paired_frames": len(paths_a),
                        "elapsed_s": time.monotonic() - started}
        except Exception as exc:
            response = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    provenance = git_provenance(root, args.allow_dirty)
    manifest_hash = sha256(args.manifest)
    model_hash = sha256(args.model_dir / "model.safetensors.index.json")
    shard_hashes = {str(path): sha256(path) for path in visual_shards(args.model_dir)}
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if args.split == "cv" and row["speaker_id"] in EXCLUDED_SPEAKERS:
                continue
            if args.split == "test" and row["speaker_id"] not in EXCLUDED_SPEAKERS:
                continue
            rows.append(row)
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated utterance IDs in manifest")
    if args.split == "cv" and len(rows) != 16366 and args.limit is None:
        raise ValueError(f"Expected 16,366 CV utterances, got {len(rows)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    base = args.run_root / "features" / "V3"
    base.mkdir(parents=True, exist_ok=True)
    meta_path = base / f"run_meta_{args.split}.json"
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest": str(args.manifest), "manifest_sha256": manifest_hash,
            "face_receipts": str(args.face_receipts),
            "model": str(args.model_dir), "model_index_sha256": model_hash,
            "visual_weight_shards_sha256": shard_hashes,
            "encoder": ENCODER, "preprocess_version": PREPROCESS_VERSION,
            "whole_max_pixels": WHOLE_MAX_PIXELS, "face_pixels": FACE_PIXELS,
            "inference_dtype": "bfloat16", "split": args.split,
            "count": len(rows), "timeout_s": args.timeout}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "model_index_sha256", "visual_weight_shards_sha256",
                    "encoder", "preprocess_version", "split"):
            if old[key] != meta[key]:
                raise RuntimeError(f"Cannot resume V3 with changed {key}")
        if old["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume V3 from a different Git commit")
        meta["started_at"] = old["started_at"]
        meta["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(meta_path, meta)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    worker = PersistentJSONWorker([args.worker_python, "-u", "-m",
                                   "emotioncore.x02.visual_v3", "--worker",
                                   "--model-dir", str(args.model_dir)],
                                  cwd=root, env=env, stderr_path=base / "worker_errors.log")
    status = {"ok": 0, "failed": 0, "no_face": 0}
    try:
        for index, row in enumerate(rows, 1):
            face_path = args.face_receipts / f"{row['utt_id']}.json"
            wait_started = time.monotonic()
            while not face_path.is_file() and args.wait_face_seconds > 0:
                if time.monotonic() - wait_started >= args.wait_face_seconds:
                    break
                time.sleep(2)
            if not face_path.is_file():
                raise RuntimeError(f"Face preprocessing incomplete: {row['utt_id']}")
            face = json.loads(face_path.read_text(encoding="utf-8"))
            face_hash = sha256(face_path)
            if face["input_sha256"] != row["mp4"]["sha256"]:
                raise ValueError(f"Face receipt input hash mismatch: {row['utt_id']}")
            paths = {variant: args.run_root / "features" / f"V3{variant}" / "receipts" /
                     f"{row['utt_id']}.json" for variant in ("a", "b")}
            if all(path.is_file() for path in paths.values()):
                receipts = {variant: json.loads(path.read_text(encoding="utf-8"))
                            for variant, path in paths.items()}
                for receipt in receipts.values():
                    if (receipt["input_sha256"] != row["mp4"]["sha256"] or
                            receipt["face_receipt_sha256"] != face_hash or
                            receipt["model_index_sha256"] != model_hash):
                        raise RuntimeError(f"V3 provenance changed: {row['utt_id']}")
            else:
                if any(path.exists() for path in paths.values()):
                    raise RuntimeError(f"Incomplete paired V3 receipts: {row['utt_id']}")
                if face["status"] == "no_face":
                    result = {"status": "no_face"}
                elif face["status"] != "ok":
                    result = {"status": "failed", "error": face.get("error", "Face failure")}
                else:
                    result = worker.request({"whole_paths": face["whole_paths"],
                        "crop_paths": face["crop_paths"],
                        "output_a": str(args.run_root / "features" / "V3a" / "data" /
                                        f"{row['utt_id']}.npz"),
                        "output_b": str(args.run_root / "features" / "V3b" / "data" /
                                        f"{row['utt_id']}.npz")}, args.timeout)
                receipts = {}
                for variant, path in paths.items():
                    variant_result = result.get("variants", {}).get(variant,
                                      {"status": result["status"],
                                       "error": result.get("error")})
                    receipt = {"id": row["utt_id"], "speaker_id": row["speaker_id"],
                               "official_split": row["official_split"],
                               "encoder": ENCODER, "variant": variant,
                               "model_index_sha256": model_hash,
                               "preprocess_version": PREPROCESS_VERSION,
                               "input_sha256": row["mp4"]["sha256"],
                               "face_receipt_sha256": face_hash, **variant_result}
                    write_json_atomic(path, receipt)
                    receipts[variant] = receipt
            if receipts["a"]["status"] != receipts["b"]["status"]:
                raise ValueError(f"Paired V3 status mismatch: {row['utt_id']}")
            status[receipts["a"]["status"]] += 1
            if index % 100 == 0 or index == len(rows):
                write_json_atomic(base / f"progress_{args.split}.json",
                                  {"done": index, "total": len(rows), "status": status})
    finally:
        worker.close()
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["status"] = status
    write_json_atomic(meta_path, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--face-receipts", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--split", choices=("cv", "test"), default="cv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--wait-face-seconds", type=float, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.model_dir)
    else:
        for name in ("manifest", "face_receipts", "run_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
