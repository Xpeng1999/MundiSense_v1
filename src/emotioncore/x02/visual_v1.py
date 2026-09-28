"""Cache EmotiEffLib face-frame features and native eight-class logits."""
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


ENCODER = "V1_enet_b0_8_best_afew"
PREPROCESS_VERSION = "1"
EXCLUDED_SPEAKERS = frozenset({"02", "13", "20", "21"})
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_main(model_path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    # This is the locally pinned EmotiEffLib model artifact, never remote code.
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    classifier = model.classifier[0]
    weights = classifier.weight.detach().clone().to("cuda:0")
    bias = classifier.bias.detach().clone().to("cuda:0")
    model.classifier = torch.nn.Identity()
    model = model.eval().to("cuda:0")
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            started = time.monotonic()
            paths = [Path(name) for name in task["crops"]]
            if not paths:
                raise ValueError("No face crops provided")
            hashes = []
            images = []
            for path in paths:
                hashes.append(sha256(path))
                with Image.open(path) as source:
                    image = np.asarray(source.convert("RGB").resize((SIZE, SIZE)),
                                       dtype=np.float32) / 255.0
                images.append((torch.from_numpy(image).permute(2, 0, 1) - mean) / std)
            batch = torch.stack(images).to("cuda:0")
            with torch.inference_mode():
                features = model(batch)
                logits = torch.nn.functional.linear(features, weights, bias)
            features = features.float().cpu().numpy()
            logits = logits.float().cpu().numpy()
            if (features.ndim != 2 or features.shape[1] != 1280 or
                    logits.shape != (len(paths), 8) or
                    not np.isfinite(features).all() or not np.isfinite(logits).all()):
                raise ValueError(f"Unexpected V1 shapes: {features.shape}, {logits.shape}")
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.npz")
            np.savez(temporary, frames=features, native_logits=logits)
            os.replace(temporary, output)
            response = {"status": "ok", "feature_path": str(output),
                        "feature_sha256": sha256(output),
                        "crop_sha256": hashes, "frame_shape": list(features.shape),
                        "native_logits_shape": list(logits.shape),
                        "model_seconds": time.monotonic() - started}
        except Exception as exc:
            response = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    provenance = git_provenance(root, args.allow_dirty)
    manifest_hash = sha256(args.manifest)
    model_hash = sha256(args.model_path)
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
    base = args.run_root / "features" / "V1"
    receipts_dir = base / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base / f"run_meta_{args.split}.json"
    meta = {"git": provenance, "command": sys.argv, "started_at":
            dt.datetime.now(dt.timezone.utc).isoformat(), "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash, "face_receipts": str(args.face_receipts),
            "model": str(args.model_path), "model_sha256": model_hash,
            "encoder": ENCODER, "preprocess_version": PREPROCESS_VERSION,
            "size": SIZE, "mean": MEAN, "std": STD,
            "split": args.split, "count": len(rows), "timeout_s": args.timeout}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "model_sha256", "encoder",
                    "preprocess_version", "split"):
            if old[key] != meta[key]:
                raise RuntimeError(f"Cannot resume V1 with changed {key}")
        if old["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume V1 from a different Git commit")
        meta["started_at"] = old["started_at"]
        meta["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(meta_path, meta)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    worker = PersistentJSONWorker([args.worker_python, "-u", "-m",
                                   "emotioncore.x02.visual_v1", "--worker",
                                   "--model-path", str(args.model_path)],
                                  cwd=root, env=env, stderr_path=base / "worker_errors.log")
    status = {"ok": 0, "failed": 0, "no_face": 0}
    try:
        for index, row in enumerate(rows, 1):
            path = receipts_dir / f"{row['utt_id']}.json"
            face_path = args.face_receipts / f"{row['utt_id']}.json"
            if not face_path.is_file():
                raise RuntimeError(f"Face preprocessing incomplete: {row['utt_id']}")
            face = json.loads(face_path.read_text(encoding="utf-8"))
            face_hash = sha256(face_path)
            if face["input_sha256"] != row["mp4"]["sha256"]:
                raise ValueError(f"Face receipt input hash mismatch: {row['utt_id']}")
            if path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if (receipt["input_sha256"] != row["mp4"]["sha256"] or
                        receipt["face_receipt_sha256"] != face_hash or
                        receipt["model_sha256"] != model_hash):
                    raise RuntimeError(f"V1 provenance changed: {row['utt_id']}")
            else:
                if face["status"] == "no_face":
                    result = {"status": "no_face"}
                elif face["status"] != "ok":
                    result = {"status": "failed", "error": face.get("error", "Face failure")}
                else:
                    result = worker.request({"crops": face["crop_paths"],
                                  "output": str(base / "data" / f"{row['utt_id']}.npz")},
                                  args.timeout)
                receipt = {"id": row["utt_id"], "speaker_id": row["speaker_id"],
                           "official_split": row["official_split"],
                           "encoder": ENCODER, "model_sha256": model_hash,
                           "preprocess_version": PREPROCESS_VERSION,
                           "input_sha256": row["mp4"]["sha256"],
                           "face_receipt_sha256": face_hash, **result}
                write_json_atomic(path, receipt)
            status[receipt["status"]] += 1
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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--split", choices=("cv", "test"), default="cv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.model_path)
    else:
        for name in ("manifest", "face_receipts", "run_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
