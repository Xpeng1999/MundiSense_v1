"""Invoke unchanged X02 worker code for new, explicitly registered corpora."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from emotioncore.x02 import audio_a3, face, visual_v1, visual_v2
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic
from emotioncore.x04.prepare import require_h, storage_gate

R2 = Path("/home/administrator/emotion-data/emotioncore_runs/X02")
VISUAL_PYTHON = "/home/administrator/emotion-data/open_resources_v1/envs/visual/bin/python"
MODELS = {"A3": R2 / "models/A3", "V2": R2 / "models/V2",
          "V1": Path("/home/administrator/emotion-data/open_resources_v1/models/emotiefflib/enet_b0_8_best_afew.pt")}
YUNET = Path("/home/administrator/projects/emotion/artifacts/perception_setup/weights/face_detection_yunet_2023mar.onnx")


def run(args):
    provenance = git_provenance(Path.cwd())
    require_h(args.run_root); storage_gate()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if any(r["corpus"] not in ("MER", "CREMA") for r in rows):
        raise ValueError("No CH-SIMS or unrelated corpus permitted in X04 extraction")
    if len({r["utt_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated new-corpus IDs")
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "face":
        started = time.monotonic()
        face.run(args.manifest, Path("/"), YUNET, args.run_root, 4, 120, False)
        write_json_atomic(args.run_root / "face/timing.json", {"cpu_wall_seconds": time.monotonic()-started,
            "gpu_seconds": 0, "manifest_sha256": face.sha256(args.manifest), "git": provenance})
        return
    encoder = args.stage
    module = {"A3": audio_a3, "V2": visual_v2, "V1": visual_v1}[encoder]
    model = MODELS[encoder]
    model_file = model / ("pytorch_model.bin" if encoder == "A3" else "model.safetensors") if model.is_dir() else model
    model_sha = face.sha256(model_file)
    base = args.run_root / "features" / encoder
    receipt_root = base / "receipts"; receipt_root.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "encoder": module.ENCODER,
        "revision": getattr(module, "REVISION", "local_artifact:" + model_sha),
        "model_sha256": model_sha, "preprocess_version": module.PREPROCESS_VERSION,
        "frozen_worker_path": str(Path(module.__file__)), "frozen_worker_sha256": face.sha256(Path(module.__file__)),
        "manifest": str(args.manifest), "manifest_sha256": face.sha256(args.manifest),
        "n": len(rows), "timeout_s": 120, "storage": storage_gate()}
    write_json_atomic(base / "run_meta.json", meta)
    command = [VISUAL_PYTHON if encoder == "V1" else sys.executable, "-u", "-m", module.__name__, "--worker",
               "--model-dir" if encoder == "A3" else "--model-path", str(model)]
    env = os.environ.copy(); env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker = PersistentJSONWorker(command, cwd=Path.cwd(), env=env, stderr_path=base / "worker_errors.log")
    started = time.monotonic(); times = []; counts = collections.Counter()
    try:
        for i, row in enumerate(rows, 1):
            if i % 50 == 1:
                storage_gate()
            receipt = receipt_root / (row["utt_id"] + ".json")
            source = row["wav" if encoder == "A3" else "mp4"]
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record["input_sha256"] != source["sha256"] or record["model_sha256"] != model_sha:
                    raise ValueError("Changed input/model during feature resume")
                if record.get("feature_path") and face.sha256(Path(record["feature_path"])) != record["feature_sha256"]:
                    raise ValueError("Changed feature during resume")
            else:
                common = {"output": str(base / "data" / (row["utt_id"] + ".npz"))}
                if source["status"] != "ok":
                    result = {"status": "failed", "error": "Source media probe failed"}
                elif encoder == "A3":
                    result = worker.request({**common, "wav": source["path"], "input_sha256": source["sha256"]}, 120)
                else:
                    face_path = args.run_root / "face/receipts" / (row["utt_id"] + ".json")
                    face_receipt = json.loads(face_path.read_text())
                    if face_receipt["input_sha256"] != source["sha256"]:
                        raise ValueError("Face source SHA mismatch")
                    if face_receipt["status"] != "ok":
                        result = {"status": "no_face" if face_receipt["status"] == "no_face" else "failed",
                                  "error": face_receipt.get("error")}
                    else:
                        result = worker.request({**common, "crops": face_receipt["crop_paths"]}, 120)
                    result["face_receipt_sha256"] = face.sha256(face_path)
                record = {"id": row["utt_id"], "speaker_id": row["speaker_id"], "official_split": row["official_split"],
                    "corpus": row["corpus"], "encoder": module.ENCODER, "revision": meta["revision"],
                    "model_sha256": model_sha, "preprocess_version": module.PREPROCESS_VERSION,
                    "input_sha256": source["sha256"], "worker_sha256": meta["frozen_worker_sha256"], **result}
                write_json_atomic(receipt, record)
                times.append(record.get("elapsed_s", 0))
            counts[record["status"]] += 1
            if i % 50 == 0 or i == len(rows):
                write_json_atomic(base / "progress.json", {"done": i, "total": len(rows), "status": dict(counts)})
    finally:
        worker.close()
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), status=dict(counts),
        gpu_wall_seconds=time.monotonic()-started, request_seconds=times,
        gpu_timing_convention="worker lifetime wall time incl loading and request I/O; face stage CPU excluded")
    write_json_atomic(base / "run_meta.json", meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--stage", choices=["face", "A3", "V2", "V1"], required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
