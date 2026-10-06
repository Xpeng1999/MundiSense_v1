"""Generate eight-frame 336px face videos from frozen X02 face receipts."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

from emotioncore.x02.asr_cer import sha256
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


FRAME_COUNT = 8
SIZE = 336
MIN_FACE_PX_AT_448 = 64
MIN_CONFIDENCE = 0.8
PREPROCESS_VERSION = "x03_face336_v1"


def equal_positions(requested_frames: int) -> list[int]:
    if requested_frames <= 0:
        raise ValueError("requested_frames must be positive")
    return [round(i * (requested_frames - 1) / (FRAME_COUNT - 1))
            for i in range(FRAME_COUNT)]


def select_nearest(available: list[int], requested_frames: int) -> list[int]:
    if not available:
        raise ValueError("No cropped frames available")
    return [min(available, key=lambda candidate: (abs(candidate - desired), candidate))
            for desired in equal_positions(requested_frames)]


def worker_main(detector_path: Path) -> None:
    import cv2

    cv2.setNumThreads(1)
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (448, 448), 0.6, 0.3, 50)
    for line in sys.stdin:
        started = time.monotonic()
        try:
            task = json.loads(line)
            source = json.loads(Path(task["face_receipt"]).read_text(encoding="utf-8"))
            if (source["id"] != task["utt_id"] or source["status"] != "ok" or
                    source["input_sha256"] != task["input_sha256"]):
                raise ValueError("Frozen X02 face receipt mismatch")
            accepted = {}
            source_hashes = {}
            for value in source["crop_paths"]:
                path = Path(value)
                index = int(path.stem.split("_")[-1])
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != (448, 448):
                    raise ValueError(f"Unreadable X02 448px crop: {path}")
                detector.setInputSize((448, 448))
                _, faces = detector.detect(frame)
                if faces is None:
                    continue
                valid = [row for row in faces if row[2] >= MIN_FACE_PX_AT_448 and
                         row[3] >= MIN_FACE_PX_AT_448 and row[14] >= MIN_CONFIDENCE]
                if valid:
                    accepted[index] = cv2.resize(frame, (SIZE, SIZE),
                                                 interpolation=cv2.INTER_AREA)
                    source_hashes[index] = sha256(path)
            if not accepted:
                raise ValueError("No X02 crop passed the fixed face size/confidence filter")
            chosen = select_nearest(sorted(accepted), int(source["requested_frames"]))
            destination = Path(task["output"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.mp4")
            fps = FRAME_COUNT / float(task["duration_s"])
            if not (0 < fps < 240):
                raise ValueError(f"Invalid clip frame rate: {fps}")
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (SIZE, SIZE))
            if not writer.isOpened():
                raise RuntimeError("Could not open video writer")
            try:
                for index in chosen:
                    writer.write(accepted[index])
            finally:
                writer.release()
            capture = cv2.VideoCapture(str(temporary))
            try:
                decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                capture.release()
            if (decoded, width, height) != (FRAME_COUNT, SIZE, SIZE):
                raise ValueError(f"Encoded face video shape mismatch: {(decoded, width, height)}")
            os.replace(temporary, destination)
            result = {"status": "ok", "output": str(destination),
                      "output_sha256": sha256(destination), "encoded_frames": decoded,
                      "pixels_per_frame": SIZE * SIZE, "fps": fps,
                      "available_crops": len(source["crop_paths"]),
                      "accepted_crops": len(accepted),
                      "desired_frame_indices": equal_positions(source["requested_frames"]),
                      "selected_frame_indices": chosen,
                      "source_crop_sha256": {str(k): source_hashes[k] for k in sorted(set(chosen))},
                      "elapsed_s": time.monotonic() - started}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                      "elapsed_s": time.monotonic() - started}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    if args.split == "sims_valid":
        from emotioncore.x02.sims_prepare import validate_adapter
        validate_adapter(rows)
    elif len(rows) != 19_250 or len({r["utt_id"] for r in rows}) != len(rows):
        raise ValueError("X03 accepts only the frozen EmotionTalk manifest")
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "run_meta.json"
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest),
            "yunet_sha256": sha256(args.detector),
            "face_summary_sha256": sha256(args.face_root / "summary.json"),
            "preprocess_version": PREPROCESS_VERSION,
            "size": SIZE, "frames": FRAME_COUNT,
            "min_face_px_at_448": MIN_FACE_PX_AT_448,
            "min_confidence": MIN_CONFIDENCE, "split": args.split, "count": len(rows),
            "workers": args.workers, "timeout_s": args.timeout}
    write_json_atomic(meta_path, meta)
    local = threading.local()
    workers = []
    lock = threading.Lock()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def process(row: dict) -> dict:
        output = args.output_dir / "videos" / f"{row['utt_id']}.mp4"
        receipt_path = args.output_dir / "receipts" / f"{row['utt_id']}.json"
        face_path = args.face_root / "receipts" / f"{row['utt_id']}.json"
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (prior["face_receipt_sha256"] != sha256(face_path) or
                    prior["input_sha256"] != row["mp4"]["sha256"] or
                    prior["preprocess_version"] != PREPROCESS_VERSION):
                raise ValueError(f"Cached face video receipt differs: {row['utt_id']}")
            if prior["status"] == "ok" and sha256(output) != prior["output_sha256"]:
                raise ValueError(f"Cached face video SHA differs: {row['utt_id']}")
            return prior
        if not hasattr(local, "worker"):
            local.worker = PersistentJSONWorker(
                [sys.executable, "-u", "-m", "emotioncore.x03.media", "--worker",
                 "--detector", str(args.detector)], cwd=Path.cwd(), env=env,
                stderr_path=args.output_dir / "worker_errors.log")
            with lock:
                workers.append(local.worker)
        result = local.worker.request({"utt_id": row["utt_id"],
                                       "face_receipt": str(face_path),
                                       "input_sha256": row["mp4"]["sha256"],
                                       "duration_s": row["mp4"]["duration_s"],
                                       "output": str(output)}, args.timeout)
        receipt = {"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                   "official_split": row["official_split"],
                   "input_sha256": row["mp4"]["sha256"],
                   "face_receipt_sha256": sha256(face_path),
                   "preprocess_version": PREPROCESS_VERSION, **result}
        write_json_atomic(receipt_path, receipt)
        return receipt

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            receipts = []
            for index, receipt in enumerate(pool.map(process, rows), 1):
                receipts.append(receipt)
                if index % 100 == 0:
                    write_json_atomic(args.output_dir / "progress.json",
                                      {"done": index, "total": len(rows)})
    finally:
        for worker in workers:
            worker.close()
    status = {key: sum(row["status"] == key for row in receipts)
              for key in sorted({row["status"] for row in receipts})}
    summary = {"n": len(rows), "status_counts": status,
               "mean_accepted_crops": (sum(row.get("accepted_crops", 0) for row in receipts) /
                                       len(rows) if rows else None),
               "mean_elapsed_s": (sum(row.get("elapsed_s", 0) for row in receipts) /
                                  len(rows) if rows else None)}
    write_json_atomic(args.output_dir / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(meta_path, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--face-root", type=Path)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("et", "sims_valid"), default="et")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.detector is None:
            parser.error("Worker needs --detector")
        worker_main(args.detector)
    else:
        for name in ("manifest", "face_root", "detector", "output_dir"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
