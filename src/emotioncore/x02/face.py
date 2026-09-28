"""X02 five-fps YuNet tracking with process-isolated decoding."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import threading
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


FPS = 5
PIPELINE_VERSION = "1"
MAX_SELECTED_FRAMES = 8
IOU_THRESHOLD = 0.3
MAX_GAP = 2
CROP_EXPANSION = 0.2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = ix * iy
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def add_detections(tracks: list[dict], frame_index: int,
                   boxes: list[tuple[float, ...]]) -> list[int]:
    """Greedily match current boxes to recent tracks by largest IoU."""
    choices = []
    for track_index, track in enumerate(tracks):
        if frame_index - track["last_frame"] > MAX_GAP:
            continue
        for box_index, box in enumerate(boxes):
            score = iou(tuple(track["last_box"]), box)
            if score >= IOU_THRESHOLD:
                choices.append((score, track_index, box_index))
    assigned_tracks: set[int] = set()
    assigned_boxes: set[int] = set()
    mapping = [-1] * len(boxes)
    for _, track_index, box_index in sorted(choices, reverse=True):
        if track_index in assigned_tracks or box_index in assigned_boxes:
            continue
        mapping[box_index] = track_index
        assigned_tracks.add(track_index)
        assigned_boxes.add(box_index)
    for box_index, box in enumerate(boxes):
        track_index = mapping[box_index]
        if track_index < 0:
            track_index = len(tracks)
            tracks.append({"id": track_index, "boxes": {}, "area_sum": 0.0,
                           "last_frame": frame_index, "last_box": list(box)})
            mapping[box_index] = track_index
        track = tracks[track_index]
        track["boxes"][frame_index] = list(box)
        track["area_sum"] += box[2] * box[3]
        track["last_frame"] = frame_index
        track["last_box"] = list(box)
    return mapping


def rank_tracks(tracks: list[dict], decoded_frames: int) -> list[dict]:
    for track in tracks:
        count = len(track["boxes"])
        track["mean_area"] = track["area_sum"] / count
        track["presence_fraction"] = count / decoded_frames if decoded_frames else 0.0
        track["target_score"] = track["mean_area"] * track["presence_fraction"]
    return sorted(tracks, key=lambda t: (-t["target_score"], -len(t["boxes"]), t["id"]))


def selected_indices(indices: list[int], maximum: int = MAX_SELECTED_FRAMES) -> list[int]:
    if len(indices) <= maximum:
        return indices
    return sorted({indices[round(i * (len(indices) - 1) / (maximum - 1))]
                   for i in range(maximum)})


def _write_jpg(path: Path, image, cv2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError(f"Could not encode JPEG: {path}")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def _worker_clip(task: dict, detector, crop_root: Path) -> dict:
    import cv2
    import numpy as np

    video = Path(task["video"])
    if not video.is_file():
        return {"status": "failed", "error": "FileNotFoundError: media missing"}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"status": "failed", "error": "VideoCapture could not open media"}
    tracks: list[dict] = []
    encoded_frames: dict[int, bytes] = {}
    frame_boxes: dict[int, list[list[float]]] = {}
    multi_face_frames = 0
    zero_face_frames = 0
    decode_failures = 0
    requested = max(1, math.ceil(float(task["duration_s"]) * FPS))
    try:
        for frame_index in range(requested):
            capture.set(cv2.CAP_PROP_POS_MSEC, frame_index * 1000.0 / FPS)
            ok, frame = capture.read()
            if not ok:
                decode_failures += 1
                continue
            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)
            boxes = [] if faces is None else [tuple(map(float, row[:4])) for row in faces]
            if len(boxes) == 0:
                zero_face_frames += 1
            if len(boxes) >= 2:
                multi_face_frames += 1
            add_detections(tracks, frame_index, boxes)
            frame_boxes[frame_index] = [list(box) for box in boxes]
            encoded_ok, compressed = cv2.imencode(".jpg", frame,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not encoded_ok:
                decode_failures += 1
                continue
            encoded_frames[frame_index] = compressed.tobytes()
    finally:
        capture.release()
    decoded = len(encoded_frames)
    if decoded == 0:
        return {"status": "failed", "error": "No video frame decoded",
                "requested_frames": requested, "decoded_frames": 0,
                "zero_face_frames": 0, "multi_face_frames": 0,
                "decode_failures": decode_failures, "tracks": 0,
                "target_track": None, "area_ratio": None,
                "crop_paths": [], "whole_paths": [], "preview_path": None}
    ordered = rank_tracks(tracks, decoded)
    if not ordered:
        return {"status": "no_face", "requested_frames": requested,
                "decoded_frames": decoded, "zero_face_frames": zero_face_frames,
                "multi_face_frames": multi_face_frames,
                "decode_failures": decode_failures, "tracks": 0,
                "target_track": None, "area_ratio": None,
                "crop_paths": [], "whole_paths": [], "preview_path": None}
    target = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    ratio = target["mean_area"] / second["mean_area"] if second and second["mean_area"] else None
    sample_dir = crop_root / task["id"]
    crop_paths, whole_paths = [], []
    for frame_index in selected_indices(sorted(target["boxes"])):
        if frame_index not in encoded_frames:
            continue
        frame = cv2.imdecode(np.frombuffer(encoded_frames[frame_index], dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is None:
            continue
        x, y, w, h = target["boxes"][frame_index]
        height, width = frame.shape[:2]
        left = max(0, math.floor(x - CROP_EXPANSION * w))
        top = max(0, math.floor(y - CROP_EXPANSION * h))
        right = min(width, math.ceil(x + (1 + CROP_EXPANSION) * w))
        bottom = min(height, math.ceil(y + (1 + CROP_EXPANSION) * h))
        if right <= left or bottom <= top:
            continue
        crop = cv2.resize(frame[top:bottom, left:right], (448, 448))
        scale = math.sqrt(50176 / (width * height))
        whole = cv2.resize(frame, (max(1, round(width * scale)),
                                     max(1, round(height * scale))))
        crop_path = sample_dir / f"crop_{frame_index:04d}.jpg"
        whole_path = sample_dir / f"whole_{frame_index:04d}.jpg"
        _write_jpg(crop_path, crop, cv2)
        _write_jpg(whole_path, whole, cv2)
        crop_paths.append(str(crop_path))
        whole_paths.append(str(whole_path))
    preview_path = None
    if multi_face_frames:
        candidates = [i for i, boxes in frame_boxes.items()
                      if len(boxes) >= 2 and i in encoded_frames]
        if not candidates:
            return {"status": "ok" if crop_paths else "failed",
                    "error": None if crop_paths else "No valid target crop",
                    "requested_frames": requested, "decoded_frames": decoded,
                    "zero_face_frames": zero_face_frames,
                    "multi_face_frames": multi_face_frames,
                    "decode_failures": decode_failures,
                    "tracks": len(tracks), "target_track": target["id"],
                    "target_score": target["target_score"],
                    "target_presence_fraction": target["presence_fraction"],
                    "area_ratio": ratio, "crop_paths": crop_paths,
                    "whole_paths": whole_paths, "preview_path": None}
        frame_index = max(candidates, key=lambda i: len(frame_boxes[i]))
        frame = cv2.imdecode(np.frombuffer(encoded_frames[frame_index], dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if frame is not None:
            for box in frame_boxes[frame_index]:
                color = (0, 255, 0) if box == target["boxes"].get(frame_index) else (0, 0, 255)
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            scale = min(1.0, 600 / max(frame.shape[:2]))
            preview = cv2.resize(frame, None, fx=scale, fy=scale)
            preview_path = sample_dir / "preview.jpg"
            _write_jpg(preview_path, preview, cv2)
    return {"status": "ok" if crop_paths else "failed",
            "error": None if crop_paths else "No valid target crop",
            "requested_frames": requested, "decoded_frames": decoded,
            "zero_face_frames": zero_face_frames,
            "multi_face_frames": multi_face_frames,
            "decode_failures": decode_failures,
            "tracks": len(tracks), "target_track": target["id"],
            "target_score": target["target_score"],
            "target_presence_fraction": target["presence_fraction"],
            "area_ratio": ratio,
            "crop_paths": crop_paths, "whole_paths": whole_paths,
            "preview_path": str(preview_path) if preview_path else None}


def worker_main(model_path: Path, crop_root: Path) -> None:
    import cv2

    cv2.setNumThreads(1)
    detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.6, 0.3, 50)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            output = _worker_clip(task, detector, crop_root)
        except Exception as exc:
            output = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(output, ensure_ascii=False), flush=True)


def summarize(receipts: list[dict]) -> dict:
    counts = collections.Counter(row["status"] for row in receipts)
    multi = [row for row in receipts if row.get("multi_face_frames", 0) > 0]
    ratios = sorted(row["area_ratio"] for row in multi if row.get("area_ratio") is not None)
    def quantile(q):
        return ratios[round((len(ratios) - 1) * q)] if ratios else None
    by_speaker = collections.defaultdict(collections.Counter)
    for row in receipts:
        by_speaker[row["speaker_id"]][row["status"]] += 1
    return {"clips": len(receipts), "status_counts": dict(counts),
            "no_face_fraction": counts["no_face"] / len(receipts) if receipts else None,
            "multi_face_clips": len(multi),
            "multi_face_fraction": len(multi) / len(receipts) if receipts else None,
            "area_ratio": {"n": len(ratios), "p10": quantile(0.1), "p50": quantile(0.5),
                           "p90": quantile(0.9)},
            "by_speaker": {speaker: dict(items) for speaker, items in sorted(by_speaker.items())},
            "decoded_frames": sum(row.get("decoded_frames", 0) for row in receipts),
            "decode_failures": sum(row.get("decode_failures", 0) for row in receipts)}


def contact_sheets(chosen: list[dict], root: Path) -> list[dict]:
    from PIL import Image, ImageDraw, ImageOps

    root.mkdir(parents=True, exist_ok=True)
    output = []
    for page in range(math.ceil(len(chosen) / 10)):
        batch = chosen[page * 10:(page + 1) * 10]
        sheet = Image.new("RGB", (5 * 340, 2 * 285), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(batch):
            x, y = (index % 5) * 340, (index // 5) * 285
            with Image.open(row["preview_path"]) as source:
                thumb = ImageOps.contain(source.convert("RGB"), (320, 240))
            sheet.paste(thumb, (x + (320 - thumb.width) // 2, y + 5))
            draw.text((x + 5, y + 247),
                      f"{row['id']}  area ratio={row.get('area_ratio') or 0:.2f}",
                      fill="black")
        path = root / f"contact_{page:02d}.jpg"
        sheet.save(path, quality=88)
        output.append({"path": str(path), "sha256": sha256(path),
                       "sample_ids": [row["id"] for row in batch]})
    return output


def run(manifest: Path, media_root: Path, model_path: Path, run_root: Path,
        workers: int, timeout_s: float, allow_dirty: bool) -> None:
    root = Path.cwd()
    provenance = git_provenance(root, allow_dirty)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    face_root = run_root / "face"
    receipt_root = face_root / "receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv, "started_at": started_at,
            "ended_at": None, "manifest": str(manifest),
            "manifest_sha256": sha256(manifest), "yunet_model": str(model_path),
            "yunet_sha256": sha256(model_path), "sampling_fps": FPS,
            "pipeline_version": PIPELINE_VERSION,
            "iou_threshold": IOU_THRESHOLD, "max_gap": MAX_GAP,
            "crop_expansion": CROP_EXPANSION,
            "max_selected_frames": MAX_SELECTED_FRAMES,
            "timeout_s": timeout_s, "workers": workers}
    rows = []
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            rows.append(source)
    meta_path = face_root / "run_meta.json"
    if meta_path.is_file():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "yunet_sha256", "sampling_fps",
                    "pipeline_version", "iou_threshold", "max_gap", "crop_expansion"):
            if previous[key] != meta[key]:
                raise RuntimeError(f"Cannot resume face audit with changed {key}")
        if previous["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume face audit from a different Git commit")
        meta["started_at"] = previous["started_at"]
        meta["resumed_at"] = started_at
    write_json_atomic(meta_path, meta)
    local = threading.local()
    all_workers: list[PersistentJSONWorker] = []
    workers_lock = threading.Lock()
    def one(row: dict) -> dict:
        path = receipt_root / f"{row['utt_id']}.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (existing["input_sha256"] != row["mp4"]["sha256"]
                    or existing["pipeline_version"] != PIPELINE_VERSION
                    or existing["yunet_sha256"] != meta["yunet_sha256"]):
                raise RuntimeError(f"Face receipt provenance changed: {row['utt_id']}")
            return existing
        if not hasattr(local, "worker"):
            command = [sys.executable, "-u", "-m", "emotioncore.x02.face",
                       "--worker", "--model", str(model_path),
                       "--crop-root", str(face_root / "frames")]
            local.worker = PersistentJSONWorker(command, cwd=root, env=os.environ.copy(),
                                                stderr_path=face_root / "worker_errors.log")
            with workers_lock:
                all_workers.append(local.worker)
        task = {"id": row["utt_id"],
                "video": str(media_root / row["mp4"]["path"]),
                "duration_s": row["mp4"]["duration_s"]}
        result = local.worker.request(task, timeout_s)
        receipt = {"id": row["utt_id"], "speaker_id": row["speaker_id"],
                   "split": row["official_split"],
                   "input_sha256": row["mp4"]["sha256"],
                   "pipeline_version": PIPELINE_VERSION,
                   "yunet_sha256": meta["yunet_sha256"], **result}
        write_json_atomic(path, receipt)
        return receipt
    try:
        receipts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for index, receipt in enumerate(pool.map(one, rows), 1):
                receipts.append(receipt)
                if index % 100 == 0:
                    write_json_atomic(face_root / "progress.json",
                                      {"done": index, "total": len(rows)})
    finally:
        for worker in all_workers:
            worker.close()
    summary = summarize(receipts)
    previews = [row for row in receipts if row.get("preview_path")]
    rng = random.Random(20260928)
    chosen = rng.sample(previews, min(60, len(previews)))
    sheets = contact_sheets(chosen, face_root / "contact_sheets")
    contact = {"seed": 20260928, "sample_count": len(chosen),
               "ids": [row["id"] for row in chosen],
               "previews": [row["preview_path"] for row in chosen],
               "sheets": sheets}
    write_json_atomic(face_root / "contact_selection.json", contact)
    summary["contact_selection"] = str(face_root / "contact_selection.json")
    summary["contact_count"] = len(chosen)
    write_json_atomic(face_root / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["completed"] = len(receipts)
    meta["summary_sha256"] = sha256(face_root / "summary.json")
    write_json_atomic(face_root / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--crop-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.model, args.crop_root)
    else:
        if not all((args.manifest, args.media_root, args.run_root)):
            parser.error("--manifest, --media-root, and --run-root are required")
        run(args.manifest, args.media_root, args.model, args.run_root,
            args.workers, args.timeout, args.allow_dirty)


if __name__ == "__main__":
    main()
