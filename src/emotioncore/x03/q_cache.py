"""Process frozen Q sample media in hard-timed child workers into H-drive cache."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import sys
import threading
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import assert_modality_receipt


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    rows = read_jsonl(args.manifest)
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Training/validation manifest contains duplicate IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest), "n": len(rows),
            "workers": args.workers, "timeout_s": args.timeout}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    local = threading.local()
    children = []
    lock = threading.Lock()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def process(row: dict) -> dict:
        uid = row["utt_id"]
        condition = row["effective_form"]
        if condition == "uniform":
            if row["assigned_form"] != "V" or not row["face_missing"]:
                raise ValueError("Uniform Q input outside missing-face V evaluation")
            return {"utt_id": uid, "status": "uniform", "effective_form": condition,
                    "face_missing": True}
        npz = args.cache_root / uid / f"{condition}.npz"
        receipt_path = args.cache_root / uid / f"{condition}.json"
        if receipt_path.is_file():
            prior = json.loads(receipt_path.read_text(encoding="utf-8"))
            if prior["status"] == "ok":
                if (prior["input_wav_sha256"] != row["wav_sha256"] or
                        prior.get("input_video_sha256") != row.get("video_sha256") or
                        prior["condition"] != condition or
                        sha256(npz) != prior["sha256"]):
                    raise ValueError(f"Cached Q processor input mismatch: {uid}/{condition}")
                return prior
        if not hasattr(local, "worker"):
            local.worker = PersistentJSONWorker(
                [sys.executable, "-u", "-m", "emotioncore.x03.q_smoke", "--worker"],
                cwd=Path.cwd(), env=env, stderr_path=args.output_dir / "worker_errors.log")
            with lock:
                children.append(local.worker)
        task = {"condition": condition, "wav": row["wav"],
                "wav_sha256": row["wav_sha256"],
                "video": row.get("video"),
                "video_sha256": row.get("video_sha256"), "output": str(npz)}
        result = local.worker.request(task, args.timeout)
        if result["status"] == "ok":
            keys = set()
            if result["processor_has_audio"]:
                keys.add("input_features")
            if result["processor_has_video"]:
                keys.add("pixel_values_videos")
            assert_modality_receipt(condition, keys)
        receipt = {"utt_id": uid, "assigned_form": row["assigned_form"],
                   "effective_form": condition, "face_missing": row["face_missing"],
                   "input_wav_sha256": row["wav_sha256"],
                   "input_video_sha256": row.get("video_sha256"), **result}
        write_json_atomic(receipt_path, receipt)
        return receipt

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            receipts = []
            for n, receipt in enumerate(pool.map(process, rows), 1):
                receipts.append(receipt)
                if n % 100 == 0:
                    write_json_atomic(args.output_dir / "progress.json",
                                      {"done": n, "total": len(rows)})
        summary = {"n": len(rows), "ok": sum(r["status"] == "ok" for r in receipts),
                   "uniform": sum(r["status"] == "uniform" for r in receipts),
                   "failed": sum(r["status"] not in ("ok", "uniform") for r in receipts),
                   "by_effective_form": {c: sum(r["effective_form"] == c for r in receipts)
                                         for c in ("A", "V", "AV", "uniform")},
                   "face_missing": sum(r["face_missing"] for r in receipts)}
        write_json_atomic(args.output_dir / "summary.json", summary)
        if summary["failed"]:
            raise RuntimeError(f"{summary['failed']} Q processor inputs failed")
        meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
        write_json_atomic(args.output_dir / "run_meta.json", meta)
    finally:
        for child in children:
            child.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
