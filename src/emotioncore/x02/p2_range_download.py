"""Budgeted, hash-verified range downloader for the pinned P2 public model."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from emotioncore.x02.download_models import MAX_DOWNLOAD_BYTES, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


CHUNK_BYTES = 64 * 1024 * 1024


class TransferBudget:
    def __init__(self, used: int):
        self.used = used
        self.lock = threading.Lock()

    def add(self, size: int) -> None:
        with self.lock:
            self.used += size
            if self.used > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("P2 actual transferred bytes exceed the 100 GB X02 budget")


def range_get(url: str, start: int, end: int, expected_total: int,
              budget: TransferBudget) -> bytes:
    import requests

    expected = end - start + 1
    last_error = None
    for attempt in range(1, 11):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(20, 90)) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}")
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range")
                    if content_range != f"bytes {start}-{end}/{expected_total}":
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                elif start != 0 or expected != expected_total:
                    raise ValueError("Server ignored a partial Range request")
                chunks = []
                received = 0
                for piece in response.iter_content(chunk_size=1024 * 1024):
                    if piece:
                        budget.add(len(piece))
                        chunks.append(piece)
                        received += len(piece)
                        if received > expected:
                            raise ValueError("Range response exceeded requested bytes")
                if received != expected:
                    raise IOError(f"Short Range response: {received} != {expected}")
                return b"".join(chunks)
        except (Exception,) as exc:
            if "exceed the 100 GB" in str(exc):
                raise
            last_error = exc
            time.sleep(min(2 * attempt, 15))
    raise RuntimeError(f"Range {start}-{end} failed after 10 attempts: {last_error}")


def download_file(spec: dict, model: dict, target_dir: Path,
                  progress_dir: Path, budget: TransferBudget, workers: int) -> dict:
    name = spec["path"]
    target = target_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_sha = spec.get("upstream_lfs_sha256")
    def verify(path: Path) -> str:
        digest = sha256(path)
        if expected_sha is not None and digest != expected_sha:
            raise ValueError(f"P2 SHA-256 mismatch for {name}")
        if expected_sha is None and spec.get("upstream_git_blob"):
            payload = path.read_bytes()
            blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
            if blob != spec["upstream_git_blob"]:
                raise ValueError(f"P2 Git blob mismatch for {name}")
        return digest

    if target.is_file() and target.stat().st_size == spec["bytes"]:
        actual = verify(target)
        return {**spec, "local_path": str(target), "local_sha256": actual,
                "resumed_verified": True}
    pending = target.with_name(target.name + ".range-pending")
    progress_path = progress_dir / (name.replace("/", "_") + ".json")
    size = spec["bytes"]
    chunks = math.ceil(size / CHUNK_BYTES)
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress["revision"] != model["revision"] or
                progress["bytes"] != size or progress["chunk_bytes"] != CHUNK_BYTES or
                not pending.is_file() or pending.stat().st_size != size):
            raise ValueError(f"Incompatible P2 partial download: {name}")
        complete = set(progress["completed_chunks"])
    else:
        complete = set()
        with pending.open("wb") as stream:
            stream.truncate(size)
        progress = {"revision": model["revision"], "path": name,
                    "bytes": size, "chunk_bytes": CHUNK_BYTES,
                    "completed_chunks": []}
        write_json_atomic(progress_path, progress)
    fd = os.open(pending, os.O_WRONLY)
    try:
        url = f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{name}"
        def fetch(index: int):
            start = index * CHUNK_BYTES
            end = min(size, start + CHUNK_BYTES) - 1
            return index, start, range_get(url, start, end, size, budget)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, index) for index in range(chunks) if index not in complete}
            try:
                for future in as_completed(futures):
                    index, start, data = future.result()
                    written = os.pwrite(fd, data, start)
                    if written != len(data):
                        raise IOError(f"Short write for P2 chunk {name}:{index}")
                    os.fsync(fd)
                    complete.add(index)
                    progress["completed_chunks"] = sorted(complete)
                    progress["actual_transferred_bytes"] = budget.used
                    write_json_atomic(progress_path, progress)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        os.close(fd)
    if len(complete) != chunks:
        raise RuntimeError(f"Incomplete P2 file: {name}")
    actual = verify(pending)
    os.replace(pending, target)
    progress["verified_sha256"] = actual
    progress["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(progress_path, progress)
    return {**spec, "local_path": str(target), "local_sha256": actual}


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    model = plan["models"]["P2"]
    if model["repo_id"] != "Qwen/Qwen3-Omni-30B-A3B-Instruct" or \
       plan["total_planned_bytes"] > MAX_DOWNLOAD_BYTES:
        raise ValueError("P2 plan is not the pinned public model within budget")
    receipts_path = args.run_root / "model_downloads.json"
    existing = json.loads(receipts_path.read_text(encoding="utf-8"))
    previous_verified = existing["total_verified_bytes"]
    if previous_verified + model["planned_bytes"] + args.prior_failed_bytes > MAX_DOWNLOAD_BYTES:
        raise RuntimeError("P2 plus prior transfers may exceed the 100 GB budget")
    budget = TransferBudget(previous_verified + args.prior_failed_bytes)
    target_dir = args.run_root / "models" / "P2"
    progress_dir = args.run_root / "p2_range_download"
    progress_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "plan_sha256": sha256(args.plan), "revision": model["revision"],
            "source_url": model["source_url"], "chunk_bytes": CHUNK_BYTES,
            "prior_verified_bytes": previous_verified,
            "prior_failed_transfer_bytes_estimate": args.prior_failed_bytes,
            "budget_bytes": MAX_DOWNLOAD_BYTES}
    write_json_atomic(progress_dir / "run_meta.json", meta)
    verified = []
    for spec in model["files"]:
        verified.append(download_file(spec, model, target_dir, progress_dir,
                                      budget, args.workers))
        write_json_atomic(progress_dir / "progress.json",
                          {"verified_files": [item["path"] for item in verified],
                           "total_files": len(model["files"]),
                           "actual_transferred_bytes_estimate": budget.used})
    existing["models"]["P2"] = {**model, "files": verified, "local_dir": str(target_dir)}
    existing["total_verified_bytes"] = sum(sum(item["bytes"] for item in entry["files"])
                                            for entry in existing["models"].values())
    write_json_atomic(receipts_path, existing)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["actual_transferred_bytes_estimate"] = budget.used
    meta["receipt_sha256"] = sha256(receipts_path)
    write_json_atomic(progress_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prior-failed-bytes", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3 or args.prior_failed_bytes < 0:
        parser.error("workers must be 1..3 and prior-failed-bytes nonnegative")
    run(args)


if __name__ == "__main__":
    main()
