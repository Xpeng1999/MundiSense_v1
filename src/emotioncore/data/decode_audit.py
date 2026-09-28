"""Verify that supplied CH-SIMS v2 clips can be fully decoded, without inference."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import subprocess
from pathlib import Path

from emotioncore.utils.common import load_paths, read_jsonl, write_json_atomic


def decode(path: Path) -> tuple[bool, str | None]:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
         "-i", str(path), "-map", "0:v:0", "-map", "0:a:0?", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stderr.strip()[:500] or None


def run(paths_config: Path, manifest: Path, output: Path, workers: int = 4) -> None:
    paths = load_paths(paths_config)
    root = paths["derived_root"] / "chsims_v2/raw"
    rows = list(read_jsonl(manifest))

    def check(row: dict) -> dict:
        media = row["media"]
        if media.get("error") == "missing":
            return {"mode": row["mode"], "video_id": row["video_id"],
                    "clip_id": row["clip_id"], "status": "missing"}
        if media.get("error"):
            return {"mode": row["mode"], "video_id": row["video_id"],
                    "clip_id": row["clip_id"], "status": "probe_failed",
                    "error": media["error"]}
        success, error = decode(root / media["path"])
        return {"mode": row["mode"], "video_id": row["video_id"],
                "clip_id": row["clip_id"], "status": "ok" if success else "decode_failed",
                **({"error": error} if error else {})}

    counts = collections.defaultdict(collections.Counter)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(check, rows):
            counts[row["mode"]][row["status"]] += 1
            if row["status"] != "ok":
                failures.append(row)
    write_json_atomic(output, {"method": "ffmpeg -xerror -err_detect explode, full video/audio decode to null",
                               "rows": len(rows),
                               "counts_by_mode": {mode: dict(counter) for mode, counter in sorted(counts.items())},
                               "failures": failures})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.paths, args.manifest, args.output, args.workers)


if __name__ == "__main__":
    main()
