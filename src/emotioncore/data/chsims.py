from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
from pathlib import Path

from emotioncore.data.emotiontalk import probe
from emotioncore.utils.common import load_paths, write_json_atomic


def build(config: Path, output: Path, workers: int) -> None:
    paths = load_paths(config)
    source = paths["source_data_root"] / "CH-SIMSV2/meta.csv"
    root = paths["derived_root"] / "chsims_v2/raw"
    with source.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))
    # Preserve every CSV row. The supplied file may differ from the planned count;
    # record the discrepancy for reviewer instead of silently dropping a clip.
    media_paths = [f"Raw/{r['video_id']}/{r['clip_id']}.mp4" for r in source_rows]
    unique = sorted(set(media_paths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        metadata = dict(zip(unique, pool.map(lambda rel: probe(root / rel), unique)))
    rows = []
    for raw, rel in zip(source_rows, media_paths):
        row = {key: raw[key] for key in ("video_id", "clip_id", "mode", "annotation")}
        for key in ("label", "label_T", "label_A", "label_V"):
            row[key] = float(raw[key])
        row["media"] = {"path": rel, **metadata[rel]}
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = collections.Counter(r["mode"] for r in rows)
    repeated = collections.Counter(media_paths)
    duplicates = {path: count for path, count in repeated.items() if count > 1}
    failures = [{"video_id": r["video_id"], "clip_id": r["clip_id"],
                 "mode": r["mode"], "path": r["media"]["path"], "error": r["media"]["error"]}
                for r in rows if "error" in r["media"]]
    write_json_atomic(output.with_name("chsims_audit.json"),
                      {"rows": len(rows), "expected_rows_in_x01": 4402,
                       "row_count_difference": len(rows) - 4402, "mode_counts": dict(counts),
                       "missing_count": sum(r["error"] == "missing" for r in failures),
                       "decode_failure_count": sum(r["error"] != "missing" for r in failures),
                       "failures": failures, "duplicate_path_count": sum(c - 1 for c in duplicates.values()),
                       "duplicate_paths": duplicates})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.config, args.output, args.workers)


if __name__ == "__main__":
    main()
