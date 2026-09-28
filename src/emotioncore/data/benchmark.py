from __future__ import annotations

import argparse
import json
from pathlib import Path

from emotioncore.utils.common import read_jsonl, sha256_file, write_json_atomic


def emit(path: Path, rows: list[dict]) -> dict:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}


def build(results: Path) -> None:
    et = list(read_jsonl(results / "et_utterances.jsonl"))
    sims = list(read_jsonl(results / "chsims_v2_clips.jsonl"))
    if len(et) != 19250:
        raise ValueError(f"ET row count: {len(et)}")
    specs = {
        "et_test": [row for row in et if row["official_split"] == "test"],
        "et_val": [row for row in et if row["official_split"] == "val"],
        "sims_valid": [row for row in sims if row["mode"] == "valid"],
        "sims_test_lockbox": [
            {"video_id": row["video_id"], "clip_id": row["clip_id"],
             "media": {"path": row["media"]["path"], "sha256": row["media"].get("sha256")}}
            for row in sims if row["mode"] == "test"],
    }
    if len(specs["et_test"]) != 1929 or len(specs["et_val"]) != 1908:
        raise ValueError("Official ET split count mismatch")
    summary = {name: emit(results / f"{name}.jsonl", rows) for name, rows in specs.items()}
    write_json_atomic(results / "benchmark_manifests.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    build(parser.parse_args().results)


if __name__ == "__main__":
    main()
