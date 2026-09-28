"""Prepare only CH-SIMS valid media and server-only text for X02 encoders."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, run_with_hard_timeout, write_json_atomic


EXPECTED_VALID = 647


def validate_adapter(rows: list[dict]) -> None:
    if (len(rows) != EXPECTED_VALID or
            len({row["utt_id"] for row in rows}) != len(rows) or
            any(row.get("official_split") != "sims_valid" or
                row.get("speaker_id") != "sims_valid" or
                row.get("dataset") != "CH-SIMS-v2" for row in rows)):
        raise ValueError("Expected the 647-row CH-SIMS valid-only adapter")


def prepare_audio(source: Path, target: Path, expected_sha: str | None,
                  timeout: float) -> dict:
    if target.is_file():
        return {"status": "ok", "wav_sha256": sha256(target), "resumed": True}
    if not source.is_file():
        return {"status": "missing", "wav_sha256": None}
    if sha256(source) != expected_sha:
        raise ValueError(f"SIMS video SHA differs from frozen X01 manifest: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.wav")
    result = run_with_hard_timeout(["ffmpeg", "-nostdin", "-v", "error", "-y",
                                    "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                                    str(temporary)], timeout_s=timeout)
    if result["timed_out"] or result["returncode"]:
        temporary.unlink(missing_ok=True)
        return {"status": "failed", "wav_sha256": None,
                "error": "timeout" if result["timed_out"] else result["stderr"][:200]}
    os.replace(temporary, target)
    return {"status": "ok", "wav_sha256": sha256(target), "elapsed_s": result["elapsed_s"]}


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    with args.valid_manifest.open(encoding="utf-8") as stream:
        valid = [json.loads(line) for line in stream]
    if len(valid) != EXPECTED_VALID or any(row["mode"] != "valid" for row in valid):
        raise ValueError("CH-SIMS test lockbox must not reach X02 preparation")
    expected = {(row["video_id"], row["clip_id"]) for row in valid}
    if len(expected) != EXPECTED_VALID:
        raise ValueError("Repeated SIMS valid IDs")
    with args.meta_csv.open(encoding="utf-8-sig", newline="") as stream:
        source_rows = {(row["video_id"], row["clip_id"]): row
                       for row in csv.DictReader(stream) if row["mode"] == "valid"}
    if set(source_rows) != expected:
        raise ValueError("SIMS valid CSV IDs differ from frozen X01 manifest")
    args.output_root.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "valid_manifest_sha256": sha256(args.valid_manifest),
            "meta_csv_sha256": sha256(args.meta_csv), "source_video_root": str(args.video_root),
            "rows": len(valid), "ffmpeg_timeout_s": args.timeout}
    write_json_atomic(args.output_root / "run_meta.json", meta)
    jobs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in valid:
            key = (row["video_id"], row["clip_id"])
            video = args.video_root / row["media"]["path"]
            wav = args.output_root / "media" / "Audio" / "wav" / row["video_id"] / f"{row['clip_id']}.wav"
            jobs[executor.submit(prepare_audio, video, wav, row["media"].get("sha256"),
                                 args.timeout)] = (row, video, wav, key)
        completed = {}
        for future in as_completed(jobs):
            row, video, wav, key = jobs[future]
            completed[key] = (row, video, wav, future.result())
            if len(completed) % 50 == 0 or len(completed) == len(valid):
                write_json_atomic(args.output_root / "progress.json",
                                  {"done": len(completed), "total": len(valid)})
    adapted = []
    statuses = {}
    for row in valid:
        key = (row["video_id"], row["clip_id"])
        _, video, wav, audio = completed[key]
        statuses[audio["status"]] = statuses.get(audio["status"], 0) + 1
        text = source_rows[key]["text"]
        text_path = args.output_root / "media" / "Text" / "json" / row["video_id"] / f"{row['clip_id']}.json"
        write_json_atomic(text_path, {"content": text})
        adapted.append({"utt_id": f"{row['video_id']}__{row['clip_id']}",
                        "dataset": "CH-SIMS-v2", "speaker_id": "sims_valid",
                        "official_split": "sims_valid",
                        "wav": {"path": str(wav.relative_to(args.output_root / "media")),
                                "sha256": audio["wav_sha256"]},
                        "mp4": {"path": str(video),
                                "sha256": row["media"].get("sha256"),
                                "duration_s": row["media"].get("duration_s") or 0.0}})
    validate_adapter(adapted)
    path = args.output_root / "sims_valid_adapter.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in adapted:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta.update({"ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "adapter_path": str(path), "adapter_sha256": sha256(path),
                 "audio_status": statuses, "transcripts_server_only": True})
    write_json_atomic(args.output_root / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
