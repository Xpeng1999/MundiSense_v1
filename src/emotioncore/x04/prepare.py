"""Register only the authorized X04a corpora; all media live on H:."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from emotioncore.x02.face import sha256
from emotioncore.x02.runtime import git_provenance, run_with_hard_timeout, write_json_atomic

ROOT_FREE_MIN = 80_000_000_000
MER_MAP = dict(neutral="neutral", happy="joy", sad="sadness", angry="anger",
               surprise="surprise", worried="worried")
CREMA_MAP = dict(ANG="anger", DIS="disgust", FEA="fear", HAP="joy",
                 NEU="neutral", SAD="sadness")


def storage_gate() -> dict:
    free = shutil.disk_usage("/").free
    result = {"root_free_bytes": free, "root_min_bytes": ROOT_FREE_MIN,
              "h_free_bytes": shutil.disk_usage("/mnt/h").free}
    if free < ROOT_FREE_MIN:
        raise RuntimeError(f"STOP root_free_bytes={free} < {ROOT_FREE_MIN}")
    return result


def require_h(path: Path) -> None:
    if not path.resolve().is_relative_to(Path("/mnt/h")):
        raise ValueError("New media and derived artifacts must be on /mnt/h")


def probe(path: Path) -> dict:
    if not path.is_file():
        return {"status": "missing", "path": str(path), "sha256": None,
                "duration_s": None}
    result = run_with_hard_timeout(["ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,sample_rate,channels",
        "-of", "json", str(path)], 120)
    base = {"path": str(path), "sha256": sha256(path),
            "bytes": path.stat().st_size}
    if result["returncode"] or result["timed_out"]:
        return {**base, "status": "failed", "duration_s": None,
                "error": result["stderr"][-500:]}
    info = json.loads(result["stdout"])
    return {**base, "status": "ok",
            "duration_s": float(info.get("format", {}).get("duration", 0)),
            "streams": info.get("streams", [])}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract(mer: Path, derived: Path, output: Path) -> dict:
    storage_gate()
    require_h(derived)
    archives = []
    view = derived / "archive_view"
    for medium in ("audio", "video"):
        for package in ("track1_train", "track1_track2_candidate"):
            source = mer / f"{medium}_7z" / f"{medium}_{package}"
            target = view / f"{medium}_7z" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source, target_is_directory=True)
            files = sorted(source.glob("*.zip"))
            if not files:
                raise ValueError(f"No archives for {medium}/{package}")
            for path in files:
                storage_gate()
                record = {"path": str(path), "sha256": sha256(path),
                          "bytes": path.stat().st_size, "medium": medium,
                          "package": package}
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
                    if any(Path(n).is_absolute() or ".." in Path(n).parts or
                           Path(n).parts[0] != medium for n in names):
                        raise ValueError(f"Unsafe archive members in {path}")
                    record["members"] = len(names)
                archives.append(record)
    receipt = {"archives": archives, "official_script": str(mer / "extract_mer2026_archives.sh"),
               "official_script_sha256": sha256(mer / "extract_mer2026_archives.sh"),
               "view": str(view), "target": str(derived / "media"),
               "openface_visible": False}
    write_json_atomic(output / "archive_receipt.json", receipt)
    marker = derived / "extract_complete.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["archives"] != archives:
            raise ValueError("Extracted archives differ from recorded source hashes")
        return old
    result = run_with_hard_timeout(["bash", receipt["official_script"], str(view),
        str(derived / "media"), "track1_train", "track1_track2_candidate"], 7200)
    write_json_atomic(output / "extract_receipt.json", result)
    if result["returncode"] or result["timed_out"]:
        raise RuntimeError("Official extraction failed; see extract_receipt.json")
    write_json_atomic(marker, receipt)
    return receipt


def register(args) -> None:
    import pandas as pd
    import numpy as np
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    require_h(args.derived)
    meta = {"git": git_provenance(Path.cwd()), "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "storage": storage_gate(), "CH_SIMS_read": False}
    write_json_atomic(args.output / "prepare_run_meta.json", meta)
    extract(args.mer, args.derived, args.output)
    requests = []
    splits = {}
    for split, expected in (("train", 9395), ("test", 572)):
        source = args.mer / f"track1_{split}.csv"
        records = pd.read_csv(source, usecols=["name", "discrete"], keep_default_na=False)
        if len(records) != expected or records.name.duplicated().any():
            raise ValueError(f"MER {split} frozen count or unique IDs mismatch")
        splits[split] = set(records.name)
        for item in records.to_dict("records"):
            if item["discrete"] not in MER_MAP:
                raise ValueError(f"Unsupported MER label: {item['discrete']}")
            requests.append({"utt_id": "mer_" + item["name"], "native_id": item["name"],
                "corpus": "MER", "official_split": split, "speaker_id": "unknown",
                "label": MER_MAP[item["discrete"]], "native_label": item["discrete"],
                "audio_path": str(args.derived / "media/audio" / (item["name"] + ".wav")),
                "video_path": str(args.derived / "media/video" / (item["name"] + ".mp4"))})
    if splits["train"] & splits["test"]:
        raise ValueError("STOP MER train/test ID overlap")
    for wav in sorted((args.crema / "AudioWAV").glob("*.wav")):
        fields = wav.stem.split("_")
        if len(fields) != 4 or fields[2] not in CREMA_MAP:
            raise ValueError(f"Unexpected CREMA official filename: {wav.name}")
        requests.append({"utt_id": "crema_" + wav.stem, "native_id": wav.stem,
            "corpus": "CREMA", "official_split": "train", "speaker_id": fields[0],
            "label": CREMA_MAP[fields[2]], "native_label": fields[2],
            "audio_path": str(wav), "video_path": str(args.crema / "VideoFlash" / (wav.stem + ".flv"))})
    records_dir = args.derived / "registration"
    records_dir.mkdir(parents=True, exist_ok=True)
    def one(row):
        path = records_dir / (row["utt_id"] + ".json")
        if path.exists():
            previous = json.loads(path.read_text())
            if any(previous[key] != row[key] for key in row):
                raise ValueError(f"Cannot resume changed corpus row {row['utt_id']}")
            return previous
        storage_gate()
        receipt = {**row, "wav": probe(Path(row["audio_path"])),
                   "mp4": probe(Path(row["video_path"]))}
        write_json_atomic(path, receipt)
        return receipt
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(one, requests), 1):
            rows.append(row)
            if i % 100 == 0:
                write_json_atomic(args.output / "prepare_progress.json", {"done": i, "total": len(requests)})
    summaries = {}
    for corpus, split in (("MER", "train"), ("MER", "test"), ("CREMA", "train")):
        selected = [row for row in rows if row["corpus"] == corpus and row["official_split"] == split]
        manifest = args.derived / "manifests" / f"{corpus.lower()}_{split}.jsonl"
        write_rows(manifest, selected)
        durations = [r["wav"]["duration_s"] for r in selected if r["wav"]["duration_s"] is not None]
        resolutions = collections.Counter(f"{s['width']}x{s['height']}" for r in selected for s in r["mp4"].get("streams", []) if s["codec_type"] == "video")
        summaries[f"{corpus}_{split}"] = {"count": len(selected),
            "classes": dict(collections.Counter(r["label"] for r in selected)),
            "speakers": len(set(r["speaker_id"] for r in selected)) if corpus == "CREMA" else None,
            "audio_probe_status": dict(collections.Counter(r["wav"]["status"] for r in selected)),
            "video_probe_status": dict(collections.Counter(r["mp4"]["status"] for r in selected)),
            "duration_s": dict(zip(("min", "p10", "p50", "p90", "max"), np.quantile(durations, [0,.1,.5,.9,1]).tolist())),
            "video_resolutions": dict(resolutions), "manifest": str(manifest), "manifest_sha256": sha256(manifest)}
    train = [r for r in rows if r["corpus"] == "MER" and r["official_split"] == "train"]
    train.sort(key=lambda r: r["utt_id"])
    indices = np.random.default_rng(20261006).choice(len(train), 200, replace=False)
    bench = args.derived / "manifests/mer_benchmark200.jsonl"
    write_rows(bench, [train[i] for i in sorted(indices)])
    write_json_atomic(args.output / "data_registration.json", {"cohorts": summaries,
        "benchmark200": {"path": str(bench), "sha256": sha256(bench), "seed": 20261006},
        "probe_is_not_full_decode": True, "full_decode_via_face_pipeline_pending": True})
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
                registration_sha256=sha256(args.output / "data_registration.json"), storage_end=storage_gate())
    write_json_atomic(args.output / "prepare_run_meta.json", meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mer", type=Path, default=Path("/mnt/h/emotion-data/MER2026"))
    p.add_argument("--crema", type=Path, default=Path("/home/administrator/emotion-data/emotionAI_dataset/CREMA-D"))
    p.add_argument("--derived", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        register(args)
    except Exception as exc:
        write_json_atomic(args.output / "prepare_stop.json", {"exception": type(exc).__name__, "reason": str(exc),
            "at": dt.datetime.now(dt.timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
