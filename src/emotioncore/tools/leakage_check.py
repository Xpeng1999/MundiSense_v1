from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from emotioncore.utils.common import read_jsonl, sha256_file, write_json_atomic

ET_ID = re.compile(r"G\d{5}_\d+_\d+_\d+")


def iter_records(path: Path):
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("rows", data.get("records", [data]))
        yield from data


def media_paths(row: dict, media_root: Path | None = None) -> set[Path]:
    paths = set()
    for key in ("videos", "audios"):
        for value in row.get(key, []) or []:
            paths.add(Path(value))
    for key in ("source", "video", "audio"):
        value = row.get(key)
        if isinstance(value, str):
            paths.add(Path(value))
    for key in ("mp4", "wav", "media"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("path"):
            paths.add(Path(value["path"]))
    if media_root:
        paths = {path if path.is_absolute() else media_root / path for path in paths}
    return paths


def record_ids(row: dict, paths: set[Path]) -> set[str]:
    explicit = row.get("utt_id") or row.get("id")
    ids = set(ET_ID.findall(str(explicit))) if explicit else set()
    for path in paths:
        ids.update(ET_ID.findall(str(path)))
    return ids


def fingerprint(path: Path, cache: dict[Path, str]) -> str | None:
    if path not in cache:
        cache[path] = sha256_file(path) if path.is_file() else ""
    return cache[path] or None


def features(row: dict, root: Path | None, hash_media: bool,
             cache: dict[Path, str]) -> dict:
    paths = media_paths(row, root)
    ids = record_ids(row, paths)
    speakers = {match.group().split("_")[2] for value in ids for match in ET_ID.finditer(value)}
    if row.get("speaker_id") is not None:
        speakers.add(str(row["speaker_id"]).zfill(2))
    dialogues = {tuple(value.split("_")[:2]) for value in ids}
    hashes = set()
    for key in ("mp4", "wav", "media"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("sha256"):
            hashes.add(value["sha256"])
    if hash_media:
        hashes.update(filter(None, (fingerprint(path, cache) for path in paths)))
    return {"ids": ids, "speakers": speakers, "dialogues": dialogues,
            "hashes": hashes, "paths": paths}


def check(train_paths: list[Path], eval_path: Path, train_media_root: Path | None = None,
          eval_media_root: Path | None = None, hash_media: bool = True) -> dict:
    cache: dict[Path, str] = {}
    training = {key: set() for key in ("ids", "speakers", "dialogues", "hashes")}
    train_rows = 0
    for path in train_paths:
        for row in iter_records(path):
            item = features(row, train_media_root, hash_media, cache)
            for key in training:
                training[key].update(item[key])
            train_rows += 1
    details = []
    counts = collections.Counter()
    for row in iter_records(eval_path):
        item = features(row, eval_media_root, hash_media, cache)
        hits = {"utterance": sorted(item["ids"] & training["ids"]),
                "media_sha256": sorted(item["hashes"] & training["hashes"]),
                "speaker": sorted(item["speakers"] & training["speakers"]),
                "dialogue": [list(x) for x in sorted(item["dialogues"] & training["dialogues"])]}
        for key, value in hits.items():
            counts[key] += bool(value)
        details.append({"id": row.get("utt_id") or row.get("id") or sorted(item["ids"]), **hits})
    return {"train_manifests": [str(x) for x in train_paths], "eval_manifest": str(eval_path),
            "train_rows": train_rows, "eval_rows": len(details),
            "overlap_counts": {key: counts[key] for key in ("utterance", "media_sha256", "speaker", "dialogue")},
            "overlap_details": [x for x in details if any(x[key] for key in ("utterance", "media_sha256", "speaker", "dialogue"))],
            "media_hashes_computed": hash_media, "missing_media_paths": [str(p) for p, digest in cache.items() if not digest]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training and evaluation manifests at four leakage levels")
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--train-media-root", type=Path)
    parser.add_argument("--eval-media-root", type=Path)
    parser.add_argument("--skip-content-hash", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.train, args.eval, args.train_media_root, args.eval_media_root,
                   hash_media=not args.skip_content_hash)
    if args.output:
        write_json_atomic(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
