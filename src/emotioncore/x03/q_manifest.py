"""Freeze Q training assignments and evaluation membership before fold zero."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA, SUBSET_SHA
from emotioncore.x03.q_smoke import MEDIA_ROOT


SEED = 20260929
FOLD_SHA = "85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def assignment(ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(ids)
    n = len(ordered)
    n_av = round(n * 0.6)
    n_a = round(n * 0.2)
    forms = ["AV"] * n_av + ["A"] * n_a + ["V"] * (n - n_av - n_a)
    random.Random(seed).shuffle(forms)
    return dict(zip(ordered, forms))


def make_record(row: dict, form: str, face_missing: set[str], media_root: Path) -> dict:
    uid = row["utt_id"]
    missing = uid in face_missing
    effective = "A" if missing and form in ("V", "AV") else form
    result = {"utt_id": uid, "speaker_id": row["speaker_id"],
              "truth_M": row["labels"]["M"], "assigned_form": form,
              "effective_form": effective, "face_missing": missing,
              "wav": str(MEDIA_ROOT / row["wav"]["path"]),
              "wav_sha256": row["wav"]["sha256"]}
    if not missing:
        receipt = json.loads((media_root / "receipts" / f"{uid}.json").read_text(encoding="utf-8"))
        if receipt["status"] != "ok" or receipt["utt_id"] != uid:
            raise ValueError(f"X03 media receipt mismatch: {uid}")
        result["video"] = receipt["output"]
        result["video_sha256"] = receipt["output_sha256"]
    return result


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sha256(path)


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if (sha256(args.folds) != FOLD_SHA or sha256(args.subsets) != SUBSET_SHA or
            sha256(args.face_policy) != FACE_SHA):
        raise ValueError("Frozen Q protocol SHA mismatch")
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    subsets = json.loads(args.subsets.read_text(encoding="utf-8"))
    policy = json.loads(args.face_policy.read_text(encoding="utf-8"))
    missing = set(policy["face_missing_utt_ids"])
    rows = read_jsonl(args.manifest)
    by_id = {row["utt_id"]: row for row in rows}
    if len(rows) != 19_250 or len(by_id) != len(rows) or len(missing) != 43:
        raise ValueError("Frozen ET or face-missing coverage differs")
    cv = set(folds["cv_pool_speakers"])
    outputs = {}
    for index in range(6):
        fold = folds["folds"][index] if index < 5 else folds["folds"][0]
        val_speakers = set(fold["inner_validation_speakers"])
        train_speakers = set(fold["inner_train_speakers"]) if index < 5 else cv - val_speakers
        if train_speakers & val_speakers:
            raise ValueError("Training and inner validation speakers overlap")
        train = [row for row in rows if row["speaker_id"] in train_speakers]
        name = f"fold{index}" if index < 5 else "final"
        forms = assignment([row["utt_id"] for row in train], SEED + index)
        train_records = [make_record(row, forms[row["utt_id"]], missing, args.media_root)
                         for row in sorted(train, key=lambda r: r["utt_id"])]
        subset = subsets["folds"][index if index < 5 else 0]
        val_ids = subset["utt_ids"]
        if any(by_id[uid]["speaker_id"] not in val_speakers for uid in val_ids):
            raise ValueError("Validation subset speaker differs")
        val_records = [make_record(by_id[uid], "AV", missing, args.media_root) for uid in val_ids]
        train_path = args.output_dir / f"{name}_train.jsonl"
        val_path = args.output_dir / f"{name}_val.jsonl"
        train_sha = write_jsonl(train_path, train_records)
        val_sha = write_jsonl(val_path, val_records)
        counts = {form: sum(r["assigned_form"] == form for r in train_records)
                  for form in ("A", "V", "AV")}
        fallback = sum(r["face_missing"] and r["assigned_form"] != "A" for r in train_records)
        outputs[name] = {"train_path": str(train_path), "train_sha256": train_sha,
                         "train_count": len(train_records), "val_path": str(val_path),
                         "val_sha256": val_sha, "val_count": len(val_records),
                         "assigned_counts": counts, "face_missing_fallback_train": fallback}
    meta = {"git": git, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest), "folds_sha256": FOLD_SHA,
            "subsets_sha256": SUBSET_SHA, "face_policy_sha256": FACE_SHA,
            "seed": SEED, "runs": outputs}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    write_json_atomic(args.output_dir / "summary.json", outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
