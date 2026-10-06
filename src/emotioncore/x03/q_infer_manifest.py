"""Freeze Q outer-fold and final-target inference membership before scoring."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA
from emotioncore.x03.q_manifest import FOLD_SHA, make_record, read_jsonl, sha256, write_jsonl


def et_record(row: dict, form: str, missing: set[str], media_root: Path,
              fold: int | None) -> dict:
    record = make_record(row, form, missing, media_root)
    record["fold"] = fold
    if form == "V" and record["face_missing"]:
        record["effective_form"] = "uniform"
        record["face_missing_fallback"] = False
    else:
        record["face_missing_fallback"] = form == "AV" and record["face_missing"]
    return record


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.folds) != FOLD_SHA or sha256(args.face_policy) != FACE_SHA:
        raise ValueError("Frozen fold or face policy changed")
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    missing = set(json.loads(args.face_policy.read_text(encoding="utf-8"))["face_missing_utt_ids"])
    rows = read_jsonl(args.manifest)
    if len(rows) != 19250 or len({r["utt_id"] for r in rows}) != len(rows) or len(missing) != 43:
        raise ValueError("ET inference manifest coverage changed")
    outputs = {}
    for fold in folds["folds"]:
        index = fold["fold"]
        held = set(fold["outer_holdout_speakers"])
        selection = sorted((r for r in rows if r["speaker_id"] in held), key=lambda r: r["utt_id"])
        if len(selection) != fold["outer_holdout_count"]:
            raise ValueError("Outer-fold count changed")
        for form in ("A", "V", "AV"):
            name = f"fold{index}_{form}"
            records = [et_record(r, form, missing, args.media_root, index) for r in selection]
            path = args.output_dir / f"{name}.jsonl"
            outputs[name] = {"path": str(path), "sha256": write_jsonl(path, records),
                             "n": len(records), "uniform": sum(r["effective_form"] == "uniform" for r in records),
                             "face_missing_fallback": sum(r["face_missing_fallback"] for r in records)}
    selected = sorted((r for r in rows if r["official_split"] == "test"), key=lambda r: r["utt_id"])
    if len(selected) != 1929:
        raise ValueError("Sealed ET-test count changed")
    for form in ("A", "V", "AV"):
        name = f"final_et_test_{form}"
        records = [et_record(r, form, missing, args.media_root, None) for r in selected]
        path = args.output_dir / f"{name}.jsonl"
        outputs[name] = {"path": str(path), "sha256": write_jsonl(path, records),
                         "n": len(records), "uniform": sum(r["effective_form"] == "uniform" for r in records),
                         "face_missing_fallback": sum(r["face_missing_fallback"] for r in records)}
    for form in ("A", "V", "AV"):
        oof = [r for fold in range(5) for r in read_jsonl(args.output_dir / f"fold{fold}_{form}.jsonl")]
        if len(oof) != folds["cv_pool_count"] or len({r["utt_id"] for r in oof}) != len(oof):
            raise ValueError("OOF inference manifest coverage incomplete")
    meta = {"git": git, "command": "emotioncore.x03.q_infer_manifest",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "et_manifest_sha256": sha256(args.manifest), "folds_sha256": FOLD_SHA,
            "face_policy_sha256": FACE_SHA, "outputs": outputs}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    write_json_atomic(args.output_dir / "summary.json", outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
