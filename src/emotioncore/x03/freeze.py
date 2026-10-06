"""Freeze reviewer-approved X03 validation subsets and face-missing policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from emotioncore.eval.metrics import LABELS


SEED = 20260929
SUBSET_N = 700
EXPECTED_FOLD_SHA = "85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce"
EXPECTED_MEDIA_SUMMARY_SHA = "b0869a835f501ef8e1d2f666a606c87d81c747652caa960ff90dafadb1e0c4e2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_frozen(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "  " + path.name + "\n", encoding="utf-8")
    return digest


def quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total < target or any(counts[label] == 0 for label in LABELS):
        raise ValueError("Insufficient validation examples for seven strata")
    exact = {label: target * counts[label] / total for label in LABELS}
    chosen = {label: math.floor(exact[label]) for label in LABELS}
    remainder = target - sum(chosen.values())
    for label in sorted(LABELS, key=lambda label: (-(exact[label] - chosen[label]), LABELS.index(label)))[:remainder]:
        chosen[label] += 1
    if sum(chosen.values()) != target or any(chosen[label] > counts[label] for label in LABELS):
        raise ValueError("Stratified quota calculation failed")
    return chosen


def freeze(manifest: Path, folds_path: Path, source_path: Path, config_dir: Path) -> dict:
    if sha256(folds_path) != EXPECTED_FOLD_SHA:
        raise ValueError("Frozen X02 fold SHA differs")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    rows = read_jsonl(manifest)
    if len(rows) != 19_250 or len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Frozen ET manifest coverage differs")
    by_id = {row["utt_id"]: row for row in rows}
    subset_folds = []
    for fold in folds["folds"]:
        inner = set(fold["inner_validation_speakers"])
        outer = set(fold["outer_holdout_speakers"])
        if inner & outer:
            raise ValueError("Inner validation speakers overlap outer holdout")
        candidates = [row for row in rows if row["speaker_id"] in inner]
        if len(candidates) != fold["inner_validation_count"]:
            raise ValueError("Inner validation count differs from frozen fold")
        grouped = {label: sorted(row["utt_id"] for row in candidates if row["labels"]["M"] == label)
                   for label in LABELS}
        counts = {label: len(grouped[label]) for label in LABELS}
        selected_counts = quotas(counts, SUBSET_N)
        rng = random.Random(SEED + fold["fold"])
        selected = sorted(uid for label in LABELS
                          for uid in rng.sample(grouped[label], selected_counts[label]))
        if len(selected) != SUBSET_N or len(set(selected)) != SUBSET_N:
            raise ValueError("Invalid validation subset count")
        if any(by_id[uid]["speaker_id"] not in inner for uid in selected):
            raise ValueError("Subset contains a non-inner speaker")
        subset_folds.append({"fold": fold["fold"], "seed": SEED + fold["fold"],
                             "inner_validation_speakers": sorted(inner),
                             "inner_validation_total": len(candidates),
                             "n": SUBSET_N, "label_counts": selected_counts,
                             "utt_ids": selected})
    subset_payload = {"version": 1, "seed": SEED, "size_per_fold": SUBSET_N,
                      "algorithm": "proportional seven-class largest-remainder quotas; sorted IDs; Python random.Random(seed+fold).sample per class",
                      "manifest_sha256": sha256(manifest), "folds_sha256": EXPECTED_FOLD_SHA,
                      "folds": subset_folds}
    subset_path = config_dir / "x03_inner_val_subsets.json"
    subset_sha = write_frozen(subset_path, subset_payload)

    source = json.loads(source_path.read_text(encoding="utf-8"))
    missing = source["face_missing_ids"]
    if source["media_summary_sha256"] != EXPECTED_MEDIA_SUMMARY_SHA or source["count"] != 43:
        raise ValueError("Face source summary differs")
    if missing != sorted(set(missing)) or len(missing) != 43 or any(uid not in by_id for uid in missing):
        raise ValueError("Face missing IDs differ from frozen ET manifest")
    face_payload = {"version": 1, "source_media_summary_sha256": EXPECTED_MEDIA_SUMMARY_SHA,
                    "source_missing_list_sha256": sha256(source_path), "face_missing_count": 43,
                    "face_missing_utt_ids": missing,
                    "training": {"assigned_A": "A", "assigned_V": "A", "assigned_AV": "A",
                                 "mark": "face_missing"},
                    "evaluation": {"V": "uniform_7_class_counted_in_denominator",
                                   "AV": "audio_only_forward_mark_face_missing_fallback",
                                   "E_visual_experts": "uniform_7_class",
                                   "H_visual_experts": "uniform_7_class"}}
    face_path = config_dir / "x03_face_missing_policy.json"
    face_sha = write_frozen(face_path, face_payload)
    return {"subset_path": str(subset_path), "subset_sha256": subset_sha,
            "face_policy_path": str(face_path), "face_policy_sha256": face_sha}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--face-source", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.manifest, args.folds, args.face_source, args.config_dir)))


if __name__ == "__main__":
    main()
