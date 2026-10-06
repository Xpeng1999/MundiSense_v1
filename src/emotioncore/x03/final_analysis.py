"""Build X03's preregistered main comparison, paired deltas and error analysis."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification, percentile
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_calibrate import ece15_with_invalid
from emotioncore.x03.q_manifest import FOLD_SHA, read_jsonl, sha256


QUALITY_SHA = "80330bb718e31a8155793ab83e396bd4741a87347858f0342fd039cdb098bb32"
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_REPS = 2000


def prediction_rows(path: Path, ids: set[str], point: str, *, force_face_error: bool = False) -> list[dict]:
    raw = read_jsonl(path)
    result = []
    for row in raw:
        uid = row.get("utt_id", row.get("id"))
        probability = row[point]
        if len(probability) != 7 or not np.isclose(sum(probability), 1, atol=1e-5):
            raise ValueError(f"Invalid seven-class X03 output: {uid}")
        forced = force_face_error and row.get("status") == "face_missing"
        pred = None if forced else LABELS[int(np.argmax(probability))]
        result.append({"id": uid, "speaker_id": row["speaker_id"],
                       "truth": row["truth_M"], "pred": pred,
                       "probabilities": probability, "forced_error": forced,
                       "fold": row.get("fold")})
    result.sort(key=lambda row: row["id"])
    if len(result) != len(ids) or {r["id"] for r in result} != ids:
        raise ValueError(f"Incomplete X03 prediction coverage: {path}")
    return result


def score(rows: list[dict]) -> dict:
    core = classification(rows)
    y = np.array([LABELS.index(r["truth"]) for r in rows])
    probabilities = np.array([r["probabilities"] for r in rows])
    invalid = np.array([r["pred"] is None for r in rows])
    core["ece15"] = ece15_with_invalid(probabilities, y, invalid)
    core["nonneutral_macro_recall"] = float(np.mean(
        [core["per_class"][label]["recall"] for label in LABELS[1:]]))
    return core


def paired_speaker_bootstrap(left: list[dict], right: list[dict]) -> dict:
    if [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("Paired systems have different ordered CV IDs")
    speakers = sorted({r["speaker_id"] for r in left})
    left_groups = defaultdict(list)
    right_groups = defaultdict(list)
    for a, b in zip(left, right, strict=True):
        if (a["truth"], a["speaker_id"]) != (b["truth"], b["speaker_id"]):
            raise ValueError("Paired CV systems disagree on labels or speakers")
        left_groups[a["speaker_id"]].append(a)
        right_groups[b["speaker_id"]].append(b)
    rng = random.Random(BOOTSTRAP_SEED)
    differences = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choices(speakers, k=len(speakers))
        a = [row for speaker in sampled for row in left_groups[speaker]]
        b = [row for speaker in sampled for row in right_groups[speaker]]
        differences.append(classification(a)["macro_f1"] - classification(b)["macro_f1"])
    return {"unit": "speaker", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS,
            "difference": classification(left)["macro_f1"] - classification(right)["macro_f1"],
            "lower_95": percentile(differences, .025),
            "upper_95": percentile(differences, .975)}


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.folds) != FOLD_SHA or sha256(args.quality) != QUALITY_SHA:
        raise ValueError("Frozen folds or truncation quality SHA changed")
    frozen = json.loads(args.folds.read_text(encoding="utf-8"))
    manifest = read_jsonl(args.et_manifest)
    cv_ids = {r["utt_id"] for r in manifest if r["speaker_id"] in frozen["cv_pool_speakers"]}
    test_ids = {r["utt_id"] for r in manifest if r["official_split"] == "test"}
    if len(cv_ids) != 16366 or len(test_ids) != 1929:
        raise ValueError("Frozen ET benchmark membership changed")
    quality = {r["utt_id"]: r["truncation_suspect"] for r in read_jsonl(args.quality)}
    if len(quality) != 19250 or not (cv_ids | test_ids) <= set(quality):
        raise ValueError("Truncation quality file coverage changed")
    e1_summary = json.loads((args.e1_root / "summary.json").read_text(encoding="utf-8"))
    h_summary = json.loads((args.h_root / "summary.json").read_text(encoding="utf-8"))
    e1_best = e1_summary["best_e1"]
    e1_best_score = e1_summary["systems"][e1_best]["cv_op_n_macro_f1"]
    e2_metrics_path = args.e2_root / "AVT" / "metrics.json"
    e2_metrics = json.loads(e2_metrics_path.read_text(encoding="utf-8"))
    e2_score = e2_metrics["cv"]["op_n"]["macro_f1"]
    e_best = (args.e2_root / "AVT", "E2_AVT") if e2_score > e1_best_score else (
        args.e1_root / e1_best, e1_best)
    h_best = h_summary["best_h"]
    systems = {
        "E_best": e_best[0],
        "Q_AV": args.q_cal_root / "Q_AV",
        "Q_A": args.q_cal_root / "Q_A",
        "Q_fusion": args.h_root / "Q_fusion",
        "H_best": args.h_root / h_best,
        "A3_reference": args.e1_root / "E1_A3",
        "V2_reference": args.v2_root,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "folds_sha256": FOLD_SHA, "quality_sha256": QUALITY_SHA,
            "et_manifest_sha256": sha256(args.et_manifest),
            "selected_e": e_best[1], "selected_h": h_best}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    table = {}
    predictions = {}
    sensitivity = {}
    errors = {}
    expected_fold = {s: f["fold"] for f in frozen["folds"] for s in f["outer_holdout_speakers"]}
    for name, root in systems.items():
        metric_path = root / "metrics.json"
        bundle = json.loads(metric_path.read_text(encoding="utf-8"))
        cv_path = root / "oof.jsonl"
        test_path = root / "et_test.jsonl"
        forced = name == "V2_reference"
        cv = prediction_rows(cv_path, cv_ids, "op_n", force_face_error=forced)
        test = prediction_rows(test_path, test_ids, "op_n", force_face_error=forced)
        if any(row["fold"] != expected_fold[row["speaker_id"]] for row in cv):
            raise ValueError(f"OOF speaker fold differs: {name}")
        predictions[name] = {"cv": cv, "et_test": test}
        table[name] = {"selected_source": str(root),
                       "source_metrics_sha256": sha256(metric_path),
                       "cv_op_a": bundle["cv"]["op_a"], "cv_op_n": bundle["cv"]["op_n"],
                       "et_test_op_n": bundle["et_test"]["op_n"],
                       "sims_valid": bundle["sims_valid"]}
        sensitivity[name] = {}
        for split, rows in (("cv", cv), ("et_test", test)):
            filtered = [row for row in rows if not quality[row["id"]]]
            full_score = score(rows)
            filtered_score = score(filtered)
            fields = ("macro_f1", "uar", "wa", "neutral_false_positive_rate",
                      "nonneutral_macro_recall", "ece15")
            sensitivity[name][split] = {
                "full_n": len(rows), "filtered_n": len(filtered),
                "removed": len(rows) - len(filtered),
                "full": {field: full_score[field] for field in fields},
                "filtered": {field: filtered_score[field] for field in fields},
                "filtered_minus_full": {field: filtered_score[field] - full_score[field]
                                        for field in fields}}
        errors[name] = {"cv_confusion_op_n": table[name]["cv_op_n"]["confusion"],
                        "et_test_confusion_op_n": table[name]["et_test_op_n"]["confusion"]}
    old = json.loads(args.x01_metrics.read_text(encoding="utf-8"))["et"]
    table["old_lora_ET_only"] = {form: {key: old[f"old_lora_et_test_{form}"][key]
                                         for key in ("macro_f1", "uar", "wa",
                                                     "neutral_false_positive_rate")}
                                 for form in ("audio", "av")}
    pairs = (("H_best", "E_best"), ("H_best", "Q_AV"),
             ("Q_AV", "Q_A"), ("E_best", "A3_reference"))
    paired = {f"{left}_minus_{right}": paired_speaker_bootstrap(
        predictions[left]["cv"], predictions[right]["cv"]) for left, right in pairs}
    e_rows = predictions["E_best"]["cv"]
    q_rows = predictions["Q_AV"]["cv"]
    categories = defaultdict(int)
    for a, b in zip(e_rows, q_rows, strict=True):
        if (a["id"], a["truth"]) != (b["id"], b["truth"]):
            raise ValueError("E/Q complementarity IDs or labels differ")
        e_ok = a["pred"] == a["truth"]
        q_ok = b["pred"] == b["truth"]
        categories[(e_ok, q_ok)] += 1
    speaker_ranks = {}
    for speaker in sorted(expected_fold):
        by_system = {name: classification([r for r in predictions[name]["cv"]
                                            if r["speaker_id"] == speaker])["macro_f1"]
                     for name in systems}
        speaker_ranks[speaker] = {"scores": by_system,
                                  "ranking": sorted(by_system, key=lambda n: (-by_system[n], n))}
    errors["E_vs_Q_AV_complementarity"] = {
        "n": len(e_rows),
        "both_correct": categories[(True, True)],
        "E_correct_Q_wrong": categories[(True, False)],
        "E_wrong_Q_correct": categories[(False, True)],
        "both_wrong": categories[(False, False)],
        "E_correct_Q_wrong_fraction": categories[(True, False)] / len(e_rows),
        "E_wrong_Q_correct_fraction": categories[(False, True)] / len(e_rows)}
    errors["speaker_ranks_cv_op_n"] = speaker_ranks
    write_json_atomic(args.output_dir / "main_table.json", table)
    write_json_atomic(args.output_dir / "paired_speaker_bootstrap.json", paired)
    write_json_atomic(args.output_dir / "truncation_sensitivity.json", sensitivity)
    write_json_atomic(args.output_dir / "error_analysis.json", errors)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["outputs"] = {name: sha256(args.output_dir / name) for name in (
        "main_table.json", "paired_speaker_bootstrap.json", "truncation_sensitivity.json",
        "error_analysis.json")}
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("e1-root", "e2-root", "q-cal-root", "h-root", "v2-root", "quality",
                 "folds", "et-manifest", "x01-metrics", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
