"""Cross-fitted E2 calibration and fixed neutral operating point."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, bootstrap_classification, classification
from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.e2 import CONDITIONS
from emotioncore.x03.fusion import calibrated_logits, fit_calibration, fit_neutral_bias, softmax


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def ece15(probabilities: np.ndarray, truth: np.ndarray, failed: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == truth) & ~failed
    bins = np.minimum((confidence * 15).astype(int), 14)
    return float(sum(np.mean(bins == k) *
                     abs(float(np.mean(correct[bins == k])) - float(np.mean(confidence[bins == k])))
                     for k in range(15) if np.any(bins == k)))


def score(rows: list[dict], probabilities: np.ndarray, condition: str) -> dict:
    truth = np.asarray([LABELS.index(row["truth_M"]) for row in rows])
    failed = np.asarray([condition == "V" and row["status"] == "face_missing"
                         for row in rows])
    scored = [{"id": row["utt_id"], "speaker_id": row["speaker_id"],
               "truth": row["truth_M"],
               "pred": None if miss else LABELS[int(np.argmax(probability))]}
              for row, probability, miss in zip(rows, probabilities, failed)]
    result = classification(scored)
    result["nonneutral_macro_recall"] = float(np.mean(
        [result["per_class"][label]["recall"] for label in LABELS[1:]]))
    result["ece15"] = ece15(probabilities, truth, failed)
    result["M_speaker_bootstrap_95"] = bootstrap_classification(
        scored, group_key="speaker_id")
    result["face_missing_forced_error"] = int(failed.sum())
    return result


def calibrated_oof(rows: list[dict], condition: str):
    y = np.asarray([LABELS.index(row["truth_M"]) for row in rows])
    folds = np.asarray([row["fold"] for row in rows])
    if sorted(set(folds)) != list(range(5)):
        raise ValueError("E2 OOF must contain the five frozen folds")
    base = np.asarray([row["probabilities"] for row in rows], dtype=float)
    if base.shape != (len(rows), 7) or not np.allclose(base.sum(axis=1), 1, atol=1e-5):
        raise ValueError("E2 base probability shape or normalization invalid")
    logp = np.log(np.clip(base, 1e-12, 1))
    op_a, op_n = np.empty_like(base), np.empty_like(base)
    parameters = []
    for fold in range(5):
        fit = folds != fold
        held = folds == fold
        calibration = fit_calibration(logp[fit], y[fit])
        bias = fit_neutral_bias(calibrated_logits(logp[fit], calibration), y[fit])
        scores = calibrated_logits(logp[held], calibration)
        op_a[held] = softmax(scores)
        scores[:, 0] += bias
        op_n[held] = softmax(scores)
        parameters.append({"fold": fold, "fit_count": int(fit.sum()),
                           "held_count": int(held.sum()),
                           "calibration": calibration.tolist(), "neutral_bias": bias})
    missing = np.asarray([condition == "V" and row["status"] == "face_missing"
                          for row in rows])
    op_a[missing] = 1 / 7
    op_n[missing] = 1 / 7
    return op_a, op_n, parameters


def apply_final(rows_oof: list[dict], rows_target: list[dict], condition: str):
    y = np.asarray([LABELS.index(row["truth_M"]) for row in rows_oof])
    oof_logp = np.log(np.clip(np.asarray([r["probabilities"] for r in rows_oof]), 1e-12, 1))
    calibration = fit_calibration(oof_logp, y)
    bias = fit_neutral_bias(calibrated_logits(oof_logp, calibration), y)
    target = np.log(np.clip(np.asarray([r["probabilities"] for r in rows_target]), 1e-12, 1))
    scores = calibrated_logits(target, calibration)
    op_a = softmax(scores)
    scores[:, 0] += bias
    op_n = softmax(scores)
    missing = np.asarray([condition == "V" and row["status"] == "face_missing"
                          for row in rows_target])
    op_a[missing] = 1 / 7
    op_n[missing] = 1 / 7
    return op_a, op_n, {"calibration": calibration.tolist(), "neutral_bias": bias}


def write_predictions(path: Path, rows: list[dict], condition: str,
                      op_a: np.ndarray, op_n: np.ndarray):
    with path.open("w", encoding="utf-8") as stream:
        for row, a, n in zip(rows, op_a, op_n):
            item = {"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                    "condition": condition, "status": row["status"],
                    "op_a": a.tolist(), "op_n": n.tolist()}
            for key in ("fold", "truth_M"):
                if key in row:
                    item[key] = row[key]
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def run(args):
    git = git_provenance(Path.cwd(), args.allow_dirty)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_jsonl(args.manifest)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    cv_ids = {row["utt_id"] for row in manifest
              if row["speaker_id"] in folds["cv_pool_speakers"]}
    test_ids = {row["utt_id"] for row in manifest if row["official_split"] == "test"}
    expected_folds = {speaker: f["fold"] for f in folds["folds"]
                      for speaker in f["outer_holdout_speakers"]}
    summary = {}
    for condition in CONDITIONS:
        rows = read_jsonl(args.cv_dir / f"oof_{condition}.jsonl")
        if (len(rows) != 16366 or {row["utt_id"] for row in rows} != cv_ids
                or any(expected_folds[row["speaker_id"]] != row["fold"] for row in rows)):
            raise ValueError("E2 OOF coverage or speaker-fold mapping invalid")
        op_a, op_n, params = calibrated_oof(rows, condition)
        target_dir = args.output_dir / condition
        target_dir.mkdir(exist_ok=True)
        oof_path = target_dir / "oof.jsonl"
        write_predictions(oof_path, rows, condition, op_a, op_n)
        cv = {"op_a": score(rows, op_a, condition),
              "op_n": score(rows, op_n, condition)}
        test_path = args.target_dir / f"et_test_{condition}.jsonl"
        test_rows = read_jsonl(test_path) if test_path.exists() else None
        test = None
        artifacts = {"oof": {"path": str(oof_path), "sha256": sha256(oof_path)}}
        if test_rows is not None:
            if len(test_rows) != 1929 or {row["utt_id"] for row in test_rows} != test_ids:
                raise ValueError("E2 ET-test coverage invalid")
            test_a, test_n, final = apply_final(rows, test_rows, condition)
            output = target_dir / "et_test.jsonl"
            write_predictions(output, test_rows, condition, test_a, test_n)
            test = {"op_a": score(test_rows, test_a, condition),
                    "op_n": score(test_rows, test_n, condition)}
            artifacts["et_test"] = {"path": str(output), "sha256": sha256(output)}
            write_json_atomic(target_dir / "parameters_final.json", final)
        write_json_atomic(target_dir / "parameters_cv.json", {"folds": params})
        write_json_atomic(target_dir / "metrics.json",
                          {"condition": condition, "cv": cv, "et_test": test,
                           "artifacts": artifacts})
        summary[condition] = {"cv_op_n_macro_f1": cv["op_n"]["macro_f1"],
                              "cv_op_n_neutral_miss_rate": cv["op_n"]["neutral_false_positive_rate"],
                              "cv_face_missing_errors": cv["op_n"]["face_missing_forced_error"]}
    write_json_atomic(args.output_dir / "summary.json", summary)
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": git, "command": sys.argv,
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "folds_sha256": sha256(args.folds),
                       "cv_results_sha256": sha256(args.cv_dir / "results.json"),
                       "summary_sha256": sha256(args.output_dir / "summary.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
