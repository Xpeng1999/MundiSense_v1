"""Leave-one-fold-out calibration for Q-A, Q-V and Q-AV."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, bootstrap_classification, classification, sims_metrics
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.fusion import calibrated_logits, fit_calibration, softmax
from emotioncore.x03.q_manifest import FOLD_SHA, read_jsonl, sha256


def ece15_with_invalid(probabilities: np.ndarray, y: np.ndarray, invalid: np.ndarray) -> float:
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y) & ~invalid
    bins = np.minimum((confidence * 15).astype(int), 14)
    return float(sum(np.mean(bins == k) *
                     abs(float(correct[bins == k].mean()) - float(confidence[bins == k].mean()))
                     for k in range(15) if np.any(bins == k)))


def fit_neutral_bias_with_invalid(logits: np.ndarray, y: np.ndarray,
                                  invalid: np.ndarray, target: float = .15) -> float:
    neutral = y == 0
    if not neutral.any():
        raise ValueError("No neutral rows in fitting folds")
    valid_neutral = neutral & ~invalid
    margin = np.max(logits[valid_neutral, 1:], axis=1) - logits[valid_neutral, 0]
    if not len(margin):
        raise ValueError("No valid neutral rows for OP-N")
    candidates = np.r_[np.nextafter(margin.min(), -np.inf), np.unique(margin),
                       np.nextafter(margin.max(), np.inf)]
    ordered = np.sort(margin)
    miss = (len(margin) - np.searchsorted(ordered, candidates, side="right")) / neutral.sum()
    best = min(range(len(candidates)), key=lambda i: (abs(miss[i] - target), miss[i], candidates[i]))
    return float(candidates[best])


def fit_single(logp: np.ndarray, y: np.ndarray, invalid: np.ndarray) -> tuple[np.ndarray, float]:
    if (~invalid).sum() < 7:
        raise ValueError("Too few valid rows for Q calibration")
    calibration = fit_calibration(logp[~invalid], y[~invalid])
    scores = calibrated_logits(logp, calibration)
    bias = fit_neutral_bias_with_invalid(scores, y, invalid)
    return calibration, bias


def apply_single(logp: np.ndarray, invalid: np.ndarray,
                 calibration: np.ndarray, bias: float) -> tuple[np.ndarray, np.ndarray]:
    scores = calibrated_logits(logp, calibration)
    op_a = softmax(scores)
    scores[:, 0] += bias
    op_n = softmax(scores)
    op_a[invalid] = 1 / 7
    op_n[invalid] = 1 / 7
    return op_a, op_n


def crossfit_single(logp: np.ndarray, y: np.ndarray, folds: np.ndarray,
                    invalid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if sorted(set(folds.tolist())) != list(range(5)):
        raise ValueError("Q OOF needs all five outer folds")
    op_a = np.empty((len(y), 7))
    op_n = np.empty((len(y), 7))
    params = []
    for fold in range(5):
        fit = folds != fold
        held = ~fit
        calibration, bias = fit_single(logp[fit], y[fit], invalid[fit])
        op_a[held], op_n[held] = apply_single(logp[held], invalid[held], calibration, bias)
        params.append({"held_fold": fold, "fit_count": int(fit.sum()),
                       "held_count": int(held.sum()), "valid_fit_count": int((fit & ~invalid).sum()),
                       "calibration": calibration.tolist(), "neutral_bias": bias})
    return op_a, op_n, params


def metrics(probabilities: np.ndarray, y: np.ndarray, invalid: np.ndarray,
            rows: list[dict]) -> dict:
    scored = [{"id": row["utt_id"], "speaker_id": row["speaker_id"],
               "truth": LABELS[int(truth)],
               "pred": None if bad else LABELS[int(np.argmax(probability))]}
              for row, truth, bad, probability in zip(rows, y, invalid, probabilities, strict=True)]
    result = classification(scored)
    result["nonneutral_macro_recall"] = float(np.mean(
        [result["per_class"][label]["recall"] for label in LABELS[1:]]))
    result["ece15"] = ece15_with_invalid(probabilities, y, invalid)
    result["speaker_bootstrap_95"] = bootstrap_classification(scored, group_key="speaker_id")
    result["face_missing_forced_error"] = int(invalid.sum())
    return result


def write_predictions(path: Path, rows: list[dict], a: np.ndarray, n: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row, pa, pn in zip(rows, a, n, strict=True):
            item = {key: row[key] for key in ("utt_id", "speaker_id", "fold", "form",
                                               "status", "face_missing", "face_missing_fallback")}
            for key in ("truth_M", "label"):
                if key in row:
                    item[key] = row[key]
            item["op_a"] = pa.tolist()
            item["op_n"] = pn.tolist()
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return sha256(path)


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.folds) != FOLD_SHA:
        raise ValueError("Frozen fold SHA changed")
    frozen = json.loads(args.folds.read_text(encoding="utf-8"))
    cv_ids = {r["utt_id"] for r in read_jsonl(args.et_manifest)
              if r["speaker_id"] in frozen["cv_pool_speakers"]}
    test_ids = {r["utt_id"] for r in read_jsonl(args.et_manifest) if r["official_split"] == "test"}
    if len(cv_ids) != 16366 or len(test_ids) != 1929:
        raise ValueError("ET membership differs from frozen benchmark")
    expected_fold = {s: f["fold"] for f in frozen["folds"] for s in f["outer_holdout_speakers"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "folds_sha256": FOLD_SHA, "et_manifest_sha256": sha256(args.et_manifest),
            "q_prediction_root": str(args.prediction_root)}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    summary = {}
    for form in ("A", "V", "AV"):
        cv = [r for fold in range(5) for r in read_jsonl(args.prediction_root / f"predict_fold{fold}" / f"{form}.jsonl")]
        cv.sort(key=lambda r: r["utt_id"])
        if (len(cv) != 16366 or {r["utt_id"] for r in cv} != cv_ids or
                any(r["fold"] != expected_fold[r["speaker_id"]] for r in cv)):
            raise ValueError(f"Q-{form} OOF coverage or speaker fold changed")
        test = read_jsonl(args.prediction_root / "predict_final_et_test" / f"{form}.jsonl")
        sims = read_jsonl(args.prediction_root / "predict_final_sims_valid" / f"{form}.jsonl")
        if len(test) != 1929 or {r["utt_id"] for r in test} != test_ids or len(sims) != 647:
            raise ValueError(f"Q-{form} target coverage changed")
        for rows in (cv, test, sims):
            if any(len(r["probabilities"]) != 7 or
                   not np.isclose(sum(r["probabilities"]), 1, atol=1e-5) for r in rows):
                raise ValueError("Invalid Q seven-class probabilities")
        y = np.array([LABELS.index(r["truth_M"]) for r in cv])
        folds = np.array([r["fold"] for r in cv])
        invalid = np.array([r["effective_form"] == "uniform" for r in cv])
        cv_logp = np.log(np.clip(np.array([r["probabilities"] for r in cv]), 1e-12, 1))
        cv_a, cv_n, params = crossfit_single(cv_logp, y, folds, invalid)
        calibration, bias = fit_single(cv_logp, y, invalid)
        output = args.output_dir / f"Q_{form}"
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        paths["oof"] = {"path": str(output / "oof.jsonl"),
                        "sha256": write_predictions(output / "oof.jsonl", cv, cv_a, cv_n)}
        result = {"cv": {"op_a": metrics(cv_a, y, invalid, cv),
                         "op_n": metrics(cv_n, y, invalid, cv)},
                  "parameters_oof": params,
                  "parameters_final": {"calibration": calibration.tolist(), "neutral_bias": bias}}
        for split, rows in (("et_test", test), ("sims_valid", sims)):
            split_invalid = np.array([r["effective_form"] == "uniform" for r in rows])
            logp = np.log(np.clip(np.array([r["probabilities"] for r in rows]), 1e-12, 1))
            a, n = apply_single(logp, split_invalid, calibration, bias)
            path = output / f"{split}.jsonl"
            paths[split] = {"path": str(path), "sha256": write_predictions(path, rows, a, n)}
            if split == "et_test":
                truth = np.array([LABELS.index(r["truth_M"]) for r in rows])
                result[split] = {"op_a": metrics(a, truth, split_invalid, rows),
                                 "op_n": metrics(n, truth, split_invalid, rows)}
            else:
                result[split] = sims_metrics([{"id": r["utt_id"], "label": r["label"],
                                               "s": None if bad else float(p[1] - p[2:6].sum())}
                                              for r, bad, p in zip(rows, split_invalid, n, strict=True)])
        result["artifacts"] = paths
        write_json_atomic(output / "metrics.json", result)
        summary[f"Q_{form}"] = {"cv_op_n_macro_f1": result["cv"]["op_n"]["macro_f1"],
                                "et_test_op_n_macro_f1": result["et_test"]["op_n"]["macro_f1"],
                                "sims_valid_pearson": result["sims_valid"]["pearson"]}
    write_json_atomic(args.output_dir / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--et-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
