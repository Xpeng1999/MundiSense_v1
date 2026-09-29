"""Descriptive fold-0/1 OP-A comparison; never used to select Q settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification
from emotioncore.x02.runtime import write_json_atomic
from emotioncore.x03.q_calibrate import apply_single, fit_single
from emotioncore.x03.q_manifest import read_jsonl, sha256


def _score(rows: list[dict], probabilities: np.ndarray, invalid: np.ndarray) -> dict:
    scored = [{"id": r["utt_id"], "speaker_id": r["speaker_id"],
               "truth": r["truth_M"],
               "pred": None if bad else LABELS[int(np.argmax(p))]}
              for r, p, bad in zip(rows, probabilities, invalid, strict=True)]
    result = classification(scored)
    return {"n": len(rows), "macro_f1": result["macro_f1"],
            "neutral_false_positive_rate": result["neutral_false_positive_rate"]}


def run(root: Path, e1_root: Path, output: Path) -> None:
    by_form = {}
    for form in ("AV", "A"):
        paths = [root / f"predict_fold{i}" / f"{form}.jsonl" for i in (0, 1)]
        rows = [r for path in paths for r in read_jsonl(path)]
        if len(rows) != 6533 or {r["fold"] for r in rows} != {0, 1}:
            raise ValueError("Fold 0/1 Q coverage differs from frozen outer folds")
        by_form[form] = (rows, paths)
    av_ids = {r["utt_id"] for r in by_form["AV"][0]}
    if av_ids != {r["utt_id"] for r in by_form["A"][0]} or len(av_ids) != 6533:
        raise ValueError("Q AV and A must cover identical outer-fold utterances")
    result = {"method": "provisional OP-A: for each held fold, fit Q calibration only on the other completed fold; formal five-fold OP-A remains unchanged",
              "gate": False, "folds": [0, 1], "n": len(av_ids), "systems": {}, "sources": {}}
    for form, (rows, paths) in by_form.items():
        rows.sort(key=lambda r: r["utt_id"])
        result["sources"][f"Q_{form}"] = [{"path": str(path), "sha256": sha256(path)}
                                             for path in paths]
        y = np.array([LABELS.index(r["truth_M"]) for r in rows])
        folds = np.array([r["fold"] for r in rows])
        invalid = np.array([r["effective_form"] == "uniform" for r in rows])
        logp = np.log(np.clip(np.array([r["probabilities"] for r in rows]), 1e-12, 1))
        op_a = np.empty((len(rows), 7))
        for held_fold in (0, 1):
            fit = folds != held_fold
            calibration, bias = fit_single(logp[fit], y[fit], invalid[fit])
            op_a[~fit], _ = apply_single(logp[~fit], invalid[~fit], calibration, bias)
        result["systems"][f"Q_{form}"] = _score(rows, op_a, invalid)
    e1_path = e1_root / "E1_A3_A2_V2_V1_T1r" / "oof.jsonl"
    e1 = {r["id"]: r for r in read_jsonl(e1_path) if r["id"] in av_ids}
    if set(e1) != av_ids:
        raise ValueError("E1 best does not cover the same fold-0/1 utterances")
    rows = sorted(by_form["AV"][0], key=lambda r: r["utt_id"])
    if any(e1[r["utt_id"]]["truth_M"] != r["truth_M"] for r in rows):
        raise ValueError("E1 and Q truth labels differ")
    probabilities = np.array([e1[r["utt_id"]]["op_a"] for r in rows])
    result["systems"]["E1_A3_A2_V2_V1_T1r"] = _score(rows, probabilities,
                                                       np.zeros(len(rows), dtype=bool))
    result["sources"]["E1"] = {"path": str(e1_path), "sha256": sha256(e1_path)}
    write_json_atomic(output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.e1_root, args.output)


if __name__ == "__main__":
    main()
