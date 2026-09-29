"""Calibrate the frozen X02 V2 visual reference under X03 missing-face rules."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, sims_metrics
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA
from emotioncore.x03.fusion import load_sources
from emotioncore.x03.q_calibrate import apply_single, crossfit_single, fit_single, metrics
from emotioncore.x03.q_manifest import FOLD_SHA, sha256


def write_rows(path: Path, ids: list[str], source: list[dict],
               a: np.ndarray, n: np.ndarray, invalid: np.ndarray, split: str) -> str:
    with path.open("w", encoding="utf-8") as stream:
        for uid, row, pa, pn, bad in zip(ids, source, a, n, invalid, strict=True):
            item = {"id": uid, "split": split, "status": "face_missing" if bad else row.get("status", "ok"),
                    "op_a": pa.tolist(), "op_n": pn.tolist(), "face_missing": bool(bad)}
            for key in ("speaker_id", "fold", "truth_M", "label"):
                if key in row:
                    item[key] = row[key]
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    return sha256(path)


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.folds) != FOLD_SHA or sha256(args.face_policy) != FACE_SHA:
        raise ValueError("Frozen X03 policy SHA changed")
    face_missing = set(json.loads(args.face_policy.read_text(encoding="utf-8"))["face_missing_utt_ids"])
    ids, logp, rows, sources = load_sources(args.artifact_index, ("V2_h2",), "oof", face_missing)
    if len(ids) != 16366:
        raise ValueError("V2 OOF coverage changed")
    y = np.array([LABELS.index(r["truth_M"]) for r in rows])
    folds = np.array([r["fold"] for r in rows])
    invalid = np.array([uid in face_missing for uid in ids])
    a, n, params = crossfit_single(logp[:, 0], y, folds, invalid)
    calibration, bias = fit_single(logp[:, 0], y, invalid)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "folds_sha256": FOLD_SHA, "face_policy_sha256": FACE_SHA,
            "artifact_index_sha256": sha256(args.artifact_index), "source": sources}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    paths = {"oof": {"path": str(args.output_dir / "oof.jsonl"),
                     "sha256": write_rows(args.output_dir / "oof.jsonl", ids, rows, a, n,
                                          invalid, "oof")}}
    result = {"cv": {"op_a": metrics(a, y, invalid, [{"utt_id": uid, **r} for uid, r in zip(ids, rows)]),
                     "op_n": metrics(n, y, invalid, [{"utt_id": uid, **r} for uid, r in zip(ids, rows)])},
              "parameters_oof": params,
              "parameters_final": {"calibration": calibration.tolist(), "neutral_bias": bias}}
    for split in ("et_test", "sims_valid"):
        target_ids, target_logp, target_rows, target_sources = load_sources(
            args.artifact_index, ("V2_h2",), split, face_missing)
        bad = np.array([uid in face_missing if split == "et_test" else row.get("status") != "ok"
                        for uid, row in zip(target_ids, target_rows, strict=True)])
        pa, pn = apply_single(target_logp[:, 0], bad, calibration, bias)
        path = args.output_dir / f"{split}.jsonl"
        paths[split] = {"path": str(path),
                        "sha256": write_rows(path, target_ids, target_rows, pa, pn, bad, split)}
        if split == "et_test":
            truth = np.array([LABELS.index(r["truth_M"]) for r in target_rows])
            wrapped = [{"utt_id": uid, **r} for uid, r in zip(target_ids, target_rows)]
            result[split] = {"op_a": metrics(pa, truth, bad, wrapped),
                             "op_n": metrics(pn, truth, bad, wrapped)}
        else:
            result[split] = sims_metrics([
                {"id": uid, "label": r["label"],
                 "s": None if missing else float(p[1] - p[2:6].sum())}
                for uid, r, p, missing in zip(target_ids, target_rows, pn, bad, strict=True)])
        result[f"{split}_source"] = target_sources
    result["artifacts"] = paths
    write_json_atomic(args.output_dir / "metrics.json", result)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["metrics_sha256"] = sha256(args.output_dir / "metrics.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
