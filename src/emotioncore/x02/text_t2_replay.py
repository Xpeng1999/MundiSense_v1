"""Evaluate ASR text with T2 heads trained exclusively on annotated text."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, bootstrap_classification, classification
from emotioncore.x02.folds import validate_oof
from emotioncore.x02.head_h1 import SEED, load_features, load_rows, predict, sha256, train_once
from emotioncore.x02.runtime import git_provenance, write_json_atomic


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if sha256(args.manifest) != folds["source_manifest_sha256"]:
        raise ValueError("Manifest differs from frozen folds")
    rows = load_rows(args.manifest, folds)
    x_o, available_o = load_features(rows, args.receipts_o, "T2")
    x_r, available_r = load_features(rows, args.receipts_r, "T2")
    original = json.loads(args.original_results.read_text(encoding="utf-8"))
    if original["encoder"] != "T2" or original["head"] != "h1":
        raise ValueError("Original result is not the annotated T2 h1 CV")
    original_oof = {}
    with Path(original["oof_path"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            original_oof[row["utt_id"]] = np.asarray(row["probabilities"])
    if len(original_oof) != len(rows):
        raise ValueError("Original annotated OOF is incomplete")
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in rows], dtype=np.int64)
    speakers = np.asarray([row["speaker_id"] for row in rows])
    output = []
    checks = []
    for fold, recorded in zip(folds["folds"], original["folds"]):
        if fold["fold"] != recorded["fold"]:
            raise ValueError("Original T2 fold selection differs from frozen folds")
        outer = np.flatnonzero(np.isin(speakers, fold["outer_holdout_speakers"]))
        train_12 = np.flatnonzero(~np.isin(speakers, fold["outer_holdout_speakers"]))
        chosen = recorded["selected"]
        trained = train_once(x_o, y, available_o, train_12, None,
                             chosen["learning_rate"], chosen["weight_decay"],
                             fixed_epochs=chosen["best_epoch"],
                             seed=SEED + fold["fold"], device=args.device)
        p_o = predict(trained, x_o, outer, available_o, args.device)
        p_r = predict(trained, x_r, outer, available_r, args.device)
        difference = max(float(np.max(np.abs(probability - original_oof[rows[index]["utt_id"]])))
                         for index, probability in zip(outer, p_o))
        if difference > 1e-5:
            raise ValueError(f"Replayed annotated OOF differs for fold {fold['fold']}: {difference}")
        checks.append({"fold": fold["fold"], "maximum_oof_probability_difference": difference,
                       "training_source": "annotated_only"})
        for index, probability in zip(outer, p_r):
            row = rows[index]
            output.append({"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                           "fold": fold["fold"], "source": "r",
                           "truth_M": row["labels"]["M"],
                           "truth_native": row["labels"]["T"],
                           "probabilities": probability.astype(float).tolist(),
                           "status": "ok" if available_r[index] else "failed"})
    validate_oof({row["utt_id"] for row in rows}, output)
    output.sort(key=lambda row: row["utt_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / "oof_r.jsonl"
    with oof_path.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    def scored(truth_key: str) -> list[dict]:
        return [{"id": row["utt_id"], "speaker_id": row["speaker_id"],
                 "truth": row[truth_key],
                 "pred": LABELS[int(np.argmax(row["probabilities"]))]
                 if row["status"] == "ok" else None} for row in output]
    primary, native = scored("truth_M"), scored("truth_native")
    metrics = {"M": classification(primary), "native": classification(native),
               "M_speaker_bootstrap_95": bootstrap_classification(primary, group_key="speaker_id"),
               "native_speaker_bootstrap_95": bootstrap_classification(native, group_key="speaker_id"),
               "M_by_speaker": {speaker: classification([r for r in primary if r["speaker_id"] == speaker])
                                for speaker in sorted(set(speakers))},
               "native_by_speaker": {speaker: classification([r for r in native if r["speaker_id"] == speaker])
                                for speaker in sorted(set(speakers))}}
    result = {"encoder": "T2", "head": "h1", "training_source": "o",
              "evaluation_source": "r", "original_results": str(args.original_results),
              "original_results_sha256": sha256(args.original_results),
              "replay_checks": checks, "metrics": metrics,
              "oof_path": str(oof_path), "oof_sha256": sha256(oof_path),
              "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": provenance, "command": sys.argv,
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "manifest_sha256": sha256(args.manifest),
                       "folds_sha256": sha256(args.folds),
                       "training_source": "annotated_only",
                       "evaluation_source": "SenseVoice_ASR",
                       "replay_max_difference": max(c["maximum_oof_probability_difference"]
                                                    for c in checks)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--receipts-o", type=Path, required=True)
    parser.add_argument("--receipts-r", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
