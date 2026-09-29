"""Apply the annotated-trained final T2 head to ET-test ASR features."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.final_eval import load_test_rows, prediction_rows
from emotioncore.x02.head_h1 import load_features, predict, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.text_t1 import metric_bundle


def run(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    rows = load_test_rows(args.test_manifest, folds)
    annotated = json.loads(args.annotated_result.read_text(encoding="utf-8"))
    if annotated["encoder"] != "T2" or annotated["head"] != "h1":
        raise ValueError("Expected final annotated-trained T2 h1 result")
    checkpoint_path = Path(annotated["checkpoint_path"])
    if sha256(checkpoint_path) != annotated["checkpoint_sha256"]:
        raise ValueError("Annotated final head checksum differs from receipt")
    # This checkpoint is produced by final_eval and SHA-verified above; its
    # normalization arrays require pickle support in recent PyTorch releases.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["encoder"] != "T2" or checkpoint["head"] != "h1" or \
       checkpoint["folds_sha256"] != sha256(args.folds):
        raise ValueError("Annotated final head has wrong provenance")
    model = torch.nn.Linear(1024, len(LABELS), device=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    features, available = load_features(rows, args.asr_receipts, "T2")
    trained = {"model": model, "mean": checkpoint["mean"], "std": checkpoint["std"]}
    probabilities = predict(trained, features, np.arange(len(rows)), available, args.device)
    predictions = prediction_rows(rows, probabilities, available, "T2")
    for prediction in predictions:
        prediction["source"] = "r"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "et_test_r.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"encoder": "T2", "source": "r", "trained_on": "o",
              "annotated_checkpoint_sha256": sha256(checkpoint_path),
              "et_test": metric_bundle(predictions,
                  np.asarray([row["speaker_id"] for row in rows])),
              "prediction_path": str(path), "prediction_sha256": sha256(path),
              "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": provenance, "command": sys.argv,
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "test_manifest_sha256": sha256(args.test_manifest),
                       "folds_sha256": sha256(args.folds),
                       "annotated_result_sha256": sha256(args.annotated_result),
                       "asr_receipts": str(args.asr_receipts),
                       "elapsed_s": result["elapsed_s"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--annotated-result", type=Path, required=True)
    parser.add_argument("--asr-receipts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
