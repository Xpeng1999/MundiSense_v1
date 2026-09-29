"""Score sealed ET-test with the annotated-only final T1 checkpoint."""
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
from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.text_t1 import make_model, metric_bundle, predict


def load_test_tokens(path: Path, meta_path: Path, rows: list[dict], source: str,
                     manifest_sha: str, folds_sha: str) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta["source"] != source or meta["split"] != "test" or
            meta["manifest_sha256"] != manifest_sha or
            meta["folds_sha256"] != folds_sha or
            sha256(path) != meta["tokens_sha256"]):
        raise ValueError(f"T1 {source} ET-test token cache provenance mismatch")
    with np.load(path) as data:
        result = {key: data[key].copy() for key in data.files}
    if (list(result["utt_ids"]) != [row["utt_id"] for row in rows] or
            result["input_ids"].shape != (len(rows), 128) or
            result["attention_mask"].shape != (len(rows), 128)):
        raise ValueError(f"T1 {source} ET-test token IDs or shape mismatch")
    return result


def run(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    rows = load_test_rows(args.test_manifest, folds)
    manifest_sha, folds_sha = sha256(args.test_manifest), sha256(args.folds)
    result_cv = json.loads(args.cv_result.read_text(encoding="utf-8"))
    checkpoint_path = Path(result_cv["final_checkpoint"]["path"])
    if sha256(checkpoint_path) != result_cv["final_checkpoint"]["sha256"]:
        raise ValueError("T1 final checkpoint hash differs from CV receipt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != result_cv["final_config"] or \
       tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("T1 final checkpoint does not match CV selection")
    model = make_model(args.model_dir, args.device, 20260928)
    model.load_state_dict(checkpoint["state_dict"])
    speakers = np.asarray([row["speaker_id"] for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for source in ("o", "r"):
        path = args.tokens_dir / f"T1_{source}_test.npz"
        tokens = load_test_tokens(path, args.tokens_dir / f"T1_{source}_test_meta.json",
                                  rows, source, manifest_sha, folds_sha)
        probabilities = predict(model, tokens, np.arange(len(rows)), args.device,
                                args.batch_size)
        predictions = prediction_rows(rows, probabilities, tokens["failed"] == 0, "T1")
        for prediction in predictions:
            prediction["source"] = source
        output_path = args.output_dir / f"et_test_{source}.jsonl"
        with output_path.open("w", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        outputs[source] = {"metrics": metric_bundle(predictions, speakers),
                           "prediction_path": str(output_path),
                           "prediction_sha256": sha256(output_path),
                           "token_sha256": sha256(path)}
    result = {"encoder": "T1_chinese_roberta_wwm_ext_large",
              "checkpoint_sha256": sha256(checkpoint_path),
              "cv_result_sha256": sha256(args.cv_result),
              "et_test": outputs, "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": provenance, "command": sys.argv,
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "test_manifest_sha256": manifest_sha, "folds_sha256": folds_sha,
                       "sample_count": len(rows), "elapsed_s": result["elapsed_s"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cv-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
