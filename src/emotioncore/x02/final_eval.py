"""Retrain a frozen X02 head on the CV pool and score ET-test once."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS
from emotioncore.x02 import head_h1, head_h2
from emotioncore.x02.frames_pack import SPECS, read_feature
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.text_t1 import metric_bundle


def load_test_rows(manifest: Path, folds: dict) -> list[dict]:
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    ids = [row["utt_id"] for row in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError("ET-test IDs must be unique and nonempty")
    if any(row["official_split"] != "test" or
           row["speaker_id"] not in folds["excluded_speakers"] for row in rows):
        raise ValueError("ET-test manifest contains an unsealed speaker or split")
    if len(rows) != folds["et_test_count"]:
        raise ValueError("ET-test count differs from frozen folds")
    return rows


def load_test_frames(rows: list[dict], receipts_root: Path, encoder: str):
    max_frames, dimension = SPECS[encoder]
    shape = ((len(rows), 4, max_frames, dimension) if encoder == "A3" else
             (len(rows), max_frames, dimension))
    frames = np.zeros(shape, dtype=np.float16)
    lengths = np.zeros(len(rows), dtype=np.int16)
    for index, row in enumerate(rows):
        path = receipts_root / f"{row['utt_id']}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        input_sha = row["wav" if encoder.startswith("A") else "mp4"]["sha256"]
        if receipt["id"] != row["utt_id"] or receipt["input_sha256"] != input_sha:
            raise ValueError(f"ET-test feature provenance mismatch: {row['utt_id']}")
        if receipt["status"] != "ok":
            continue
        feature_path = Path(receipt["feature_path"])
        if head_h1.sha256(feature_path) != receipt["feature_sha256"]:
            raise ValueError(f"ET-test feature SHA mismatch: {row['utt_id']}")
        source = read_feature(feature_path, encoder)
        if source.shape[-1] != dimension or source.shape[-2] < 1 or not np.isfinite(source).all():
            raise ValueError(f"Invalid ET-test frame feature: {row['utt_id']}")
        indices = np.linspace(0, source.shape[-2] - 1,
                              min(source.shape[-2], max_frames)).round().astype(int)
        frames[index, ..., :len(indices), :] = source[..., indices, :]
        lengths[index] = len(indices)
    return frames, lengths


def prediction_rows(rows: list[dict], probabilities: np.ndarray,
                    available: np.ndarray, encoder: str) -> list[dict]:
    native = "A" if encoder.startswith("A") else "V" if encoder.startswith("V") else "T"
    return [{"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
             "truth_M": row["labels"]["M"], "truth_native": row["labels"][native],
             "probabilities": probability.astype(float).tolist(),
             "status": "ok" if valid else "failed"}
            for row, probability, valid in zip(rows, probabilities, available, strict=True)]


def run(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if head_h1.sha256(args.manifest) != folds["source_manifest_sha256"]:
        raise ValueError("CV manifest differs from frozen folds")
    cv_rows = head_h1.load_rows(args.manifest, folds)
    test_rows = load_test_rows(args.test_manifest, folds)
    if {row["utt_id"] for row in cv_rows} & {row["utt_id"] for row in test_rows}:
        raise ValueError("CV and ET-test IDs overlap")
    cv_result = json.loads(args.cv_result.read_text(encoding="utf-8"))
    if cv_result["encoder"] != args.encoder or cv_result["head"] != args.head:
        raise ValueError("CV result does not match requested encoder/head")
    config = cv_result["final_config"]
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in cv_rows], dtype=np.int64)
    all_ids = np.arange(len(cv_rows))
    if args.head == "h1":
        x_cv, available_cv = head_h1.load_features(cv_rows, args.cv_receipts, args.encoder)
        x_test, available_test = head_h1.load_features(test_rows, args.test_receipts, args.encoder)
        trained = head_h1.train_once(x_cv, y, available_cv, all_ids, None,
                                     config["learning_rate"], config["weight_decay"],
                                     fixed_epochs=config["best_epoch"], device=args.device)
        probabilities = head_h1.predict(trained, x_test, np.arange(len(test_rows)),
                                        available_test, args.device)
        checkpoint = {"state_dict": trained["model"].cpu().state_dict(),
                      "mean": trained["mean"], "std": trained["std"]}
    else:
        if args.cv_pack is None:
            raise ValueError("h2 requires --cv-pack")
        pack = json.loads((args.cv_pack / "pack_meta.json").read_text(encoding="utf-8"))
        if (pack["encoder"] != args.encoder or pack["manifest_sha256"] !=
                folds["source_manifest_sha256"] or pack["folds_sha256"] !=
                head_h1.sha256(args.folds)):
            raise ValueError("CV frame pack differs from frozen protocol")
        if head_h1.sha256(Path(pack["frame_file"])) != pack["frame_sha256"] or \
           head_h1.sha256(Path(pack["lengths_file"])) != pack["lengths_sha256"]:
            raise ValueError("CV frame pack hash mismatch")
        frames_cv = np.load(pack["frame_file"], mmap_mode="r")
        lengths_cv = np.load(pack["lengths_file"])
        frames_test, lengths_test = load_test_frames(test_rows, args.test_receipts,
                                                     args.encoder)
        trained = head_h2.train_once(frames_cv, lengths_cv, y, all_ids, None,
                                     config["learning_rate"], config["weight_decay"],
                                     fixed_epochs=config["best_epoch"], device=args.device,
                                     batch_size=args.batch_size)
        probabilities = head_h2.predict(trained["model"], frames_test, lengths_test,
                                        np.arange(len(test_rows)), args.device,
                                        args.batch_size)
        available_test = lengths_test > 0
        checkpoint = {"state_dict": trained["model"].cpu().state_dict()}
    if probabilities.shape != (len(test_rows), len(LABELS)) or \
       not np.isfinite(probabilities).all() or \
       not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("ET-test probability matrix is invalid")
    predictions = prediction_rows(test_rows, probabilities, available_test, args.encoder)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "et_test.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    checkpoint_path = args.output_dir / "final_head.pt"
    torch.save({**checkpoint, "encoder": args.encoder, "head": args.head,
                "config": config, "folds_sha256": head_h1.sha256(args.folds)}, checkpoint_path)
    result = {"encoder": args.encoder, "head": args.head,
              "final_config": config, "et_test": metric_bundle(predictions,
                  np.asarray([row["speaker_id"] for row in test_rows])),
              "prediction_path": str(prediction_path),
              "prediction_sha256": head_h1.sha256(prediction_path),
              "checkpoint_path": str(checkpoint_path),
              "checkpoint_sha256": head_h1.sha256(checkpoint_path),
              "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    meta = {"git": provenance, "command": sys.argv,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "cv_manifest_sha256": head_h1.sha256(args.manifest),
            "et_test_manifest_sha256": head_h1.sha256(args.test_manifest),
            "folds_sha256": head_h1.sha256(args.folds),
            "cv_result_sha256": head_h1.sha256(args.cv_result),
            "cv_receipts": str(args.cv_receipts), "test_receipts": str(args.test_receipts),
            "cv_pack": str(args.cv_pack) if args.cv_pack else None,
            "sample_count": len(test_rows), "feature_count": int(available_test.sum()),
            "elapsed_s": result["elapsed_s"]}
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--encoder", choices=("A1", "A2", "A3", "V1", "V2",
                                              "V3a", "V3b", "T2"), required=True)
    parser.add_argument("--head", choices=("h1", "h2"), required=True)
    parser.add_argument("--cv-receipts", type=Path, required=True)
    parser.add_argument("--test-receipts", type=Path, required=True)
    parser.add_argument("--cv-pack", type=Path)
    parser.add_argument("--cv-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
