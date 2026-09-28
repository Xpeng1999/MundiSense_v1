"""Nested speaker-disjoint CV for a class-balanced multinomial linear head."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, bootstrap_classification, classification
from emotioncore.x02.folds import validate_oof
from emotioncore.x02.runtime import git_provenance, write_json_atomic


LEARNING_RATES = (1e-4, 3e-4, 1e-3)
WEIGHT_DECAYS = (0.0, 1e-2)
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 20260928


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(manifest: Path, folds: dict) -> list[dict]:
    excluded = set(folds["excluded_speakers"])
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = [row for row in rows if row["speaker_id"] not in excluded]
    if len(rows) != folds["cv_pool_count"] or len({r["utt_id"] for r in rows}) != len(rows):
        raise ValueError("CV manifest count/IDs differ from frozen folds")
    if {r["speaker_id"] for r in rows} != set(folds["cv_pool_speakers"]):
        raise ValueError("CV manifest speakers differ from frozen folds")
    return rows


def load_features(rows: list[dict], receipts_root: Path, encoder: str) -> tuple[np.ndarray, np.ndarray]:
    features = []
    available = []
    dimension = 3072 if encoder == "A1" else 1024
    for row in rows:
        path = receipts_root / f"{row['utt_id']}.json"
        if not path.is_file():
            raise ValueError(f"Missing feature receipt: {row['utt_id']}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["id"] != row["utt_id"]:
            raise ValueError("Feature receipt ID mismatch")
        source_sha = row["wav"]["sha256"] if encoder == "A1" else receipt.get("input_sha256")
        if receipt.get("input_sha256") != source_sha:
            raise ValueError("Feature receipt input hash mismatch")
        if receipt["status"] != "ok":
            features.append(np.zeros(dimension, dtype=np.float32))
            available.append(False)
            continue
        feature_file = Path(receipt["feature_path"])
        if sha256(feature_file) != receipt["feature_sha256"]:
            raise ValueError(f"Feature file hash mismatch: {row['utt_id']}")
        if encoder == "A1":
            with np.load(feature_file) as data:
                frames = data["frames"]
                vector = np.concatenate((data["utterance"], frames.mean(0), frames.std(0)))
        elif encoder == "T2":
            vector = np.load(feature_file)
        else:
            raise ValueError(f"Unsupported h1 encoder: {encoder}")
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid feature vector: {row['utt_id']}")
        features.append(vector.astype(np.float32))
        available.append(True)
    return np.stack(features), np.asarray(available, dtype=bool)


def scores(truth: np.ndarray, pred: np.ndarray, available: np.ndarray) -> dict:
    rows = [{"truth": LABELS[int(t)], "pred": LABELS[int(p)] if a else None}
            for t, p, a in zip(truth, pred, available)]
    result = classification(rows)
    return {"macro_f1": result["macro_f1"], "uar": result["uar"]}


def train_once(x: np.ndarray, y: np.ndarray, available: np.ndarray,
               train_ids: np.ndarray, val_ids: np.ndarray | None,
               learning_rate: float, weight_decay: float,
               *, fixed_epochs: int | None = None, seed: int = SEED,
               device: str = "cuda:0") -> dict:
    import torch

    valid_train = train_ids[available[train_ids]]
    if len(valid_train) == 0:
        raise ValueError("No training features available")
    mean = x[valid_train].mean(0, dtype=np.float64).astype(np.float32)
    std = x[valid_train].std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    x_train = torch.from_numpy(((x[valid_train] - mean) / std).astype(np.float32)).to(device)
    y_train = torch.from_numpy(y[valid_train]).to(device)
    x_val = None
    if val_ids is not None:
        x_val = torch.from_numpy(((x[val_ids] - mean) / std).astype(np.float32)).to(device)
    counts = np.bincount(y[valid_train], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("At least one class is absent from the training fold")
    weights = torch.tensor(len(valid_train) / (len(LABELS) * counts),
                           dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(x.shape[1], len(LABELS), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_epochs = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    best = None
    best_epoch = 0
    best_state = None
    no_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(valid_train), generator=generator)
        for batch in order.split(1024):
            indices = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_train[indices]),
                                                     y_train[indices], weight=weights)
            loss.backward()
            optimizer.step()
        if val_ids is None:
            continue
        model.eval()
        with torch.inference_mode():
            predicted = model(x_val).argmax(1).cpu().numpy()
        result = scores(y[val_ids], predicted, available[val_ids])
        rank = (result["macro_f1"], result["uar"])
        if best is None or rank > best:
            best, best_epoch, no_improvement = rank, epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if fixed_epochs is None and no_improvement >= PATIENCE:
                break
    if val_ids is not None and best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "mean": mean, "std": std,
            "best_epoch": best_epoch if val_ids is not None else max_epochs,
            "validation": {"macro_f1": best[0], "uar": best[1]} if best else None,
            "class_counts": counts.tolist()}


def predict(trained: dict, x: np.ndarray, indices: np.ndarray,
            available: np.ndarray, device: str) -> np.ndarray:
    import torch

    transformed = torch.from_numpy(((x[indices] - trained["mean"]) /
                                    trained["std"]).astype(np.float32)).to(device)
    trained["model"].eval()
    with torch.inference_mode():
        probabilities = torch.softmax(trained["model"](transformed), dim=1).cpu().numpy()
    probabilities[~available[indices]] = 1.0 / len(LABELS)
    return probabilities


def select_config(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda c: (c["validation"]["macro_f1"],
                                          c["validation"]["uar"],
                                          -c["learning_rate"], -c["weight_decay"]))


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if sha256(args.manifest) != folds["source_manifest_sha256"]:
        raise ValueError("Manifest differs from frozen folds")
    rows = load_rows(args.manifest, folds)
    x, available = load_features(rows, args.receipts_root, args.encoder)
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in rows], dtype=np.int64)
    speakers = np.asarray([row["speaker_id"] for row in rows])
    oof: list[dict] = []
    fold_records = []
    config_history: dict[tuple[float, float], list[dict]] = {
        (lr, wd): [] for lr in LEARNING_RATES for wd in WEIGHT_DECAYS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest), "folds_sha256": sha256(args.folds),
            "encoder": args.encoder, "head": "h1", "optimizer": "AdamW",
            "class_balance": "inverse_frequency_train_fold", "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "seed": SEED,
            "feature_count": int(available.sum()), "sample_count": len(rows),
            "feature_receipts_root": str(args.receipts_root),
            "learning_rates": LEARNING_RATES, "weight_decays": WEIGHT_DECAYS}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    for fold in folds["folds"]:
        outer = np.flatnonzero(np.isin(speakers, fold["outer_holdout_speakers"]))
        inner_val = np.flatnonzero(np.isin(speakers, fold["inner_validation_speakers"]))
        inner_train = np.flatnonzero(np.isin(speakers, fold["inner_train_speakers"]))
        if len(outer) != fold["outer_holdout_count"] or len(inner_val) != fold["inner_validation_count"]:
            raise ValueError("Frozen fold sample count mismatch")
        candidates = []
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                trained = train_once(x, y, available, inner_train, inner_val, lr, wd,
                                     seed=SEED + fold["fold"], device=args.device)
                candidate = {"learning_rate": lr, "weight_decay": wd,
                             "best_epoch": trained["best_epoch"],
                             "validation": trained["validation"]}
                candidates.append(candidate)
                config_history[(lr, wd)].append(candidate)
                del trained
        chosen = select_config(candidates)
        train_12 = np.flatnonzero(~np.isin(speakers, fold["outer_holdout_speakers"]))
        trained = train_once(x, y, available, train_12, None,
                             chosen["learning_rate"], chosen["weight_decay"],
                             fixed_epochs=chosen["best_epoch"],
                             seed=SEED + fold["fold"], device=args.device)
        probabilities = predict(trained, x, outer, available, args.device)
        for index, probability in zip(outer, probabilities):
            row = rows[index]
            status = "ok" if available[index] else "failed"
            oof.append({"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                        "fold": fold["fold"], "truth_M": row["labels"]["M"],
                        "truth_native": row["labels"]["A" if args.encoder == "A1" else "T"],
                        "probabilities": probability.astype(float).tolist(),
                        "status": status})
        fold_records.append({"fold": fold["fold"], "inner_grid": candidates,
                             "selected": chosen, "outer_count": len(outer),
                             "outer_feature_count": int(available[outer].sum())})
        write_json_atomic(args.output_dir / "progress.json",
                          {"completed_folds": len(fold_records), "total_folds": len(folds["folds"])})
    expected = {row["utt_id"] for row in rows}
    validate_oof(expected, oof)
    oof.sort(key=lambda row: row["utt_id"])
    predictions_path = args.output_dir / "oof.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in oof:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    def scored(truth_key: str) -> list[dict]:
        return [{"id": row["utt_id"], "speaker_id": row["speaker_id"],
                 "truth": row[truth_key],
                 "pred": LABELS[int(np.argmax(row["probabilities"]))]
                 if row["status"] == "ok" else None} for row in oof]
    primary, native = scored("truth_M"), scored("truth_native")
    overall = {"M": classification(primary), "native": classification(native),
               "M_speaker_bootstrap_95": bootstrap_classification(primary,
                                            group_key="speaker_id"),
               "native_speaker_bootstrap_95": bootstrap_classification(native,
                                            group_key="speaker_id"),
               "M_by_speaker": {speaker: classification([r for r in primary
                                                if r["speaker_id"] == speaker])
                                for speaker in sorted(set(speakers))},
               "native_by_speaker": {speaker: classification([r for r in native
                                                if r["speaker_id"] == speaker])
                                for speaker in sorted(set(speakers))}}
    final_candidates = []
    for (lr, wd), history in config_history.items():
        final_candidates.append({"learning_rate": lr, "weight_decay": wd,
                                 "validation": {"macro_f1": statistics.mean(
                                     item["validation"]["macro_f1"] for item in history),
                                     "uar": statistics.mean(item["validation"]["uar"] for item in history)},
                                 "best_epoch": int(statistics.median_low(
                                     item["best_epoch"] for item in history))})
    final_config = select_config(final_candidates)
    result = {"encoder": args.encoder, "head": "h1", "folds": fold_records,
              "final_grid": final_candidates, "final_config": final_config,
              "metrics": overall, "oof_path": str(predictions_path),
              "oof_sha256": sha256(predictions_path), "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["elapsed_s"] = result["elapsed_s"]
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--receipts-root", type=Path, required=True)
    parser.add_argument("--encoder", choices=("A1", "T2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
