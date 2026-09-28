"""Nested speaker CV for temporal attention and a two-layer MLP."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, bootstrap_classification, classification
from emotioncore.x02.folds import validate_oof
from emotioncore.x02.head_h1 import LEARNING_RATES, WEIGHT_DECAYS, SEED, load_rows, scores, select_config, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


MAX_EPOCHS = 50
PATIENCE = 5
DROPOUT = 0.25


def build_model(dimension: int, layered: bool, device: str):
    import torch

    class AttentionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_logits = torch.nn.Parameter(torch.zeros(4)) if layered else None
            self.norm = torch.nn.LayerNorm(dimension)
            self.attention = torch.nn.Linear(dimension, 1)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(dimension, 256), torch.nn.GELU(),
                torch.nn.Dropout(DROPOUT), torch.nn.Linear(256, len(LABELS)))

        def forward(self, frames, lengths):
            if self.layer_logits is not None:
                weights = torch.softmax(self.layer_logits, dim=0)
                frames = (frames * weights.view(1, 4, 1, 1)).sum(dim=1)
            frames = self.norm(frames)
            scores_ = self.attention(frames).squeeze(-1)
            mask = torch.arange(frames.shape[1], device=frames.device)[None, :] >= lengths[:, None]
            scores_ = scores_.masked_fill(mask, -1e4)
            weights = torch.softmax(scores_, dim=1)
            pooled = (frames * weights.unsqueeze(-1)).sum(dim=1)
            return self.classifier(pooled)

    return AttentionHead().to(device)


def batches(indices: np.ndarray, size: int, seed: int | None = None):
    if seed is not None:
        indices = np.random.default_rng(seed).permutation(indices)
    for offset in range(0, len(indices), size):
        yield indices[offset:offset + size]


def tensor_batch(frames: np.ndarray, lengths: np.ndarray, indices: np.ndarray, device: str):
    import torch

    x = torch.from_numpy(np.asarray(frames[indices], dtype=np.float32)).to(device)
    n = torch.from_numpy(lengths[indices].astype(np.int64)).to(device)
    return x, n


def predict(model, frames: np.ndarray, lengths: np.ndarray,
            indices: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    import torch

    probabilities = np.full((len(indices), len(LABELS)), 1 / len(LABELS), dtype=np.float32)
    valid = lengths[indices] > 0
    valid_positions = np.flatnonzero(valid)
    model.eval()
    with torch.inference_mode():
        for positions in batches(valid_positions, batch_size):
            x, n = tensor_batch(frames, lengths, indices[positions], device)
            probabilities[positions] = torch.softmax(model(x, n), dim=1).cpu().numpy()
    return probabilities


def train_once(frames: np.ndarray, lengths: np.ndarray, y: np.ndarray,
               train_ids: np.ndarray, val_ids: np.ndarray | None,
               learning_rate: float, weight_decay: float,
               *, fixed_epochs: int | None = None, seed: int = SEED,
               device: str = "cuda:0", batch_size: int = 128) -> dict:
    import torch

    valid_train = train_ids[lengths[train_ids] > 0]
    if not len(valid_train):
        raise ValueError("No training frame features available")
    counts = np.bincount(y[valid_train], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("At least one class is absent from the training fold")
    weights = torch.tensor(len(valid_train) / (len(LABELS) * counts),
                           dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    layered = frames.ndim == 4
    model = build_model(frames.shape[-1], layered, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    best = None
    best_epoch = 0
    best_state = None
    no_improvement = 0
    for epoch in range(1, fixed_epochs + 1 if fixed_epochs is not None else MAX_EPOCHS + 1):
        model.train()
        for batch in batches(valid_train, batch_size, seed + epoch):
            x, n = tensor_batch(frames, lengths, batch, device)
            labels = torch.from_numpy(y[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x, n), labels, weight=weights)
            loss.backward()
            optimizer.step()
        if val_ids is None:
            continue
        probabilities = predict(model, frames, lengths, val_ids, device, batch_size)
        result = scores(y[val_ids], probabilities.argmax(1), lengths[val_ids] > 0)
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
    return {"model": model,
            "best_epoch": best_epoch if val_ids is not None else fixed_epochs,
            "validation": {"macro_f1": best[0], "uar": best[1]} if best else None,
            "layer_weights": torch.softmax(model.layer_logits, dim=0).detach().cpu().tolist()
                             if layered else None, "class_counts": counts.tolist()}


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    pack = json.loads((args.pack_dir / "pack_meta.json").read_text(encoding="utf-8"))
    if (sha256(args.manifest) != folds["source_manifest_sha256"] or
            pack["manifest_sha256"] != folds["source_manifest_sha256"] or
            pack["folds_sha256"] != sha256(args.folds) or
            pack["encoder"] != args.encoder):
        raise ValueError("Packed features do not match the frozen protocol")
    rows = load_rows(args.manifest, folds)
    frames_path = Path(pack["frame_file"])
    lengths_path = Path(pack["lengths_file"])
    if sha256(frames_path) != pack["frame_sha256"] or sha256(lengths_path) != pack["lengths_sha256"]:
        raise ValueError("Packed feature sha256 mismatch")
    frames = np.load(frames_path, mmap_mode="r")
    lengths = np.load(lengths_path)
    if len(frames) != len(rows) or len(lengths) != len(rows):
        raise ValueError("Packed feature sample count mismatch")
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in rows], dtype=np.int64)
    speakers = np.asarray([row["speaker_id"] for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest), "folds_sha256": sha256(args.folds),
            "pack_meta_sha256": sha256(args.pack_dir / "pack_meta.json"),
            "encoder": args.encoder, "head": "h2", "optimizer": "AdamW",
            "class_balance": "inverse_frequency_train_fold", "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE, "dropout": DROPOUT, "seed": SEED,
            "sample_count": len(rows), "feature_count": int((lengths > 0).sum()),
            "batch_size": args.batch_size,
            "learning_rates": LEARNING_RATES, "weight_decays": WEIGHT_DECAYS}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    oof = []
    fold_records = []
    config_history = {(lr, wd): [] for lr in LEARNING_RATES for wd in WEIGHT_DECAYS}
    for fold in folds["folds"]:
        outer = np.flatnonzero(np.isin(speakers, fold["outer_holdout_speakers"]))
        inner_val = np.flatnonzero(np.isin(speakers, fold["inner_validation_speakers"]))
        inner_train = np.flatnonzero(np.isin(speakers, fold["inner_train_speakers"]))
        if len(outer) != fold["outer_holdout_count"] or len(inner_val) != fold["inner_validation_count"]:
            raise ValueError("Frozen fold sample count mismatch")
        candidates = []
        for lr in LEARNING_RATES:
            for wd in WEIGHT_DECAYS:
                trained = train_once(frames, lengths, y, inner_train, inner_val, lr, wd,
                                     seed=SEED + fold["fold"], device=args.device,
                                     batch_size=args.batch_size)
                candidate = {"learning_rate": lr, "weight_decay": wd,
                             "best_epoch": trained["best_epoch"],
                             "validation": trained["validation"],
                             "layer_weights": trained["layer_weights"]}
                candidates.append(candidate)
                config_history[(lr, wd)].append(candidate)
                del trained
        chosen = select_config(candidates)
        train_12 = np.flatnonzero(~np.isin(speakers, fold["outer_holdout_speakers"]))
        trained = train_once(frames, lengths, y, train_12, None,
                             chosen["learning_rate"], chosen["weight_decay"],
                             fixed_epochs=chosen["best_epoch"], seed=SEED + fold["fold"],
                             device=args.device, batch_size=args.batch_size)
        probabilities = predict(trained["model"], frames, lengths, outer,
                                args.device, args.batch_size)
        for index, probability in zip(outer, probabilities):
            row = rows[index]
            oof.append({"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                        "fold": fold["fold"], "truth_M": row["labels"]["M"],
                        "truth_native": row["labels"]["A" if args.encoder.startswith("A") else "V"],
                        "probabilities": probability.astype(float).tolist(),
                        "status": "ok" if lengths[index] else "failed"})
        fold_records.append({"fold": fold["fold"], "inner_grid": candidates,
                             "selected": chosen, "outer_count": len(outer),
                             "outer_feature_count": int((lengths[outer] > 0).sum()),
                             "retrained_layer_weights": trained["layer_weights"]})
        write_json_atomic(args.output_dir / "progress.json",
                          {"completed_folds": len(fold_records), "total_folds": len(folds["folds"])})
        del trained
    validate_oof({row["utt_id"] for row in rows}, oof)
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
               "M_speaker_bootstrap_95": bootstrap_classification(primary, group_key="speaker_id"),
               "native_speaker_bootstrap_95": bootstrap_classification(native, group_key="speaker_id"),
               "M_by_speaker": {speaker: classification([r for r in primary if r["speaker_id"] == speaker])
                                for speaker in sorted(set(speakers))},
               "native_by_speaker": {speaker: classification([r for r in native if r["speaker_id"] == speaker])
                                for speaker in sorted(set(speakers))}}
    final_candidates = []
    for (lr, wd), history in config_history.items():
        final_candidates.append({"learning_rate": lr, "weight_decay": wd,
                                 "validation": {"macro_f1": statistics.mean(
                                     item["validation"]["macro_f1"] for item in history),
                                     "uar": statistics.mean(item["validation"]["uar"] for item in history)},
                                 "best_epoch": int(statistics.median_low(
                                     item["best_epoch"] for item in history))})
    result = {"encoder": args.encoder, "head": "h2", "folds": fold_records,
              "final_grid": final_candidates,
              "final_config": select_config(final_candidates),
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
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=("A1", "A3", "V1", "V2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
