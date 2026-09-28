"""Full RoBERTa-large fine-tuning under X02 nested speaker CV."""
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
from emotioncore.x02.head_h1 import SEED, load_rows, scores, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


LEARNING_RATES = (1e-5, 2e-5)
MAX_EPOCHS = 5
PATIENCE = 1


def load_tokens(path: Path, meta_path: Path, rows: list[dict], source: str,
                manifest_sha: str, folds_sha: str) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta["source"] != source or meta["manifest_sha256"] != manifest_sha or
            meta["folds_sha256"] != folds_sha or sha256(path) != meta["tokens_sha256"]):
        raise ValueError(f"T1 {source} token cache provenance mismatch")
    with np.load(path) as data:
        result = {key: data[key].copy() for key in data.files}
    if (list(result["utt_ids"]) != [r["utt_id"] for r in rows] or
            result["input_ids"].shape != (len(rows), 128) or
            result["attention_mask"].shape != (len(rows), 128)):
        raise ValueError(f"T1 {source} token cache IDs or shape mismatch")
    return result


def make_model(model_dir: Path, device: str, seed: int):
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=len(LABELS), local_files_only=True).to(device)


def batches(indices: np.ndarray, size: int, seed: int | None = None):
    if seed is not None:
        indices = np.random.default_rng(seed).permutation(indices)
    for offset in range(0, len(indices), size):
        yield indices[offset:offset + size]


def tensor_batch(tokens: dict, indices: np.ndarray, device: str):
    import torch

    ids = torch.from_numpy(tokens["input_ids"][indices].astype(np.int64)).to(device)
    mask = torch.from_numpy(tokens["attention_mask"][indices].astype(np.int64)).to(device)
    return ids, mask


def predict(model, tokens: dict, indices: np.ndarray,
            device: str, batch_size: int) -> np.ndarray:
    import torch

    probabilities = np.full((len(indices), len(LABELS)), 1 / len(LABELS), dtype=np.float32)
    valid_positions = np.flatnonzero(tokens["failed"][indices] == 0)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for positions in batches(valid_positions, batch_size):
            ids, mask = tensor_batch(tokens, indices[positions], device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probabilities[positions] = torch.softmax(logits.float(), dim=1).cpu().numpy()
    return probabilities


def train_once(tokens: dict, y: np.ndarray, model_dir: Path,
               train_ids: np.ndarray, val_ids: np.ndarray | None,
               learning_rate: float, *, fixed_epochs: int | None = None,
               seed: int = SEED, device: str = "cuda:0", batch_size: int = 16) -> dict:
    import torch

    valid_train = train_ids[tokens["failed"][train_ids] == 0]
    counts = np.bincount(y[valid_train], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("At least one class absent from T1 training fold")
    weights = torch.tensor(len(valid_train) / (len(LABELS) * counts),
                           dtype=torch.float32, device=device)
    model = make_model(model_dir, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best = None
    best_epoch = 0
    best_state = None
    no_improvement = 0
    limit = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    for epoch in range(1, limit + 1):
        model.train()
        for batch in batches(valid_train, batch_size, seed + epoch):
            ids, mask = tensor_batch(tokens, batch, device)
            labels = torch.from_numpy(y[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if val_ids is None:
            continue
        probabilities = predict(model, tokens, val_ids, device, batch_size)
        result = scores(y[val_ids], probabilities.argmax(1), tokens["failed"][val_ids] == 0)
        rank = (result["macro_f1"], result["uar"])
        if best is None or rank > best:
            best, best_epoch, no_improvement = rank, epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            no_improvement += 1
            if fixed_epochs is None and no_improvement >= PATIENCE:
                break
    del optimizer
    if val_ids is not None and best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model,
            "best_epoch": best_epoch if val_ids is not None else limit,
            "validation": {"macro_f1": best[0], "uar": best[1]} if best else None,
            "class_counts": counts.tolist()}


def scored(rows: list[dict], truth_key: str) -> list[dict]:
    return [{"id": row["utt_id"], "speaker_id": row["speaker_id"],
             "truth": row[truth_key],
             "pred": LABELS[int(np.argmax(row["probabilities"]))]
             if row["status"] == "ok" else None} for row in rows]


def metric_bundle(predictions: list[dict], speakers: np.ndarray) -> dict:
    primary, native = scored(predictions, "truth_M"), scored(predictions, "truth_native")
    return {"M": classification(primary), "native": classification(native),
            "M_speaker_bootstrap_95": bootstrap_classification(primary, group_key="speaker_id"),
            "native_speaker_bootstrap_95": bootstrap_classification(native, group_key="speaker_id"),
            "M_by_speaker": {speaker: classification([r for r in primary if r["speaker_id"] == speaker])
                             for speaker in sorted(set(speakers))},
            "native_by_speaker": {speaker: classification([r for r in native if r["speaker_id"] == speaker])
                             for speaker in sorted(set(speakers))}}


def run(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    manifest_sha, folds_sha = sha256(args.manifest), sha256(args.folds)
    if manifest_sha != folds["source_manifest_sha256"]:
        raise ValueError("Manifest differs from frozen folds")
    rows = load_rows(args.manifest, folds)
    tokens_o = load_tokens(args.tokens_dir / "T1_o_cv.npz",
                           args.tokens_dir / "T1_o_cv_meta.json", rows, "o",
                           manifest_sha, folds_sha)
    tokens_r = load_tokens(args.tokens_dir / "T1_r_cv.npz",
                           args.tokens_dir / "T1_r_cv_meta.json", rows, "r",
                           manifest_sha, folds_sha)
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in rows], dtype=np.int64)
    speakers = np.asarray([row["speaker_id"] for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / "run_meta.json"
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": manifest_sha, "folds_sha256": folds_sha,
            "tokens_o_sha256": sha256(args.tokens_dir / "T1_o_cv.npz"),
            "tokens_r_sha256": sha256(args.tokens_dir / "T1_r_cv.npz"),
            "model_sha256": sha256(args.model_dir / "pytorch_model.bin"),
            "head": "full_finetune", "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "learning_rates": LEARNING_RATES, "batch_size": args.batch_size,
            "class_balance": "inverse_frequency_train_fold", "seed": SEED,
            "weight_decay": 0.01, "mixed_precision": "bfloat16_autocast",
            "maximum_length": 128}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "folds_sha256", "tokens_o_sha256",
                    "tokens_r_sha256", "model_sha256", "batch_size"):
            if old[key] != meta[key]:
                raise RuntimeError(f"Cannot resume T1 CV with changed {key}")
        if old["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume T1 CV from a different Git commit")
        meta["started_at"] = old["started_at"]
        meta["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(meta_path, meta)
    oof_o, oof_r, fold_records = [], [], []
    history = {lr: [] for lr in LEARNING_RATES}
    for fold in folds["folds"]:
        fold_path = args.output_dir / f"fold_{fold['fold']}.json"
        if fold_path.exists():
            saved = json.loads(fold_path.read_text(encoding="utf-8"))
            if saved["fold"] != fold["fold"] or saved["outer_count"] != fold["outer_holdout_count"]:
                raise ValueError("T1 fold resume record differs from frozen fold")
        else:
            outer = np.flatnonzero(np.isin(speakers, fold["outer_holdout_speakers"]))
            inner_val = np.flatnonzero(np.isin(speakers, fold["inner_validation_speakers"]))
            inner_train = np.flatnonzero(np.isin(speakers, fold["inner_train_speakers"]))
            if len(outer) != fold["outer_holdout_count"] or len(inner_val) != fold["inner_validation_count"]:
                raise ValueError("Frozen T1 fold sample count mismatch")
            candidates = []
            for lr in LEARNING_RATES:
                trained = train_once(tokens_o, y, args.model_dir, inner_train, inner_val, lr,
                                     seed=SEED + fold["fold"], device=args.device,
                                     batch_size=args.batch_size)
                candidates.append({"learning_rate": lr,
                                   "best_epoch": trained["best_epoch"],
                                   "validation": trained["validation"]})
                del trained
                torch.cuda.empty_cache()
            chosen = max(candidates, key=lambda c: (c["validation"]["macro_f1"],
                                                    c["validation"]["uar"],
                                                    -c["learning_rate"]))
            train_12 = np.flatnonzero(~np.isin(speakers, fold["outer_holdout_speakers"]))
            trained = train_once(tokens_o, y, args.model_dir, train_12, None,
                                 chosen["learning_rate"], fixed_epochs=chosen["best_epoch"],
                                 seed=SEED + fold["fold"], device=args.device,
                                 batch_size=args.batch_size)
            predictions_o = predict(trained["model"], tokens_o, outer, args.device, args.batch_size)
            predictions_r = predict(trained["model"], tokens_r, outer, args.device, args.batch_size)
            records_o, records_r = [], []
            for index, probability_o, probability_r in zip(outer, predictions_o, predictions_r):
                row = rows[index]
                common = {"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                          "fold": fold["fold"], "truth_M": row["labels"]["M"],
                          "truth_native": row["labels"]["T"]}
                records_o.append({**common, "source": "o",
                                  "probabilities": probability_o.astype(float).tolist(),
                                  "status": "ok"})
                records_r.append({**common, "source": "r",
                                  "probabilities": probability_r.astype(float).tolist(),
                                  "status": "failed" if tokens_r["failed"][index] else "ok"})
            saved = {"fold": fold["fold"], "inner_grid": candidates,
                     "selected": chosen, "outer_count": len(outer),
                     "oof_o": records_o, "oof_r": records_r}
            write_json_atomic(fold_path, saved)
            del trained
            torch.cuda.empty_cache()
        fold_records.append({key: saved[key] for key in ("fold", "inner_grid", "selected", "outer_count")})
        for candidate in saved["inner_grid"]:
            history[candidate["learning_rate"]].append(candidate)
        oof_o.extend(saved["oof_o"])
        oof_r.extend(saved["oof_r"])
        write_json_atomic(args.output_dir / "progress.json",
                          {"completed_folds": len(fold_records), "total_folds": 5})
    expected = {row["utt_id"] for row in rows}
    validate_oof(expected, oof_o)
    validate_oof(expected, oof_r)
    oof_o.sort(key=lambda row: row["utt_id"])
    oof_r.sort(key=lambda row: row["utt_id"])
    paths = {}
    for source, predictions in (("o", oof_o), ("r", oof_r)):
        path = args.output_dir / f"oof_{source}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[source] = {"path": str(path), "sha256": sha256(path)}
    final_grid = [{"learning_rate": lr,
                   "validation": {"macro_f1": statistics.mean(c["validation"]["macro_f1"]
                                                          for c in history[lr]),
                                  "uar": statistics.mean(c["validation"]["uar"]
                                                         for c in history[lr])},
                   "best_epoch": int(statistics.median_low(c["best_epoch"] for c in history[lr]))}
                  for lr in LEARNING_RATES]
    final_config = max(final_grid, key=lambda c: (c["validation"]["macro_f1"],
                                                  c["validation"]["uar"],
                                                  -c["learning_rate"]))
    result = {"encoder": "T1_chinese_roberta_wwm_ext_large", "head": "full_finetune",
              "folds": fold_records, "final_grid": final_grid, "final_config": final_config,
              "metrics": {"o": metric_bundle(oof_o, speakers),
                          "r": metric_bundle(oof_r, speakers)},
              "oof": paths, "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    if args.train_final:
        all_indices = np.arange(len(rows))
        trained = train_once(tokens_o, y, args.model_dir, all_indices, None,
                             final_config["learning_rate"],
                             fixed_epochs=final_config["best_epoch"],
                             seed=SEED, device=args.device, batch_size=args.batch_size)
        checkpoint = args.output_dir / "final_model.pt"
        torch.save({"state_dict": {key: value.detach().cpu()
                                   for key, value in trained["model"].state_dict().items()},
                    "config": final_config, "labels": LABELS}, checkpoint)
        result["final_checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
        write_json_atomic(args.output_dir / "results.json", result)
        del trained
        torch.cuda.empty_cache()
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["elapsed_s"] = time.monotonic() - started
    write_json_atomic(meta_path, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--tokens-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-final", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
