"""Nested speaker-CV E2 feature fusion with frozen X02 features."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.folds import validate_oof
from emotioncore.x02.head_h1 import SEED, load_rows, scores, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


LEARNING_RATES = (1e-4, 3e-4)
DROPOUTS = (0.2, 0.4)
MODALITY_DROPOUT = 0.2
MAX_EPOCHS = 50
PATIENCE = 5
CONDITIONS = ("AVT", "A", "V", "AV")


def select_config(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda item: (item["validation"]["macro_f1"],
                                             item["validation"]["uar"],
                                             -item["learning_rate"],
                                             -item["dropout"]))


class E2Model:
    """Wrapper keeping torch import out of command parsing and light tests."""

    @staticmethod
    def build(dropout: float):
        import torch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_logits = torch.nn.Parameter(torch.zeros(4))
                self.audio_norm = torch.nn.LayerNorm(1024)
                self.audio_attention = torch.nn.Linear(1024, 1)
                self.visual_norm = torch.nn.LayerNorm(768)
                self.visual_attention = torch.nn.Linear(768, 1)
                self.text_norm = torch.nn.LayerNorm(1024)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(1024 + 768 + 1024, 256),
                    torch.nn.GELU(), torch.nn.Dropout(dropout),
                    torch.nn.Linear(256, len(LABELS)))

            @staticmethod
            def pool(frames, lengths, norm, attention):
                frames = norm(frames)
                mask = torch.arange(frames.shape[1], device=frames.device)[None, :] >= lengths[:, None]
                logits = attention(frames).squeeze(-1).masked_fill(mask, -1e4)
                weights = torch.softmax(logits, dim=1)
                pooled = (frames * weights[..., None]).sum(dim=1)
                return pooled * (lengths > 0)[:, None]

            def forward(self, audio, audio_lengths, visual, visual_lengths, text, keep):
                weights = torch.softmax(self.layer_logits, dim=0)
                mixed_audio = (audio * weights[None, :, None, None]).sum(dim=1)
                pooled_audio = self.pool(mixed_audio, audio_lengths,
                                         self.audio_norm, self.audio_attention)
                pooled_visual = self.pool(visual, visual_lengths,
                                          self.visual_norm, self.visual_attention)
                pooled_text = self.text_norm(text)
                joined = torch.cat((pooled_audio * keep[:, 0:1],
                                    pooled_visual * keep[:, 1:2],
                                    pooled_text * keep[:, 2:3]), dim=1)
                return self.classifier(joined)

        return Net()


def sample_keep(available, probability: float = MODALITY_DROPOUT,
                *, generator=None):
    """Independent modality dropout with a deterministic nonempty fallback."""
    import torch

    retain = (torch.rand(available.shape, device=available.device,
                         generator=generator) >= probability) & available
    empty = ~retain.any(dim=1) & available.any(dim=1)
    if empty.any():
        first = available.float().argmax(dim=1)
        retain[empty, first[empty]] = True
    return retain.float()


def condition_keep(available, condition: str):
    import torch

    selected = {"AVT": (1, 1, 1), "A": (1, 0, 0),
                "V": (0, 1, 0), "AV": (1, 1, 0)}[condition]
    return available.float() * torch.tensor(selected, device=available.device)


def load_cv(args):
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if sha256(args.folds) != "85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce":
        raise ValueError("Frozen fold SHA mismatch")
    if sha256(args.manifest) != folds["source_manifest_sha256"]:
        raise ValueError("ET manifest differs from frozen folds")
    rows = load_rows(args.manifest, folds)
    packs = {}
    for name, directory in (("A3", args.a3_pack), ("V2", args.v2_pack)):
        meta = json.loads((directory / "pack_meta.json").read_text(encoding="utf-8"))
        if (meta["encoder"] != name or meta["folds_sha256"] != sha256(args.folds)
                or meta["manifest_sha256"] != sha256(args.manifest)):
            raise ValueError(f"{name} feature pack provenance mismatch")
        frames = Path(meta["frame_file"])
        lengths = Path(meta["lengths_file"])
        if sha256(frames) != meta["frame_sha256"] or sha256(lengths) != meta["lengths_sha256"]:
            raise ValueError(f"{name} feature pack hash mismatch")
        packs[name] = (np.load(frames, mmap_mode="r"), np.load(lengths), meta)
    with np.load(args.t1_tokens) as archive:
        token_ids = [str(uid) for uid in archive["utt_ids"]]
        token_failed = archive["failed"].copy()
    if token_ids != [row["utt_id"] for row in rows]:
        raise ValueError("T1 token IDs not aligned to frozen CV pool")
    cls_meta = json.loads(args.t1_cls_meta.read_text(encoding="utf-8"))
    cls = np.load(args.t1_cls, mmap_mode="r")
    if (cls_meta["checkpoint_policy"].startswith("X02 unfine-tuned T1 base") is False
            or sha256(args.t1_cls) != cls_meta["feature_sha256"]
            or sha256(args.t1_tokens) != cls_meta["tokens_sha256"]
            or cls.shape != (len(rows), 1024)):
        raise ValueError("T1 base [CLS] provenance or shape mismatch")
    if any(len(packs[name][0]) != len(rows) for name in packs):
        raise ValueError("A3/V2 feature count mismatch")
    face = json.loads(args.face_policy.read_text(encoding="utf-8"))
    missing_ids = set(face["face_missing_utt_ids"])
    if face["face_missing_count"] != 43 or len(missing_ids) != 43:
        raise ValueError("Frozen face-missing policy differs")
    face_missing = np.asarray([row["utt_id"] in missing_ids for row in rows])
    visual_lengths = packs["V2"][1].copy()
    visual_lengths[face_missing] = 0
    return {"folds": folds, "rows": rows,
            "audio": packs["A3"][0], "audio_lengths": packs["A3"][1],
            "visual": packs["V2"][0], "visual_lengths": visual_lengths,
            "text": cls, "text_available": token_failed == 0,
            "face_missing": face_missing,
            "source_meta_sha256": {"A3": sha256(args.a3_pack / "pack_meta.json"),
                                   "V2": sha256(args.v2_pack / "pack_meta.json"),
                                   "T1_cls": sha256(args.t1_cls_meta),
                                   "face_policy": sha256(args.face_policy)}}


def batch(data, indices, device):
    import torch

    audio = torch.from_numpy(np.asarray(data["audio"][indices], dtype=np.float32)).to(device)
    alen = torch.from_numpy(data["audio_lengths"][indices].astype(np.int64)).to(device)
    visual = torch.from_numpy(np.asarray(data["visual"][indices], dtype=np.float32)).to(device)
    vlen = torch.from_numpy(data["visual_lengths"][indices].astype(np.int64)).to(device)
    text = torch.from_numpy(np.asarray(data["text"][indices], dtype=np.float32)).to(device)
    text_ok = torch.from_numpy(data["text_available"][indices]).to(device)
    available = torch.stack((alen > 0, vlen > 0, text_ok), dim=1)
    return audio, alen, visual, vlen, text, available


def predict(model, data, indices, condition: str, device: str, batch_size: int):
    import torch

    output = np.full((len(indices), len(LABELS)), 1 / len(LABELS), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            b = batch(data, indices[start:end], device)
            keep = condition_keep(b[-1], condition)
            valid = keep.any(dim=1).cpu().numpy()
            if valid.any():
                logits = model(*b[:-1], keep)
                output[start:end][valid] = torch.softmax(logits[valid], dim=1).cpu().numpy()
    return output


def train_once(data, y, train_ids, val_ids, learning_rate, dropout,
               *, fixed_epochs=None, seed=SEED, device="cpu", batch_size=32):
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = E2Model.build(dropout).to(device)
    counts = np.bincount(y[train_ids], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("A class is absent from the E2 training fold")
    weights = torch.tensor(len(train_ids) / (len(LABELS) * counts),
                           dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best = None
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    limit = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    for epoch in range(1, limit + 1):
        model.train()
        shuffled = np.random.default_rng(seed + epoch).permutation(train_ids)
        running = 0.0
        for offset in range(0, len(shuffled), batch_size):
            indices = shuffled[offset:offset + batch_size]
            inputs = batch(data, indices, device)
            keep = sample_keep(inputs[-1])
            labels = torch.from_numpy(y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs[:-1], keep)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(indices)
        record = {"epoch": epoch, "train_loss": running / len(shuffled)}
        if val_ids is not None:
            probability = predict(model, data, val_ids, "AVT", device, batch_size)
            valid = np.ones(len(val_ids), dtype=bool)
            score = scores(y[val_ids], probability.argmax(1), valid)
            record["validation"] = score
            rank = (score["macro_f1"], score["uar"])
            if best is None or rank > best:
                best, best_epoch, stale = rank, epoch, 0
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if val_ids is not None and stale >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "best_epoch": best_epoch if val_ids is not None else limit,
            "validation": {"macro_f1": best[0], "uar": best[1]} if best else None,
            "history": history, "class_counts": counts.tolist(),
            "layer_weights": torch.softmax(model.layer_logits, dim=0).detach().cpu().tolist()}


def run(args):
    started = time.monotonic()
    git = git_provenance(Path.cwd(), args.allow_dirty)
    import torch

    torch.set_num_threads(args.threads)
    data = load_cv(args)
    rows = data["rows"]
    folds = data["folds"]
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in rows], dtype=np.int64)
    speakers = np.asarray([row["speaker_id"] for row in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_meta_sha256": data["source_meta_sha256"],
            "folds_sha256": sha256(args.folds),
            "T1_policy": "unfine-tuned frozen X02 T1 base [CLS] to prevent outer-fold label leakage",
            "learning_rates": LEARNING_RATES, "dropouts": DROPOUTS,
            "modality_dropout_independent_p": MODALITY_DROPOUT,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "class_balance": "inverse_frequency_train_fold", "batch_size": args.batch_size,
            "device": args.device, "threads": args.threads,
            "face_missing_count_cv": int(data["face_missing"].sum())}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    oof = {condition: [] for condition in CONDITIONS}
    fold_records = []
    config_history = {(lr, dropout): [] for lr in LEARNING_RATES for dropout in DROPOUTS}
    for fold in folds["folds"]:
        number = fold["fold"]
        outer = np.flatnonzero(np.isin(speakers, fold["outer_holdout_speakers"]))
        inner_val = np.flatnonzero(np.isin(speakers, fold["inner_validation_speakers"]))
        inner_train = np.flatnonzero(np.isin(speakers, fold["inner_train_speakers"]))
        if (len(outer) != fold["outer_holdout_count"] or
                len(inner_val) != fold["inner_validation_count"] or
                set(outer) & set(inner_val) or set(outer) & set(inner_train)):
            raise ValueError("Frozen E2 speaker split mismatch")
        candidates = []
        for lr in LEARNING_RATES:
            for dropout in DROPOUTS:
                trained = train_once(data, y, inner_train, inner_val, lr, dropout,
                                     seed=SEED + number, device=args.device,
                                     batch_size=args.batch_size)
                candidate = {"learning_rate": lr, "dropout": dropout,
                             "best_epoch": trained["best_epoch"],
                             "validation": trained["validation"],
                             "history": trained["history"]}
                candidates.append(candidate)
                config_history[(lr, dropout)].append(candidate)
                del trained
        chosen = select_config(candidates)
        train_12 = np.flatnonzero(~np.isin(speakers, fold["outer_holdout_speakers"]))
        trained = train_once(data, y, train_12, None,
                             chosen["learning_rate"], chosen["dropout"],
                             fixed_epochs=chosen["best_epoch"], seed=SEED + number,
                             device=args.device, batch_size=args.batch_size)
        checkpoint = args.output_dir / f"fold_{number}.pt"
        torch.save(trained["model"].state_dict(), checkpoint)
        for condition in CONDITIONS:
            probabilities = predict(trained["model"], data, outer, condition,
                                    args.device, args.batch_size)
            for index, probability in zip(outer, probabilities):
                row = rows[index]
                oof[condition].append({"utt_id": row["utt_id"],
                                       "speaker_id": row["speaker_id"],
                                       "fold": number, "truth_M": row["labels"]["M"],
                                       "probabilities": probability.astype(float).tolist(),
                                       "status": "face_missing" if data["face_missing"][index] else "ok"})
        fold_records.append({"fold": number, "inner_grid": candidates,
                             "selected": chosen, "outer_count": len(outer),
                             "checkpoint": str(checkpoint),
                             "checkpoint_sha256": sha256(checkpoint),
                             "layer_weights": trained["layer_weights"]})
        write_json_atomic(args.output_dir / "progress.json",
                          {"completed_folds": len(fold_records), "total_folds": 5})
        del trained
    for condition in CONDITIONS:
        validate_oof({row["utt_id"] for row in rows}, oof[condition])
        output = args.output_dir / f"oof_{condition}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for row in sorted(oof[condition], key=lambda row: row["utt_id"]):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    final_candidates = []
    for (lr, dropout), history in config_history.items():
        final_candidates.append({"learning_rate": lr, "dropout": dropout,
                                 "validation": {"macro_f1": statistics.mean(
                                     item["validation"]["macro_f1"] for item in history),
                                     "uar": statistics.mean(item["validation"]["uar"] for item in history)},
                                 "best_epoch": int(statistics.median_low(
                                     item["best_epoch"] for item in history))})
    result = {"folds": fold_records, "final_grid": final_candidates,
              "final_config": select_config(final_candidates),
              "oof": {condition: {"path": str(args.output_dir / f"oof_{condition}.jsonl"),
                                   "sha256": sha256(args.output_dir / f"oof_{condition}.jsonl")}
                      for condition in CONDITIONS},
              "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["elapsed_s"] = result["elapsed_s"]
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--a3-pack", type=Path, required=True)
    parser.add_argument("--v2-pack", type=Path, required=True)
    parser.add_argument("--t1-tokens", type=Path, required=True)
    parser.add_argument("--t1-cls", type=Path, required=True)
    parser.add_argument("--t1-cls-meta", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size and threads must be positive")
    run(args)


if __name__ == "__main__":
    main()
