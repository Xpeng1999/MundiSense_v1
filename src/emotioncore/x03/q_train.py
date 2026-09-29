"""Frozen Qwen2.5-Omni LoRA v2 training with speaker-held-out AV validation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA, SUBSET_SHA
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MODEL, assert_modality_receipt, model_and_optimizer


SEED = 20260929
MAX_EPOCHS = 2
EFFECTIVE_BATCH = 8
LR = 5e-5
WARMUP_RATIO = 0.05


def load_input(cache_root: Path, row: dict, device: str = "cuda:0") -> dict:
    import torch

    uid = row["utt_id"]
    condition = row["effective_form"]
    receipt = json.loads((cache_root / uid / f"{condition}.json").read_text(encoding="utf-8"))
    if (receipt["status"] != "ok" or receipt["utt_id"] != uid or
            receipt["condition"] != condition or
            receipt["input_wav_sha256"] != row["wav_sha256"] or
            receipt.get("input_video_sha256") != row.get("video_sha256")):
        raise ValueError(f"Frozen Q processor receipt mismatch: {uid}/{condition}")
    with np.load(cache_root / uid / f"{condition}.npz") as source:
        assert_modality_receipt(condition, set(source.files))
        values = {key: torch.from_numpy(source[key]).to(device) for key in source.files}
    assert_modality_receipt(condition, set(values))
    return values


def prepare_target(values: dict, truth: str, tokenizer) -> dict:
    import torch

    if truth not in LABELS:
        raise ValueError(f"Unknown ET M label: {truth}")
    suffix = truth + '"}'
    answer = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(
        values["input_ids"].device)
    # The prompt already ends at {"coarse_emotion":"; finish the JSON and message.
    end = torch.tensor([[tokenizer.convert_tokens_to_ids("<|im_end|>")]],
                       dtype=answer.dtype, device=answer.device)
    answer = torch.cat((answer, end), dim=1)
    prompt = values["input_ids"]
    values["input_ids"] = torch.cat((prompt, answer), dim=1)
    values["attention_mask"] = torch.ones_like(values["input_ids"])
    values["labels"] = torch.cat((torch.full_like(prompt, -100), answer), dim=1)
    return values


def eval_boundaries(n: int, epochs: int = MAX_EPOCHS) -> list[int]:
    return sorted({min((epoch * n) + n, epoch * n + math.ceil((n * quarter / 4) / EFFECTIVE_BATCH)
                        * EFFECTIVE_BATCH)
                   for epoch in range(epochs) for quarter in range(1, 5)})


def validate(thinker, rows: list[dict], cache_root: Path, first_ids: list[int],
             output: Path, fold_name: str, eval_index: int) -> dict:
    import torch

    thinker.eval()
    scored = []
    out_rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            if row["assigned_form"] != "AV":
                raise ValueError("Inner validation must evaluate AV only")
            values = load_input(cache_root, row)
            actual = set(values)
            assert_modality_receipt(row["effective_form"], actual)
            result = thinker(**values, use_cache=False, use_audio_in_video=False)
            scores = result.logits[0, -1].float()[first_ids]
            probabilities = scores.softmax(0).cpu().tolist()
            pred = LABELS[int(np.argmax(probabilities))]
            scored.append({"id": row["utt_id"], "truth": row["truth_M"],
                           "pred": pred, "speaker_id": row["speaker_id"]})
            out_rows.append({"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                             "truth_M": row["truth_M"], "probabilities": probabilities,
                             "pred": pred, "condition": "AV",
                             "face_missing": row["face_missing"],
                             "face_missing_fallback": row["face_missing"],
                             "forward_has_audio": "input_features" in actual,
                             "forward_has_video": "pixel_values_videos" in actual})
    if len(out_rows) != 700 or len({r["utt_id"] for r in out_rows}) != 700:
        raise ValueError("Inner validation coverage differs from frozen 700")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in out_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = classification(scored)
    result.update({"eval_index": eval_index, "fold_name": fold_name,
                   "latency_s": time.monotonic() - started,
                   "predictions_sha256": sha256(output)})
    thinker.train()
    return result


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen2_5OmniProcessor

    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(Path("configs/x03_inner_val_subsets.json")) != SUBSET_SHA or \
       sha256(Path("configs/x03_face_missing_policy.json")) != FACE_SHA:
        raise ValueError("Reviewer-approved X03 frozen config SHA mismatch")
    manifest_meta = json.loads(args.manifest_meta.read_text(encoding="utf-8"))
    if (manifest_meta["git"]["dirty"] or manifest_meta["subsets_sha256"] != SUBSET_SHA or
            manifest_meta["face_policy_sha256"] != FACE_SHA):
        raise ValueError("Q manifest run_meta does not match frozen protocol")
    name = args.fold_name
    manifest_info = manifest_meta["runs"][name]
    if (sha256(args.train_manifest) != manifest_info["train_sha256"] or
            sha256(args.val_manifest) != manifest_info["val_sha256"]):
        raise ValueError("Q train/validation manifest SHA mismatch")
    train_rows = read_jsonl(args.train_manifest)
    val_rows = read_jsonl(args.val_manifest)
    if (len(train_rows) != manifest_info["train_count"] or len(val_rows) != 700 or
            len({r["utt_id"] for r in train_rows}) != len(train_rows) or
            len({r["utt_id"] for r in val_rows}) != 700 or
            {r["speaker_id"] for r in train_rows} & {r["speaker_id"] for r in val_rows}):
        raise ValueError("Q training coverage or speaker separation differs")
    if any(r["face_missing"] and r["effective_form"] != "A" for r in train_rows + val_rows):
        raise ValueError("Frozen face-missing fallback not applied")
    if args.smoke_examples is not None:
        if args.smoke_examples <= 0 or args.smoke_examples > len(train_rows):
            raise ValueError("Invalid smoke example count")
        train_rows = train_rows[:args.smoke_examples]
    for row in train_rows + ([] if args.smoke_examples is not None else val_rows):
        receipt = args.cache_root / row["utt_id"] / f"{row['effective_form']}.json"
        if not receipt.is_file():
            raise FileNotFoundError(f"Q processor cache missing: {receipt}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fold_name": name, "train_manifest_sha256": sha256(args.train_manifest),
            "val_manifest_sha256": sha256(args.val_manifest),
            "subsets_sha256": SUBSET_SHA, "face_policy_sha256": FACE_SHA,
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "seed": SEED, "max_epochs": MAX_EPOCHS, "effective_batch": EFFECTIVE_BATCH,
            "learning_rate": LR, "warmup_ratio": WARMUP_RATIO,
            "training_count": len(train_rows), "validation_count": len(val_rows),
            "smoke_examples": args.smoke_examples}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    first_ids = [processor.tokenizer(label, add_special_tokens=False).input_ids[0]
                 for label in LABELS]
    if len(set(first_ids)) != 7:
        raise RuntimeError("Seven class first tokens collide")
    thinker, optimizer, parameter_info = model_and_optimizer()
    thinker.train()
    total_examples = len(train_rows) if args.smoke_examples is not None else len(train_rows) * MAX_EPOCHS
    optimizer_steps = math.ceil(total_examples / EFFECTIVE_BATCH)
    warmup_steps = max(1, math.ceil(optimizer_steps * WARMUP_RATIO))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        fraction = min(1.0, (step - warmup_steps) / max(1, optimizer_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    boundaries = set() if args.smoke_examples is not None else set(eval_boundaries(len(train_rows)))
    progress = {"examples_seen": 0, "optimizer_steps": 0, "evals": [],
                "best_av_macro_f1": -1.0, "best_checkpoint": None,
                "gpu_elapsed_s": 0.0, "modality_forward_counts": {"A": 0, "V": 0, "AV": 0},
                "face_missing_fallback_train": 0}
    write_json_atomic(args.output_dir / "progress.json", progress)
    records_path = args.output_dir / "train_modality_receipts.jsonl"
    torch.cuda.reset_peak_memory_stats()
    training_started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with records_path.open("w", encoding="utf-8") as record_stream:
        for epoch in range(1 if args.smoke_examples is not None else MAX_EPOCHS):
            shuffled = train_rows.copy()
            random.Random(SEED + epoch + (0 if name == "final" else int(name.removeprefix("fold")) * 100)).shuffle(shuffled)
            for row in shuffled:
                values = load_input(args.cache_root, row)
                actual = set(values)
                assert_modality_receipt(row["effective_form"], actual)
                values = prepare_target(values, row["truth_M"], processor.tokenizer)
                result = thinker(**values, use_cache=False, use_audio_in_video=False)
                loss = result.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Q nonfinite training loss at {row['utt_id']}")
                (loss / EFFECTIVE_BATCH).backward()
                del result, loss, values
                progress["examples_seen"] += 1
                progress["modality_forward_counts"][row["effective_form"]] += 1
                progress["face_missing_fallback_train"] += int(row["face_missing"] and row["assigned_form"] != "A")
                record_stream.write(json.dumps({"utt_id": row["utt_id"], "epoch": epoch,
                                                "assigned_form": row["assigned_form"],
                                                "effective_form": row["effective_form"],
                                                "face_missing": row["face_missing"],
                                                "forward_has_audio": "input_features" in actual,
                                                "forward_has_video": "pixel_values_videos" in actual}) + "\n")
                if progress["examples_seen"] % EFFECTIVE_BATCH == 0 or \
                   progress["examples_seen"] == total_examples:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress["optimizer_steps"] += 1
                if progress["examples_seen"] in boundaries:
                    torch.cuda.synchronize()
                    progress["gpu_elapsed_s"] = time.monotonic() - training_started
                    eval_index = len(progress["evals"]) + 1
                    evaluation = validate(thinker, val_rows, args.cache_root, first_ids,
                                          args.output_dir / "validation" / f"eval{eval_index}.jsonl",
                                          name, eval_index)
                    evaluation["examples_seen"] = progress["examples_seen"]
                    progress["evals"].append(evaluation)
                    score = evaluation["macro_f1"]
                    if score > progress["best_av_macro_f1"]:
                        checkpoint = args.output_dir / "best_adapter"
                        thinker.save_pretrained(str(checkpoint), safe_serialization=True)
                        progress["best_av_macro_f1"] = score
                        progress["best_checkpoint"] = str(checkpoint)
                        progress["best_eval_index"] = eval_index
                        progress["best_checkpoint_sha256"] = sha256(checkpoint / "adapter_model.safetensors")
                    write_json_atomic(args.output_dir / "progress.json", progress)
                elif progress["examples_seen"] % 100 == 0:
                    torch.cuda.synchronize()
                    progress["gpu_elapsed_s"] = time.monotonic() - training_started
                    write_json_atomic(args.output_dir / "progress.json", progress)
    torch.cuda.synchronize()
    progress["gpu_elapsed_s"] = time.monotonic() - training_started
    progress["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    progress["parameter_info"] = parameter_info
    progress["train_modality_receipts_sha256"] = sha256(records_path)
    progress["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    progress["sanity_pass"] = None if args.smoke_examples is not None else progress["best_av_macro_f1"] >= 0.45
    progress["smoke_only"] = args.smoke_examples is not None
    write_json_atomic(args.output_dir / "summary.json", progress)
    meta["ended_at"] = progress["completed_at"]
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-name", required=True, choices=[*(f"fold{i}" for i in range(5)), "final"])
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--manifest-meta", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-examples", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
