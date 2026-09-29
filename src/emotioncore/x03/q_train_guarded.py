"""Q training from fold 1 onward with restartable validation checkpoints."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA, SUBSET_SHA
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MODEL, assert_modality_receipt, model_and_optimizer
from emotioncore.x03.q_train import (EFFECTIVE_BATCH, LR, MAX_EPOCHS, SEED,
                                     WARMUP_RATIO, eval_boundaries, load_input,
                                     prepare_target, validate)


def ordered_training_rows(rows: list[dict], name: str) -> list[dict]:
    """Exactly match the frozen fold-specific order of the original trainer."""
    ordered = []
    for epoch in range(MAX_EPOCHS):
        shuffled = rows.copy()
        random.Random(SEED + epoch + (0 if name == "final" else
                                     int(name.removeprefix("fold")) * 100)).shuffle(shuffled)
        ordered.extend(shuffled)
    return ordered


def segment_ends(n: int) -> list[int]:
    """Split at optimizer and validation boundaries without moving either."""
    total = n * MAX_EPOCHS
    return sorted(set(range(EFFECTIVE_BATCH, total + 1, EFFECTIVE_BATCH)) |
                  set(eval_boundaries(n)) | {total})


def nvidia_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5)
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def trainable_parameters(thinker):
    return {name: parameter for name, parameter in thinker.named_parameters()
            if parameter.requires_grad}


def gradient_snapshot(parameters: dict) -> dict:
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in parameters.items()}


def restore_gradients(optimizer, parameters: dict, snapshot: dict) -> None:
    optimizer.zero_grad(set_to_none=True)
    for name, saved in snapshot.items():
        if saved is not None:
            parameters[name].grad = saved.to(device=parameters[name].device,
                                             dtype=parameters[name].dtype)


def save_last(root: Path, thinker, optimizer, scheduler, parameters: dict,
              progress: dict, position: int, records, skipped, manifest_sha: str,
              val_sha: str) -> dict:
    import torch

    for stream in (records, skipped):
        stream.flush()
        os.fsync(stream.fileno())
    checkpoint_root = root / "last_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    identifier = f"cursor_{position:08d}_{uuid.uuid4().hex[:8]}"
    temporary = checkpoint_root / f".{identifier}.tmp"
    final = checkpoint_root / identifier
    temporary.mkdir()
    thinker.save_pretrained(str(temporary / "adapter"), safe_serialization=True)
    state = {
        "position": position, "progress": progress,
        "train_manifest_sha256": manifest_sha, "val_manifest_sha256": val_sha,
        "face_policy_sha256": FACE_SHA, "subsets_sha256": SUBSET_SHA,
        "record_bytes": (root / "train_modality_receipts.jsonl").stat().st_size,
        "skipped_bytes": (root / "skipped_oom.jsonl").stat().st_size,
        "adapter_parameters": {name: value.detach().cpu().clone()
                               for name, value in parameters.items()},
        "gradients": gradient_snapshot(parameters),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }
    torch.save(state, temporary / "state.pt")
    pointer = {"path": str(final), "position": position,
               "state_sha256": sha256(temporary / "state.pt"),
               "adapter_sha256": sha256(temporary / "adapter" / "adapter_model.safetensors"),
               "train_manifest_sha256": manifest_sha,
               "val_manifest_sha256": val_sha}
    os.replace(temporary, final)
    write_json_atomic(root / "last.json", pointer)
    return pointer


def restore_last(root: Path, thinker, optimizer, scheduler, parameters: dict,
                 manifest_sha: str, val_sha: str) -> tuple[int, dict, int, int]:
    import torch

    pointer = json.loads((root / "last.json").read_text(encoding="utf-8"))
    path = Path(pointer["path"])
    if (pointer["train_manifest_sha256"] != manifest_sha or
            pointer["val_manifest_sha256"] != val_sha or
            sha256(path / "state.pt") != pointer["state_sha256"] or
            sha256(path / "adapter" / "adapter_model.safetensors") !=
            pointer["adapter_sha256"]):
        raise ValueError("Last Q checkpoint provenance or SHA mismatch")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if (state["train_manifest_sha256"] != manifest_sha or
            state["val_manifest_sha256"] != val_sha or
            state["face_policy_sha256"] != FACE_SHA or
            state["subsets_sha256"] != SUBSET_SHA or
            state["position"] != pointer["position"] or
            set(state["adapter_parameters"]) != set(parameters)):
        raise ValueError("Last Q checkpoint state differs from frozen run")
    for name, parameter in parameters.items():
        saved = state["adapter_parameters"][name]
        if saved.shape != parameter.shape:
            raise ValueError(f"Last Q adapter shape differs: {name}")
        parameter.data.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_gradients(optimizer, parameters, state["gradients"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return (state["position"], state["progress"], state["record_bytes"],
            state["skipped_bytes"])


def input_token_length(cache_root: Path, row: dict) -> int:
    with np.load(cache_root / row["utt_id"] / f"{row['effective_form']}.npz") as data:
        return int(data["input_ids"].shape[-1])


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen2_5OmniProcessor

    if args.fold_name == "fold0":
        raise ValueError("The active fold0 must remain on the original trainer")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("Guarded Q training requires expandable CUDA segments")
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if (sha256(Path("configs/x03_inner_val_subsets.json")) != SUBSET_SHA or
            sha256(Path("configs/x03_face_missing_policy.json")) != FACE_SHA):
        raise ValueError("Frozen X03 config SHA changed")
    manifest_meta = json.loads(args.manifest_meta.read_text(encoding="utf-8"))
    info = manifest_meta["runs"][args.fold_name]
    train_sha, val_sha = sha256(args.train_manifest), sha256(args.val_manifest)
    if (manifest_meta["git"]["dirty"] or manifest_meta["subsets_sha256"] != SUBSET_SHA or
            manifest_meta["face_policy_sha256"] != FACE_SHA or
            info["train_sha256"] != train_sha or info["val_sha256"] != val_sha):
        raise ValueError("Frozen Q manifest or provenance changed")
    train_rows, val_rows = read_jsonl(args.train_manifest), read_jsonl(args.val_manifest)
    if (len(train_rows) != info["train_count"] or len(val_rows) != 700 or
            len({r["utt_id"] for r in train_rows}) != len(train_rows) or
            len({r["utt_id"] for r in val_rows}) != len(val_rows) or
            {r["speaker_id"] for r in train_rows} & {r["speaker_id"] for r in val_rows}):
        raise ValueError("Frozen Q speaker split or coverage changed")
    if any(r["face_missing"] and r["effective_form"] != "A" for r in train_rows + val_rows):
        raise ValueError("Frozen face-missing training fallback changed")
    for row in train_rows + val_rows:
        if not (args.cache_root / row["utt_id"] / f"{row['effective_form']}.json").is_file():
            raise FileNotFoundError(f"Q cache missing: {row['utt_id']}/{row['effective_form']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "fold_name": args.fold_name, "train_manifest_sha256": train_sha,
            "val_manifest_sha256": val_sha, "subsets_sha256": SUBSET_SHA,
            "face_policy_sha256": FACE_SHA,
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "seed": SEED, "max_epochs": MAX_EPOCHS, "effective_batch": EFFECTIVE_BATCH,
            "learning_rate": LR, "warmup_ratio": WARMUP_RATIO,
            "training_count": len(train_rows), "validation_count": len(val_rows),
            "cuda_allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            "resume_last": args.resume_last}
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
    parameters = trainable_parameters(thinker)
    total_examples = len(train_rows) * MAX_EPOCHS
    optimizer_steps = math.ceil(total_examples / EFFECTIVE_BATCH)
    warmup_steps = max(1, math.ceil(optimizer_steps * WARMUP_RATIO))

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        fraction = min(1.0, (step - warmup_steps) / max(1, optimizer_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * fraction))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    ordered = ordered_training_rows(train_rows, args.fold_name)
    boundaries = set(eval_boundaries(len(train_rows)))
    ends = segment_ends(len(train_rows))
    records_path = args.output_dir / "train_modality_receipts.jsonl"
    skipped_path = args.output_dir / "skipped_oom.jsonl"
    progress = {"examples_seen": 0, "trained_examples": 0, "optimizer_steps": 0,
                "evals": [], "best_av_macro_f1": -1.0, "best_checkpoint": None,
                "gpu_elapsed_s": 0.0, "modality_forward_counts": {"A": 0, "V": 0, "AV": 0},
                "face_missing_fallback_train": 0, "skipped_oom": 0,
                "time_breakdown_s": {"input_load": 0.0, "forward_backward": 0.0,
                                     "optimizer": 0.0}}
    if args.resume_last:
        if not (args.output_dir / "last.json").is_file():
            raise FileNotFoundError("No last checkpoint for guarded Q resume")
        position, progress, record_bytes, skipped_bytes = restore_last(
            args.output_dir, thinker, optimizer, scheduler, parameters, train_sha, val_sha)
        for path, size in ((records_path, record_bytes), (skipped_path, skipped_bytes)):
            with path.open("r+b") as stream:
                stream.truncate(size)
        mode = "a"
        meta["resumed_from_position"] = position
        meta["resumed_from_sha256"] = sha256(args.output_dir / "last.json")
        write_json_atomic(args.output_dir / "run_meta.json", meta)
    else:
        if (args.output_dir / "last.json").exists():
            raise FileExistsError("Guarded Q output already has last checkpoint; use --resume-last")
        position = 0
        records_path.write_text("", encoding="utf-8")
        skipped_path.write_text("", encoding="utf-8")
        optimizer.zero_grad(set_to_none=True)
    if position not in {0, *ends} or position != progress["examples_seen"]:
        raise ValueError("Last checkpoint cursor is not a frozen segment boundary")
    thinker.train()
    torch.cuda.reset_peak_memory_stats()
    session_started = time.monotonic()
    prior_elapsed = progress["gpu_elapsed_s"]
    recent_steps: list[float] = []
    slow_streak = 0
    batch_duration = 0.0
    peak_allocated = 0

    def elapsed() -> float:
        return prior_elapsed + time.monotonic() - session_started

    def append_json(path: Path, data: dict) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")

    with records_path.open("a", encoding="utf-8") as records, \
            skipped_path.open("a", encoding="utf-8") as skipped:
        if not args.resume_last:
            progress["last_checkpoint"] = "initial"
            save_last(args.output_dir, thinker, optimizer, scheduler, parameters,
                      progress, 0, records, skipped, train_sha, val_sha)
            write_json_atomic(args.output_dir / "progress.json", progress)
        while position < total_examples:
            stop = next(end for end in ends if end > position)
            segment = ordered[position:stop]
            baseline_gradients = gradient_snapshot(parameters)
            completed = False
            local_records = []
            for attempt in (1, 2):
                if attempt == 2:
                    gc.collect()
                    torch.cuda.empty_cache()
                    restore_gradients(optimizer, parameters, baseline_gradients)
                local_records = []
                try:
                    for row in segment:
                        io_started = time.monotonic()
                        values = load_input(args.cache_root, row)
                        io_s = time.monotonic() - io_started
                        actual = set(values)
                        assert_modality_receipt(row["effective_form"], actual)
                        tokens = int(values["input_ids"].shape[-1])
                        values = prepare_target(values, row["truth_M"], processor.tokenizer)
                        compute_started = time.monotonic()
                        result = thinker(**values, use_cache=False, use_audio_in_video=False)
                        loss = result.loss
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Q nonfinite loss: {row['utt_id']}")
                        (loss / EFFECTIVE_BATCH).backward()
                        compute_s = time.monotonic() - compute_started
                        local_records.append({"utt_id": row["utt_id"],
                                              "epoch": (position + len(local_records)) // len(train_rows),
                                              "assigned_form": row["assigned_form"],
                                              "effective_form": row["effective_form"],
                                              "face_missing": row["face_missing"],
                                              "forward_has_audio": "input_features" in actual,
                                              "forward_has_video": "pixel_values_videos" in actual,
                                              "input_token_length": tokens,
                                              "input_load_s": io_s,
                                              "forward_backward_s": compute_s})
                        del result, loss, values
                    completed = True
                    break
                except torch.cuda.OutOfMemoryError:
                    # The whole current accumulation segment is retried, so
                    # no partial gradient from the failed backward is used.
                    values = result = loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    if attempt == 1:
                        append_json(args.output_dir / "oom_incidents.jsonl",
                                    {"position": position, "stop": stop,
                                     "attempt": attempt, "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                     "utt_ids": [r["utt_id"] for r in segment]})
            if not completed:
                restore_gradients(optimizer, parameters, baseline_gradients)
                for row in segment:
                    item = {"utt_id": row["utt_id"], "epoch": position // len(train_rows),
                            "effective_form": row["effective_form"],
                            "input_token_length": input_token_length(args.cache_root, row),
                            "segment_start": position, "segment_stop": stop,
                            "reason": "oom_after_one_retry"}
                    skipped.write(json.dumps(item, ensure_ascii=False) + "\n")
                skipped.flush()
                progress["skipped_oom"] += len(segment)
                progress["examples_seen"] = stop
                position = stop
                if progress["skipped_oom"] > 10:
                    write_json_atomic(args.output_dir / "terminal_stop.json",
                                      {"reason": "more_than_10_skipped_oom",
                                       "skipped_oom": progress["skipped_oom"],
                                       "position": position})
                    write_json_atomic(args.output_dir / "progress.json", progress)
                    raise RuntimeError("Q skipped OOM count exceeded 10")
            else:
                for item in local_records:
                    records.write(json.dumps(item, ensure_ascii=False) + "\n")
                    progress["modality_forward_counts"][item["effective_form"]] += 1
                    progress["face_missing_fallback_train"] += int(
                        item["face_missing"] and item["assigned_form"] != "A")
                    progress["time_breakdown_s"]["input_load"] += item["input_load_s"]
                    progress["time_breakdown_s"]["forward_backward"] += item["forward_backward_s"]
                    batch_duration += item["input_load_s"] + item["forward_backward_s"]
                progress["trained_examples"] += len(local_records)
                progress["examples_seen"] = stop
                position = stop
            if position % EFFECTIVE_BATCH == 0 or position == total_examples:
                if completed or any(value is not None for value in baseline_gradients.values()):
                    optimizer_started = time.monotonic()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    progress["time_breakdown_s"]["optimizer"] += time.monotonic() - optimizer_started
                    progress["optimizer_steps"] += 1
                    step_s = batch_duration + time.monotonic() - optimizer_started
                    baseline = statistics.median(recent_steps) if len(recent_steps) >= 10 else None
                    slow_streak = slow_streak + 1 if baseline and step_s > 3 * baseline else 0
                    if slow_streak >= 2:
                        append_json(args.output_dir / "memory_warnings.jsonl",
                                    {"optimizer_step": progress["optimizer_steps"],
                                     "step_s": step_s, "median_previous_s": baseline,
                                     "consecutive_slow_steps": slow_streak,
                                     "possible_wsl_spill": True})
                    recent_steps.append(step_s)
                    recent_steps = recent_steps[-50:]
                    batch_duration = 0.0
                    if progress["optimizer_steps"] % 50 == 0:
                        torch.cuda.synchronize()
                        allocated = torch.cuda.max_memory_allocated()
                        peak_allocated = max(peak_allocated, allocated)
                        append_json(args.output_dir / "memory_monitor.jsonl",
                                    {"optimizer_step": progress["optimizer_steps"],
                                     "examples_seen": position,
                                     "max_memory_allocated_bytes": allocated,
                                     "nvidia_smi_used_mib": nvidia_memory_mib(),
                                     "step_s": step_s, "median_previous_s": baseline,
                                     "at": dt.datetime.now(dt.timezone.utc).isoformat()})
                        torch.cuda.reset_peak_memory_stats()
            if position in boundaries:
                torch.cuda.synchronize()
                progress["gpu_elapsed_s"] = elapsed()
                eval_index = len(progress["evals"]) + 1
                evaluation = validate(thinker, val_rows, args.cache_root, first_ids,
                                      args.output_dir / "validation" / f"eval{eval_index}.jsonl",
                                      args.fold_name, eval_index)
                evaluation["examples_seen"] = position
                progress["evals"].append(evaluation)
                if evaluation["macro_f1"] > progress["best_av_macro_f1"]:
                    best = args.output_dir / f"best_adapter_eval{eval_index}"
                    thinker.save_pretrained(str(best), safe_serialization=True)
                    progress["best_av_macro_f1"] = evaluation["macro_f1"]
                    progress["best_checkpoint"] = str(best)
                    progress["best_eval_index"] = eval_index
                    progress["best_checkpoint_sha256"] = sha256(best / "adapter_model.safetensors")
                progress["gpu_elapsed_s"] = elapsed()
                progress["last_checkpoint"] = f"eval{eval_index}"
                save_last(args.output_dir, thinker, optimizer, scheduler, parameters,
                          progress, position, records, skipped, train_sha, val_sha)
                write_json_atomic(args.output_dir / "progress.json", progress)
            elif position % 100 == 0:
                torch.cuda.synchronize()
                progress["gpu_elapsed_s"] = elapsed()
                write_json_atomic(args.output_dir / "progress.json", progress)
    torch.cuda.synchronize()
    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
    progress["gpu_elapsed_s"] = elapsed()
    progress["peak_cuda_allocated_mib"] = peak_allocated / 2**20
    progress["parameter_info"] = parameter_info
    progress["train_modality_receipts_sha256"] = sha256(records_path)
    progress["skipped_oom_sha256"] = sha256(skipped_path)
    progress["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    progress["sanity_pass"] = progress["best_av_macro_f1"] >= .45
    progress["smoke_only"] = False
    write_json_atomic(args.output_dir / "summary.json", progress)
    meta["ended_at"] = progress["completed_at"]
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-name", required=True, choices=[*(f"fold{i}" for i in range(1, 5)), "final"])
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--manifest-meta", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-last", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
