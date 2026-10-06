"""Compare the frozen longest AV step with and without gradient checkpointing."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import sha256
from emotioncore.x03.q_smoke import MODEL, model_and_optimizer
from emotioncore.x03.q_train import load_input, prepare_target


SELECTION_SHA = "feb1e44736402825dcdaafe9afee25f2c3b9b53edb6cfdf8b0124dbf8807cf10"
TOLERANCE = 1e-6


def relative_error(a, b) -> float:
    """Relative L2 error with an exact comparison for zero references."""
    import torch

    reference = torch.linalg.vector_norm(a.float()).item()
    difference = torch.linalg.vector_norm(a.float() - b.float()).item()
    return difference / reference if reference else (0.0 if difference == 0 else math.inf)


def checkpoint_flags(thinker) -> dict[str, bool]:
    wrapped = thinker.base_model.model
    return {"llm": bool(getattr(wrapped.model, "gradient_checkpointing", False)),
            "vision": bool(getattr(wrapped.visual, "gradient_checkpointing", False)),
            "audio": bool(getattr(wrapped.audio_tower, "gradient_checkpointing", False))}


def one_step(thinker, processor, row: dict, cache_root: Path, *, enabled: bool) -> dict:
    import torch

    wrapped = thinker.base_model.model
    if enabled:
        wrapped.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        wrapped.gradient_checkpointing_disable()
    flags = checkpoint_flags(thinker)
    if any(value != enabled for value in flags.values()):
        raise RuntimeError(f"Q checkpoint flags differ from requested state: {flags}")
    thinker.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    values = prepare_target(load_input(cache_root, row), row["truth_M"], processor.tokenizer)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    output = thinker(**values, use_cache=False, use_audio_in_video=False)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite Q checkpoint-equivalence loss")
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    gradients = {name: parameter.grad.detach().float().cpu().clone()
                 for name, parameter in thinker.named_parameters()
                 if parameter.requires_grad and parameter.grad is not None}
    trainable = {name for name, parameter in thinker.named_parameters() if parameter.requires_grad}
    if set(gradients) != trainable:
        raise RuntimeError(f"Missing Q LoRA gradients: {sorted(trainable - set(gradients))[:5]}")
    result = {"loss": float(loss.detach().cpu()), "gradients": gradients,
              "elapsed_s": elapsed,
              "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
              "flags": flags}
    del output, loss, values
    thinker.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen2_5OmniProcessor

    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        raise RuntimeError("Q checkpoint equivalence requires the default CUDA allocator")
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.selection) != SELECTION_SHA:
        raise ValueError("Frozen longest-20 AV selection SHA changed")
    selected = json.loads(args.selection.read_text(encoding="utf-8"))["selection"]
    if len(selected) != 20 or selected[0]["rank"] != 1:
        raise ValueError("Frozen longest AV selection is incomplete")
    item = selected[0]
    row = item["row"]
    if sha256(args.cache_root / row["utt_id"] / "AV.npz") != item["cache_sha256"]:
        raise ValueError("Longest AV cache SHA changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selection_sha256": SELECTION_SHA,
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "cuda_allocator": "default", "utt_id": row["utt_id"],
            "input_token_length": item["input_token_length"],
            "tolerance": TOLERANCE}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    torch.set_num_threads(4)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    thinker, optimizer, info = model_and_optimizer()
    del optimizer
    thinker.train()
    off = one_step(thinker, processor, row, args.cache_root, enabled=False)
    on = one_step(thinker, processor, row, args.cache_root, enabled=True)
    if set(off["gradients"]) != set(on["gradients"]):
        raise RuntimeError("Q LoRA gradient keys differ across checkpoint states")
    gradient_errors = {name: relative_error(off["gradients"][name], on["gradients"][name])
                       for name in off["gradients"]}
    loss_error = abs(off["loss"] - on["loss"]) / max(abs(off["loss"]), 1e-30)
    receipt = {"status": "pass" if loss_error <= TOLERANCE and
               max(gradient_errors.values()) <= TOLERANCE else "fail",
               "utt_id": row["utt_id"], "input_token_length": item["input_token_length"],
               "selection_sha256": SELECTION_SHA, "cuda_allocator": "default",
               "loss_without_checkpointing": off["loss"],
               "loss_with_checkpointing": on["loss"],
               "loss_relative_error": loss_error,
               "max_lora_gradient_relative_error": max(gradient_errors.values()),
               "lora_gradient_count": len(gradient_errors),
               "gradient_relative_errors": gradient_errors,
               "without_checkpointing": {k: v for k, v in off.items() if k != "gradients"},
               "with_checkpointing": {k: v for k, v in on.items() if k != "gradients"},
               "parameter_info": info}
    write_json_atomic(args.output_dir / "receipt.json", receipt)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["receipt_sha256"] = sha256(args.output_dir / "receipt.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    if receipt["status"] != "pass":
        raise RuntimeError("Q checkpoint equivalence exceeds the 1e-6 tolerance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
