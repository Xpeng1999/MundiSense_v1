"""Measure A/V/AV Q training and inference steps with active checkpointing."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MODEL, model_and_optimizer
from emotioncore.x03.q_train import load_input, prepare_target


QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def choose_rows(rows: list[dict], cache_root: Path) -> dict[str, list[dict]]:
    chosen = {}
    for form in ("A", "V", "AV"):
        ranked = []
        for row in rows:
            if row["effective_form"] != form:
                continue
            with np.load(cache_root / row["utt_id"] / f"{form}.npz") as data:
                length = int(data["input_ids"].shape[-1])
            ranked.append((length, row["utt_id"], row))
        if len(ranked) < len(QUANTILES):
            raise ValueError(f"Insufficient cached {form} training rows")
        ranked.sort(key=lambda value: (value[0], value[1]))
        picks = {round(q * (len(ranked) - 1)) for q in QUANTILES}
        chosen[form] = [{"input_token_length": ranked[i][0], "row": ranked[i][2]}
                        for i in sorted(picks)]
        if len(chosen[form]) != len(QUANTILES):
            raise ValueError(f"Repeated {form} quantile picks")
    return chosen


def measure(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen2_5OmniProcessor

    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        raise RuntimeError("Q checkpoint benchmark requires the default CUDA allocator")
    git = git_provenance(Path.cwd(), args.allow_dirty)
    rows = read_jsonl(args.manifest)
    selected = choose_rows(rows, args.cache_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "selection.json", selected)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest),
            "selection_sha256": sha256(args.output_dir / "selection.json"),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "cuda_allocator": "default", "quantiles": QUANTILES}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    torch.manual_seed(20260929)
    torch.set_num_threads(4)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    thinker, optimizer, info = model_and_optimizer()
    meta["gradient_checkpointing"] = info["gradient_checkpointing"]
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    results = {}
    for form, items in selected.items():
        samples = []
        thinker.train()
        for item in items:
            row = item["row"]
            optimizer.zero_grad(set_to_none=True)
            values = prepare_target(load_input(args.cache_root, row), row["truth_M"],
                                    processor.tokenizer)
            torch.cuda.synchronize()
            started = time.monotonic()
            output = thinker(**values, use_cache=False, use_audio_in_video=False)
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Nonfinite Q benchmark loss: {row['utt_id']}")
            output.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            train_s = time.monotonic() - started
            del output, values
            optimizer.zero_grad(set_to_none=True)
            thinker.eval()
            with torch.inference_mode():
                values = load_input(args.cache_root, row)
                torch.cuda.synchronize()
                started = time.monotonic()
                output = thinker(**values, use_cache=False, use_audio_in_video=False)
                torch.cuda.synchronize()
                infer_s = time.monotonic() - started
                del output, values
            thinker.train()
            samples.append({"utt_id": row["utt_id"],
                            "input_token_length": item["input_token_length"],
                            "train_step_s": train_s, "inference_s": infer_s})
            write_json_atomic(args.output_dir / "progress.json",
                              {"form": form, "samples_done": len(samples),
                               "total_per_form": len(items), "samples": samples})
        results[form] = {"samples": samples,
                         "train_step_mean_s": statistics.mean(s["train_step_s"] for s in samples),
                         "train_step_median_s": statistics.median(s["train_step_s"] for s in samples),
                         "inference_mean_s": statistics.mean(s["inference_s"] for s in samples),
                         "inference_median_s": statistics.median(s["inference_s"] for s in samples)}
    summary = {"status": "complete", "cuda_allocator": "default",
               "gradient_checkpointing": info["gradient_checkpointing"],
               "selection_sha256": sha256(args.output_dir / "selection.json"),
               "manifest_sha256": sha256(args.manifest), "forms": results}
    write_json_atomic(args.output_dir / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
