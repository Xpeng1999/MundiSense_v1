"""Preflight the longest frozen training AV inputs before Q fold 1."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA, SUBSET_SHA
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MODEL, model_and_optimizer
from emotioncore.x03.q_train import load_input, prepare_target
from emotioncore.x03.q_train_guarded import nvidia_memory_mib


LIMIT_BYTES = 30_000_000_000  # The user's literal 30 GB limit, in bytes.
TOP_N = 20
FROZEN_SELECTION_SHA = "feb1e44736402825dcdaafe9afee25f2c3b9b53edb6cfdf8b0124dbf8807cf10"


def training_av_rows(root: Path) -> list[dict]:
    manifest_meta = json.loads((root / "manifests" / "run_meta.json").read_text(encoding="utf-8"))
    if (manifest_meta["git"]["dirty"] or manifest_meta["subsets_sha256"] != SUBSET_SHA or
            manifest_meta["face_policy_sha256"] != FACE_SHA):
        raise ValueError("Frozen Q manifest provenance differs")
    by_uid = {}
    for name in (*[f"fold{i}" for i in range(5)], "final"):
        path = root / "manifests" / f"{name}_train.jsonl"
        if sha256(path) != manifest_meta["runs"][name]["train_sha256"]:
            raise ValueError(f"Frozen Q training manifest SHA changed: {name}")
        for row in read_jsonl(path):
            if row["effective_form"] != "AV":
                continue
            uid = row["utt_id"]
            if uid in by_uid and any(by_uid[uid].get(key) != row.get(key)
                                     for key in ("wav_sha256", "video_sha256", "truth_M")):
                raise ValueError(f"AV training media differs across folds: {uid}")
            by_uid.setdefault(uid, row)
    return [by_uid[uid] for uid in sorted(by_uid)]


def select_longest(rows: list[dict], cache_root: Path) -> list[dict]:
    ranked = []
    for row in rows:
        path = cache_root / row["utt_id"] / "AV.npz"
        receipt = cache_root / row["utt_id"] / "AV.json"
        if not path.is_file() or not receipt.is_file():
            raise FileNotFoundError(f"AV preflight cache missing: {path}")
        actual = json.loads(receipt.read_text(encoding="utf-8"))
        if (actual["status"] != "ok" or actual["condition"] != "AV" or
                actual["input_wav_sha256"] != row["wav_sha256"] or
                actual["input_video_sha256"] != row["video_sha256"]):
            raise ValueError(f"AV preflight cache receipt changed: {row['utt_id']}")
        with np.load(path) as data:
            length = int(data["input_ids"].shape[-1])
        ranked.append((length, row["utt_id"], row))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for index, (length, uid, row) in enumerate(ranked[:TOP_N], 1):
        digest = sha256(cache_root / uid / "AV.npz")
        receipt = json.loads((cache_root / uid / "AV.json").read_text(encoding="utf-8"))
        if digest != receipt["sha256"]:
            raise ValueError(f"Selected AV processor cache SHA changed: {uid}")
        selected.append({"rank": index, "input_token_length": length,
                         "row": row, "cache_sha256": digest})
    return selected


def prepare(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if (sha256(Path("configs/x03_inner_val_subsets.json")) != SUBSET_SHA or
            sha256(Path("configs/x03_face_missing_policy.json")) != FACE_SHA):
        raise ValueError("Frozen X03 preflight config SHA changed")
    rows = training_av_rows(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "all_training_av.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json_atomic(args.output_dir / "prepare_summary.json",
                      {"git": git, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "n_unique_training_av": len(rows), "manifest": str(manifest),
                       "manifest_sha256": sha256(manifest)})


def probe(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen2_5OmniProcessor

    git = git_provenance(Path.cwd(), args.allow_dirty)
    if (sha256(Path("configs/x03_inner_val_subsets.json")) != SUBSET_SHA or
            sha256(Path("configs/x03_face_missing_policy.json")) != FACE_SHA):
        raise ValueError("Frozen X03 preflight config SHA changed")
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):
        raise RuntimeError("Q AV preflight requires the default CUDA allocator")
    prepared = json.loads((args.output_dir / "prepare_summary.json").read_text(encoding="utf-8"))
    manifest = Path(prepared["manifest"])
    if sha256(manifest) != prepared["manifest_sha256"]:
        raise ValueError("AV preflight source manifest SHA changed")
    rows = read_jsonl(manifest)
    selected = select_longest(rows, args.cache_root)
    if len(selected) != TOP_N:
        raise ValueError("Fewer than 20 AV training samples available")
    selected_path = args.output_dir / "selected_longest20.json"
    write_json_atomic(selected_path,
                      {"selection": selected, "source_manifest_sha256": sha256(manifest),
                       "ranking": "input_ids token length descending, utt_id ascending"})
    if sha256(selected_path) != FROZEN_SELECTION_SHA:
        raise ValueError("Frozen longest-20 AV selection SHA changed")
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "source_manifest_sha256": sha256(manifest),
            "selection_sha256": FROZEN_SELECTION_SHA,
            "cuda_allocator": "default",
            "memory_limit_bytes": LIMIT_BYTES}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    torch.manual_seed(20260929)
    torch.set_num_threads(4)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    thinker, optimizer, info = model_and_optimizer()
    meta["gradient_checkpointing"] = info["gradient_checkpointing"]
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    thinker.train()
    receipts = []
    gate = "pass"
    for item in selected:
        row = item["row"]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        status = "ok"
        error = None
        try:
            values = load_input(args.cache_root, row)
            values = prepare_target(values, row["truth_M"], processor.tokenizer)
            result = thinker(**values, use_cache=False, use_audio_in_video=False)
            loss = result.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite AV preflight loss: {row['utt_id']}")
            loss.backward()
            torch.cuda.synchronize()
            del loss, result, values
        except torch.cuda.OutOfMemoryError as exc:
            status = "oom"
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
            gate = "stop"
        allocated = int(torch.cuda.max_memory_allocated())
        if allocated > LIMIT_BYTES:
            gate = "stop"
            status = "over_30_gb"
        receipt = {"rank": item["rank"], "utt_id": row["utt_id"],
                   "input_token_length": item["input_token_length"],
                   "cache_sha256": item["cache_sha256"],
                   "max_memory_allocated_bytes": allocated,
                   "max_memory_allocated_mib": allocated / 2**20,
                   "nvidia_smi_used_mib": nvidia_memory_mib(),
                   "status": status, "error": error}
        receipts.append(receipt)
        write_json_atomic(args.output_dir / "progress.json",
                          {"tested": len(receipts), "target": TOP_N, "gate": gate,
                           "latest": receipt})
        if gate == "stop":
            break
    summary = {"tested": len(receipts), "target": TOP_N, "gate": gate,
               "memory_limit_bytes": LIMIT_BYTES, "memory_limit_gb": 30,
               "peak_allocated_bytes": max(r["max_memory_allocated_bytes"] for r in receipts),
               "parameter_info": info, "receipts": receipts,
               "selection_sha256": sha256(args.output_dir / "selected_longest20.json")}
    write_json_atomic(args.output_dir / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    if gate != "pass":
        write_json_atomic(args.root / "pipeline_stop.json",
                          {"reason": "worst_case_av_over_30_gb_or_oom",
                           "preflight_summary": str(args.output_dir / "summary.json"),
                           "tested": len(receipts), "peak_allocated_bytes": summary["peak_allocated_bytes"]})
        raise RuntimeError("Q worst-case AV memory preflight failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "probe"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    else:
        if args.cache_root is None:
            parser.error("probe requires --cache-root")
        probe(args)


if __name__ == "__main__":
    main()
