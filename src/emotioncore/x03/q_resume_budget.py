"""Gate the remaining Q schedule using measured checkpointed A/V/AV steps."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import GPU_LIMIT_H
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_worst_case import LIMIT_BYTES, TOP_N, FROZEN_SELECTION_SHA


CONTINGENCY = 0.2
DEBUG_RESERVE_H = 1.0
LOAD_CHECKPOINT_OVERHEAD_H = 0.5
PREDICTION_UTTERANCES = 16_366 + 1_929 + 647


def estimate(root: Path, benchmark: dict, fold0: dict) -> dict:
    counts = collections.Counter()
    for name in (*[f"fold{i}" for i in range(1, 5)], "final"):
        for row in read_jsonl(root / "manifests" / f"{name}_train.jsonl"):
            counts[row["effective_form"]] += 2  # Frozen two-epoch upper bound.
    if set(counts) != {"A", "V", "AV"}:
        raise ValueError("Remaining Q form counts differ from frozen protocol")
    forms = benchmark["forms"]
    if benchmark["status"] != "complete" or benchmark["cuda_allocator"] != "default" or \
            any(not benchmark["gradient_checkpointing"].get(k)
                for k in ("llm", "vision", "audio")):
        raise ValueError("Q checkpointed step benchmark is not valid")
    training_h = sum(counts[form] * forms[form]["train_step_mean_s"]
                     for form in ("A", "V", "AV")) / 3600 * (1 + CONTINGENCY)
    validation_count = 5 * 8 * 700  # Five remaining models, AV only.
    validation_h = validation_count * forms["AV"]["inference_mean_s"] / 3600 * (1 + CONTINGENCY)
    prediction_h = PREDICTION_UTTERANCES * sum(
        forms[form]["inference_mean_s"] for form in ("A", "V", "AV")) / 3600 * (1 + CONTINGENCY)
    fold0_h = fold0["gpu_elapsed_s"] / 3600
    total_h = (fold0_h + DEBUG_RESERVE_H + training_h + validation_h +
               prediction_h + LOAD_CHECKPOINT_OVERHEAD_H)
    return {"gpu_limit_h": GPU_LIMIT_H,
            "status": "within_budget" if total_h <= GPU_LIMIT_H else "stop",
            "fold0_gpu_h": fold0_h, "diagnostic_gpu_reserve_h": DEBUG_RESERVE_H,
            "remaining_training_examples_by_form": dict(counts),
            "remaining_training_h_with_contingency": training_h,
            "remaining_validation_examples": validation_count,
            "remaining_validation_h_with_contingency": validation_h,
            "prediction_utterances_per_form": PREDICTION_UTTERANCES,
            "remaining_prediction_h_with_contingency": prediction_h,
            "load_checkpoint_overhead_h": LOAD_CHECKPOINT_OVERHEAD_H,
            "contingency_fraction": CONTINGENCY,
            "projected_total_gpu_h": total_h,
            "measured_step_s": {form: {k: forms[form][k]
                                       for k in ("train_step_mean_s", "train_step_median_s",
                                                 "inference_mean_s", "inference_median_s")}
                                for form in ("A", "V", "AV")}}


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    root = args.root
    benchmark_path = root / "gc_step_benchmark" / "summary.json"
    equivalence_path = root / "checkpoint_equivalence" / "receipt.json"
    preflight_path = root / "worst_case_preflight" / "summary.json"
    fold0_path = root / "fold0" / "summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if equivalence["status"] != "pass" or equivalence["selection_sha256"] != FROZEN_SELECTION_SHA or \
            equivalence["loss_relative_error"] > 1e-6 or \
            equivalence["max_lora_gradient_relative_error"] > 1e-6:
        raise ValueError("Q checkpoint equivalence gate is closed")
    if preflight["gate"] != "pass" or preflight["tested"] != TOP_N or \
            preflight["selection_sha256"] != FROZEN_SELECTION_SHA or \
            preflight["peak_allocated_bytes"] > LIMIT_BYTES:
        raise ValueError("Q longest-20 AV memory gate is closed")
    fold0 = json.loads(fold0_path.read_text(encoding="utf-8"))
    result = estimate(root, benchmark, fold0)
    result.update({"git": git, "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "benchmark_sha256": sha256(benchmark_path),
                   "equivalence_sha256": sha256(equivalence_path),
                   "preflight_sha256": sha256(preflight_path),
                   "fold0_sha256": sha256(fold0_path),
                   "selection_sha256": FROZEN_SELECTION_SHA})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    if result["status"] != "within_budget":
        raise RuntimeError(f"Checkpointed Q remaining schedule exceeds {GPU_LIMIT_H:g} GPU hours")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
