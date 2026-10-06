"""Auditable per-model Q budget projection under the approved 75-hour cap."""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

from emotioncore.x03.budget_v2 import GPU_LIMIT_H
from emotioncore.x03.q_resume_budget import CONTINGENCY, DEBUG_RESERVE_H, LOAD_CHECKPOINT_OVERHEAD_H
from emotioncore.x03.q_manifest import read_jsonl

MODELS = ("fold0", "fold1", "fold2", "fold3", "fold4", "final")
FORMS = ("A", "V", "AV")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_runs(model: str) -> tuple[str, ...]:
    return ("final_et_test", "final_sims_valid") if model == "final" else (model,)


def project(root: Path, progress: dict, completed_model: str) -> dict:
    """Use completed wall times and measured checkpointed steps for unfinished work.

    Every completed run is charged at its full process duration, including model
    loading and retries. The 20% margin applies to all remaining work.
    """
    if completed_model not in MODELS:
        raise ValueError(completed_model)
    completed = MODELS[:MODELS.index(completed_model) + 1]
    benchmark = _read(root / "gc_step_benchmark" / "summary.json")["forms"]
    form_counts = {model: Counter(r["effective_form"] for r in read_jsonl(
        root / "manifests" / f"{model}_train.jsonl")) for model in MODELS[1:]}
    inference = _read(root / "inference_manifests" / "summary.json")
    sims = _read(root / "sims_valid_manifests" / "summary.json")
    step_times = {}
    for step in progress["steps"]:
        if step["name"].startswith(("train_", "predict_")):
            step_times[step["name"]] = step_times.get(step["name"], 0.0) + step["elapsed_s"]
    fold0_s = _read(root / "fold0" / "summary.json")["gpu_elapsed_s"]
    actual_train_s = fold0_s
    completed_gc_baseline_s = 0.0
    completed_gc_actual_s = 0.0
    for model in completed:
        if model == "fold0":
            continue
        key = "train_" + model
        if key not in step_times or not (root / model / "summary.json").is_file():
            raise RuntimeError(f"Completed Q model lacks observed training time: {model}")
        actual_train_s += step_times[key]
        completed_gc_actual_s += step_times[key]
        completed_gc_baseline_s += sum(
            2 * form_counts[model][form] * benchmark[form]["train_step_mean_s"]
            for form in FORMS) + 8 * 700 * benchmark["AV"]["inference_mean_s"]
    observed_ratio = (completed_gc_actual_s / completed_gc_baseline_s
                      if completed_gc_baseline_s else 1.0)
    pending_train_s = 0.0
    for model in MODELS[len(completed):]:
        pending_train_s += sum(
            2 * form_counts[model][form] * benchmark[form]["train_step_mean_s"]
            for form in FORMS) + 8 * 700 * benchmark["AV"]["inference_mean_s"]
    pending_train_s *= observed_ratio * (1 + CONTINGENCY)

    actual_prediction_s = 0.0
    pending_prediction_s = 0.0
    for model in MODELS:
        for run in _prediction_runs(model):
            key = "predict_" + run
            if key in step_times and (root / key / "summary.json").is_file():
                actual_prediction_s += step_times[key]
                continue
            manifest_summary = sims if run == "final_sims_valid" else inference
            pending_prediction_s += sum(
                manifest_summary[f"{run}_{form}"]["n"] * benchmark[form]["inference_mean_s"]
                for form in FORMS) * (1 + CONTINGENCY)

    projected_h = (actual_train_s + pending_train_s + actual_prediction_s +
                   pending_prediction_s) / 3600 + DEBUG_RESERVE_H + LOAD_CHECKPOINT_OVERHEAD_H
    return {
        "completed_model": completed_model,
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_limit_h": GPU_LIMIT_H,
        "fold0_actual_h": fold0_s / 3600,
        "completed_checkpointed_model_actual_h": completed_gc_actual_s / 3600,
        "completed_prediction_actual_h": actual_prediction_s / 3600,
        "remaining_training_validation_h_with_contingency": pending_train_s / 3600,
        "remaining_prediction_h_with_contingency": pending_prediction_s / 3600,
        "checkpointed_actual_to_benchmark_ratio": observed_ratio,
        "contingency_fraction": CONTINGENCY,
        "diagnostic_gpu_reserve_h": DEBUG_RESERVE_H,
        "load_checkpoint_overhead_h": LOAD_CHECKPOINT_OVERHEAD_H,
        "projected_total_gpu_h": projected_h,
        "status": "within_budget" if projected_h <= GPU_LIMIT_H else "stop_before_next_model",
    }


def record(root: Path, progress: dict, completed_model: str) -> dict:
    receipt = project(root, progress, completed_model)
    path = root / "q_budget_progress.jsonl"
    old = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []
    if not old or old[-1]["completed_model"] != completed_model:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    if receipt["status"] != "within_budget":
        from emotioncore.x02.runtime import write_json_atomic
        write_json_atomic(root / "pipeline_stop.json", {"reason": "projected_total_exceeds_75h",
                                                       "budget_receipt": receipt})
        raise RuntimeError(f"Q projected total {receipt['projected_total_gpu_h']:.2f} h exceeds 75 h; stop before next model")
    return receipt
