"""Resume approved Q folds after fold-zero sanity, enforcing the 55-hour GPU cap."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA, GPU_LIMIT_H, SUBSET_SHA
from emotioncore.x03.q_manifest import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_step(name: str, command: list[str], root: Path, progress: dict) -> None:
    log = root / "pipeline_logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    progress["active_step"] = name
    progress["active_command"] = command
    progress["active_started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(root / "pipeline_progress.json", progress)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=Path.cwd(), env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    progress["steps"].append({"name": name, "elapsed_s": elapsed,
                              "exit_code": result.returncode, "log": str(log)})
    progress["active_step"] = None
    progress.pop("active_command", None)
    progress.pop("active_started_at", None)
    write_json_atomic(root / "pipeline_progress.json", progress)
    if result.returncode:
        raise RuntimeError(f"Q pipeline {name} failed ({result.returncode}); see {log}")


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if (sha256(Path("configs/x03_face_missing_policy.json")) != FACE_SHA or
            sha256(Path("configs/x03_inner_val_subsets.json")) != SUBSET_SHA):
        raise ValueError("Frozen X03 Q protocol changed")
    budget = read(Path("experiments/X03_fusion/results/q_budget_fold0_preflight.json"))
    if budget["status"] != "within_budget_estimate" or budget["gpu_limit_h"] != GPU_LIMIT_H:
        raise RuntimeError("Q preflight budget gate is closed")
    root = args.root
    fold0 = read(root / "fold0" / "summary.json")
    if not fold0["sanity_pass"] or fold0["best_av_macro_f1"] < .45:
        raise RuntimeError("Fold 0 AV sanity gate failed; do not continue Q")
    manifests = read(root / "manifests" / "summary.json")
    projected_train_h = fold0["gpu_elapsed_s"] * sum(
        manifests[name]["train_count"] for name in (*[f"fold{i}" for i in range(5)], "final")) / \
        manifests["fold0"]["train_count"] / 3600
    projected_total_h = projected_train_h + budget["projected_prediction_h"] + \
        budget["model_load_checkpoint_overhead_h"]
    if projected_total_h > GPU_LIMIT_H:
        write_json_atomic(root / "pipeline_stop.json",
                          {"reason": "fold0_reestimate_exceeds_55h",
                           "projected_total_h": projected_total_h,
                           "fold0_best_av_macro_f1": fold0["best_av_macro_f1"]})
        raise RuntimeError("Fold 0 measured Q projection exceeds the 55-hour GPU cap")
    root.mkdir(parents=True, exist_ok=True)
    progress = {"git": git, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "fold0_sanity_pass": True, "fold0_best_av_macro_f1": fold0["best_av_macro_f1"],
                "fold0_checkpoint_sha256": fold0["best_checkpoint_sha256"],
                "fold0_reestimated_total_gpu_h": projected_total_h,
                "gpu_limit_h": GPU_LIMIT_H, "active_step": None, "steps": []}
    write_json_atomic(root / "pipeline_progress.json", progress)
    py = sys.executable
    cache = args.cache_root

    def command(module: str, *parts: str) -> list[str]:
        return [py, "-u", "-m", f"emotioncore.x03.{module}", *parts]

    def cache_manifest(name: str, manifest: Path) -> None:
        summary = root / f"cache_{name}" / "summary.json"
        if summary.is_file() and read(summary)["failed"] == 0:
            return
        run_step(f"cache_{name}", command("q_cache", "--manifest", str(manifest),
                 "--cache-root", str(cache), "--output-dir", str(root / f"cache_{name}"),
                 "--workers", str(args.cache_workers)), root, progress)

    def predict(run_name: str, meta: Path, adapter_summary: dict) -> None:
        summary = root / f"predict_{run_name}" / "summary.json"
        if summary.is_file():
            return
        run_step(f"predict_{run_name}", command("q_predict", "--run-name", run_name,
                 "--manifest-meta", str(meta), "--cache-root", str(cache),
                 "--adapter", adapter_summary["best_checkpoint"],
                 "--adapter-sha256", adapter_summary["best_checkpoint_sha256"],
                 "--output-dir", str(root / f"predict_{run_name}")), root, progress)

    inference = root / "inference_manifests"
    for form in ("A", "V", "AV"):
        cache_manifest(f"fold0_infer_{form}", inference / f"fold0_{form}.jsonl")
    predict("fold0", inference / "run_meta.json", fold0)
    for index in range(1, 5):
        name = f"fold{index}"
        train_summary = root / name / "summary.json"
        if not train_summary.is_file():
            cache_manifest(f"{name}_train", root / "manifests" / f"{name}_train.jsonl")
            cache_manifest(f"{name}_val", root / "manifests" / f"{name}_val.jsonl")
            run_step(f"train_{name}", command("q_train", "--fold-name", name,
                     "--train-manifest", str(root / "manifests" / f"{name}_train.jsonl"),
                     "--val-manifest", str(root / "manifests" / f"{name}_val.jsonl"),
                     "--manifest-meta", str(root / "manifests" / "run_meta.json"),
                     "--cache-root", str(cache), "--output-dir", str(root / name)), root, progress)
        trained = read(train_summary)
        for form in ("A", "V", "AV"):
            cache_manifest(f"{name}_infer_{form}", inference / f"{name}_{form}.jsonl")
        predict(name, inference / "run_meta.json", trained)
        consumed = fold0["gpu_elapsed_s"] + sum(
            read(root / f"fold{i}" / "summary.json")["gpu_elapsed_s"]
            for i in range(1, index + 1))
        consumed += sum(step["elapsed_s"] for step in progress["steps"]
                        if step["name"].startswith("predict_"))
        progress["gpu_elapsed_h_upper_bound"] = consumed / 3600
        write_json_atomic(root / "pipeline_progress.json", progress)
        if consumed / 3600 > GPU_LIMIT_H:
            raise RuntimeError("Q GPU budget exceeded after completed fold")
    final_summary = root / "final" / "summary.json"
    if not final_summary.is_file():
        cache_manifest("final_train", root / "manifests" / "final_train.jsonl")
        cache_manifest("final_val", root / "manifests" / "final_val.jsonl")
        run_step("train_final", command("q_train", "--fold-name", "final",
                 "--train-manifest", str(root / "manifests" / "final_train.jsonl"),
                 "--val-manifest", str(root / "manifests" / "final_val.jsonl"),
                 "--manifest-meta", str(root / "manifests" / "run_meta.json"),
                 "--cache-root", str(cache), "--output-dir", str(root / "final")), root, progress)
    trained = read(final_summary)
    for form in ("A", "V", "AV"):
        cache_manifest(f"final_et_test_{form}", inference / f"final_et_test_{form}.jsonl")
    predict("final_et_test", inference / "run_meta.json", trained)
    sims_manifest = root / "sims_valid_manifests"
    for form in ("A", "V", "AV"):
        cache_manifest(f"final_sims_valid_{form}", sims_manifest / f"final_sims_valid_{form}.jsonl")
    predict("final_sims_valid", sims_manifest / "run_meta.json", trained)
    consumed = sum(read(root / name / "summary.json")["gpu_elapsed_s"]
                   for name in (*[f"fold{i}" for i in range(5)], "final"))
    consumed += sum(step["elapsed_s"] for step in progress["steps"]
                    if step["name"].startswith("predict_"))
    progress["gpu_elapsed_h_upper_bound"] = consumed / 3600
    progress["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    progress["completed"] = True
    write_json_atomic(root / "pipeline_progress.json", progress)
    if consumed / 3600 > GPU_LIMIT_H:
        raise RuntimeError("Completed Q pipeline exceeded 55 GPU hours")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=2)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
