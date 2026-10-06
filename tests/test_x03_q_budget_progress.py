import json
from pathlib import Path

import pytest

from emotioncore.x03.q_budget_progress import project, record


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_completed_fold_time_changes_projection_and_stops_before_next_model(tmp_path):
    forms = {name: {"train_step_mean_s": 1.0, "inference_mean_s": 0.1}
             for name in ("A", "V", "AV")}
    write(tmp_path / "gc_step_benchmark/summary.json", {"forms": forms})
    write(tmp_path / "fold0/summary.json", {"gpu_elapsed_s": 3600})
    write(tmp_path / "fold1/summary.json", {"gpu_elapsed_s": 3600})
    write(tmp_path / "manifests/summary.json", {
        name: {"assigned_counts": {"A": 100, "V": 100, "AV": 100}}
        for name in ("fold0", "fold1", "fold2", "fold3", "fold4", "final")})
    for name in ("fold1", "fold2", "fold3", "fold4", "final"):
        path = tmp_path / "manifests" / f"{name}_train.jsonl"
        path.write_text("".join(json.dumps({"effective_form": form}) + "\n"
                                for form in ("A", "V", "AV") for _ in range(100)))
    write(tmp_path / "inference_manifests/summary.json", {
        f"{run}_{form}": {"n": 10}
        for run in ("fold0", "fold1", "fold2", "fold3", "fold4", "final_et_test")
        for form in forms})
    write(tmp_path / "sims_valid_manifests/summary.json", {
        f"final_sims_valid_{form}": {"n": 10} for form in forms})
    progress = {"steps": [{"name": "train_fold1", "elapsed_s": 3600}]}
    first = project(tmp_path, progress, "fold1")
    assert first["checkpointed_actual_to_benchmark_ratio"] > 1
    assert first["projected_total_gpu_h"] > 2
    progress["steps"][0]["elapsed_s"] = 300_000
    with pytest.raises(RuntimeError, match="stop before next model"):
        record(tmp_path, progress, "fold1")
    assert json.loads((tmp_path / "pipeline_stop.json").read_text())["reason"] == "projected_total_exceeds_75h"
