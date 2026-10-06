"""Re-estimate the frozen Q schedule after reviewer-approved 700-row validation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SUBSET_SHA = "be2cd442346cfa978e22bea8ab2ae3776d52187b2581fe93e377fc4fb58d3cfd"
FACE_SHA = "ce65415b5f7129b62ef5edc7eda5b20874ed88931f5a7cc295c276b8b704b77c"
GPU_LIMIT_H = 75.0  # Reviewer approval after the checkpointed-step re-estimate.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(smoke: dict, subset: dict) -> dict:
    if len(subset["folds"]) != 5 or any(f["n"] != 700 for f in subset["folds"]):
        raise ValueError("Frozen 700-row validation subsets differ")
    steps = smoke["steps"]
    validation_examples = 8 * (sum(f["n"] for f in subset["folds"]) + subset["folds"][0]["n"])
    validation_h = validation_examples * steps["AV"]["inference_s"] / 3600
    total_h = smoke["projected_training_h"] + validation_h + smoke["projected_prediction_h"]
    return {"gpu_limit_h": GPU_LIMIT_H, "projected_training_h": smoke["projected_training_h"],
            "projected_validation_h": validation_h,
            "projected_prediction_h": smoke["projected_prediction_h"],
            "projected_total_h_without_loading_or_preprocessing": total_h,
            "validation_examples": validation_examples,
            "validation_frequency": "every_0.25_epoch_AV_only",
            "status": "stop" if total_h > GPU_LIMIT_H else "within_budget_estimate",
            "assumptions": "Six runs; five outer folds plus final; at most two epochs; batch-one measured per-example times with effective batch eight via accumulation; no hyperparameter scan"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.subsets) != SUBSET_SHA or sha256(args.face_policy) != FACE_SHA:
        raise ValueError("Reviewer-approved frozen protocol SHA mismatch")
    payload = estimate(json.loads(args.smoke.read_text()), json.loads(args.subsets.read_text()))
    payload.update({"source_smoke_sha256": sha256(args.smoke),
                    "subset_sha256": SUBSET_SHA, "face_policy_sha256": FACE_SHA,
                    "computed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
