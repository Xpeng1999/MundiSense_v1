"""Estimate X04b from frozen X03 measurements; never load a learned model."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
import numpy as np
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.q_manifest import FOLD_SHA, SUBSET_SHA, read_jsonl, sha256
from emotioncore.x04.prepare import storage_gate

FORMS = ("A", "V", "AV")
RATIOS = {"A": .2, "V": .2, "AV": .6}


def conservative_curve(samples, lengths, field):
    """Monotone upper envelope; quadratic extrapolation beyond measured length.

    Seven X03 probes per form do not identify a validated latency law. This
    deliberately retains cold/slow observations and labels extrapolation.
    """
    points = sorted((s["input_token_length"], s[field]) for s in samples)
    x = np.array(sorted({p[0] for p in points}), dtype=float)
    y = np.maximum.accumulate([max(t for length, t in points if length == v) for v in x])
    lengths = np.asarray(lengths, dtype=float)
    result = np.interp(lengths, x, y, left=y[0], right=y[-1])
    high = lengths > x[-1]
    result[high] = y[-1]*(lengths[high]/x[-1])**2
    return result, {"measured_min_length": int(x[0]), "measured_max_length": int(x[-1]),
                    "extrapolated_count": int(high.sum()), "upper_envelope_seconds": y.tolist(),
                    "measured_lengths": x.astype(int).tolist()}


def distribution(values):
    return {"n": len(values), **dict(zip(("min", "p10", "p50", "p90", "p95", "p99", "max"),
                np.quantile(values, [0, .1, .5, .9, .95, .99, 1]).tolist()))}


def run(args):
    git = git_provenance(Path.cwd()); storage_gate()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    if sha256(Path("configs/folds_v1.json")) != FOLD_SHA or sha256(Path("configs/x03_inner_val_subsets.json")) != SUBSET_SHA:
        raise ValueError("Frozen Q fold/subset protocol changed")
    benchmark_path = args.r3/"Q/gc_step_benchmark/summary.json"
    benchmark = json.loads(benchmark_path.read_text())
    if not all(benchmark["gradient_checkpointing"][key] for key in ("llm", "vision", "audio")):
        raise ValueError("Budget requires checkpointed X03 benchmark")
    budget_path = args.r3/"Q/q_budget_progress.jsonl"
    last_budget = read_jsonl(budget_path)[-1]
    observed_ratio = max(1., last_budget["checkpointed_actual_to_benchmark_ratio"])
    qmeta = json.loads((args.derived/"Q_media/summary.json").read_text())
    token_distribution = {}; costs = {}; extrapolations = {}; provenance = {}
    train_ids = None
    for form in FORMS:
        spec = qmeta["outputs"]["train_"+form]; path = Path(spec["path"])
        if sha256(path) != spec["sha256"]: raise ValueError("MER Q budget inputs changed")
        rows = read_jsonl(path)
        ids = {r["utt_id"] for r in rows}
        if len(ids) != len(rows) or train_ids is not None and ids != train_ids:
            raise ValueError("Unequal MER budget form coverage")
        train_ids = ids
        lengths = []; step = []; inference = []; counts = {"A": 0, "V": 0, "AV": 0}
        for i, r in enumerate(rows):
            if i % 100 == 0: storage_gate()
            if r["split"] != "mer_train": raise ValueError("Only deduplicated MER-train enters budget")
            effective = r["effective_form"]
            receipt_path = args.derived/"Q_inputs"/r["utt_id"]/(effective+".json")
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] != "ok" or receipt["input_wav_sha256"] != r["wav_sha256"] or receipt.get("input_video_sha256") != r.get("video_sha256"):
                raise ValueError("MER processor receipt mismatch")
            length = receipt["shapes"]["input_ids"][-1]
            lengths.append(length); counts[effective] += 1
            measured = benchmark["forms"][effective]["samples"]
            train_time, extra = conservative_curve(measured, [length], "train_step_s")
            infer_time, _ = conservative_curve(measured, [length], "inference_s")
            step.append(train_time[0]); inference.append(infer_time[0])
            extrapolations[form] = extrapolations.get(form, 0) + extra["extrapolated_count"]
        token_distribution[form] = {**distribution(lengths), "by_effective_form": counts}
        costs[form] = {"train_all_MER_one_epoch_s": float(sum(step)),
                       "inference_all_MER_s": float(sum(inference))}
        provenance[form] = spec
    mer_epoch = sum(RATIOS[f]*costs[f]["train_all_MER_one_epoch_s"] for f in FORMS)
    qmanifest = json.loads((args.r3/"Q/manifests/summary.json").read_text())
    inference = json.loads((args.r3/"Q/inference_manifests/summary.json").read_text())
    sims = json.loads((args.r3/"Q/sims_valid_manifests/summary.json").read_text())
    runs = {}
    for name in ("fold0", "fold1", "fold2", "fold3", "fold4", "final"):
        spec = qmanifest[name]
        train_path = Path(spec["train_path"]); val_path = Path(spec["val_path"])
        if sha256(train_path) != spec["train_sha256"] or sha256(val_path) != spec["val_sha256"]:
            raise ValueError("X03 frozen training/validation manifest changed")
        train = read_jsonl(train_path); val = read_jsonl(val_path)
        if len(val) != 700: raise ValueError("Validation must retain 700 ET sentences")
        et_epoch = sum(benchmark["forms"][r["effective_form"]]["train_step_mean_s"] for r in train)
        validation = 8*sum(benchmark["forms"][r["effective_form"]]["inference_mean_s"] for r in val)
        prediction = 0.
        for prefix in (("final_et_test", "final_sims_valid") if name == "final" else (name,)):
            source = sims if prefix == "final_sims_valid" else inference
            prediction += sum(source[prefix+"_"+f]["n"]*benchmark["forms"][f]["inference_mean_s"] for f in FORMS)
        if name == "final":
            # Conservative mean train-duration MER cost, 572 report-only targets.
            prediction += 572/len(train_ids)*(costs["A"]["inference_all_MER_s"]+costs["AV"]["inference_all_MER_s"])
        runs[name] = {"ET_train_n": len(train), "MER_train_n": len(train_ids),
            "epochs": 2, "validation_n": 700, "validation_calls": 8,
            "training_validation_s_before_observed_ratio": 2*(et_epoch+mer_epoch)+validation,
            "prediction_s_before_observed_ratio": prediction}
    plans = {}
    for name, selected in (("five_folds_plus_final", list(runs)), ("fold0_fold1_plus_final", ["fold0", "fold1", "final"])):
        train_s = sum(runs[k]["training_validation_s_before_observed_ratio"] for k in selected)*observed_ratio
        pred_s = sum(runs[k]["prediction_s_before_observed_ratio"] for k in selected)*observed_ratio
        plans[name] = {"models": selected, "training_validation_hours": train_s/3600,
            "prediction_hours": pred_s/3600, "reserve_fraction": .2,
            "load_checkpoint_reserve_hours": .5, "diagnostic_reserve_hours": 1.,
            "total_gpu_hours": (train_s+pred_s)*1.2/3600+.5+1.}
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"Q_training": False, "estimate_only": True, "forms_ratio": RATIOS,
        "method": "X03 per-form mean for ET; monotone upper envelope for MER; quadratic extrapolation above measured maximum; then observed checkpointed actual/benchmark ratio",
        "limitation": "Only seven X03 probes per form; upper envelope and extrapolation are conservative planning assumptions, not validated MER training timings",
        "observed_actual_to_benchmark_ratio": observed_ratio, "token_lengths": token_distribution,
        "extrapolated_counts": extrapolations, "Q_media_face_missing": qmeta["by_split"],
        "costs": costs, "runs": runs, "plans": plans,
        "source_benchmark": {"path": str(benchmark_path), "sha256": sha256(benchmark_path)},
        "source_actual_budget": {"path": str(budget_path), "sha256": sha256(budget_path)},
        "source_MER_manifests": provenance}
    write_json_atomic(args.output/"x04b_budget.json", result)
    write_json_atomic(args.output/"run_meta.json", {"git": git, "command": sys.argv,
        "started_at": started, "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(), "learned_inference": False,
        "result_sha256": sha256(args.output/"x04b_budget.json")})


def main():
    p = argparse.ArgumentParser(); p.add_argument("--derived", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--r3", type=Path, default=Path("/home/administrator/emotion-data/emotioncore_runs/X03"))
    run(p.parse_args())


if __name__ == "__main__": main()
