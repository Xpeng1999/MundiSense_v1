from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from emotioncore.eval.metrics import bootstrap, classification, paired_bootstrap, percentile, sims_metrics
from emotioncore.utils.common import load_paths, read_jsonl, sha256_file, write_json_atomic


def load_predictions(manifest: Path, directory: Path, dataset: str, truth_kind: str = "M") -> list[dict]:
    rows = []
    for source in read_jsonl(manifest):
        sample_id = source["utt_id"] if dataset.startswith("et_") else f"{source['video_id']}__{source['clip_id']}"
        path = directory / f"{sample_id}.json"
        receipt = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        item = {"id": sample_id, "pred": receipt.get("pred") if receipt.get("status") == "ok" else None,
                "pred_with_other": receipt.get("pred_with_other") if receipt.get("status") == "ok" else None,
                "scores": receipt.get("scores") if receipt.get("status") == "ok" else None,
                "latency_s": receipt.get("latency_s"),
                "peak_cuda_allocated_mib": receipt.get("peak_cuda_allocated_mib"),
                "status": receipt.get("status", "missing_receipt"),
                "error": receipt.get("error")}
        if dataset.startswith("et_"):
            item.update(truth=source["labels"][truth_kind], speaker_id=source["speaker_id"])
        else:
            item["label"] = source["label"]
            if item["scores"] and len(item["scores"]) == 8:
                p = item["scores"]
                item["s"] = p[1] - (p[3] + p[5] + p[4] + p[2])
            else:
                item["s"] = None
        rows.append(item)
    return rows


def et_result(rows: list[dict], replicates: int, seed: int) -> dict:
    result = classification(rows)
    result["failure_reasons"] = dict(collections.Counter(
        (row.get("error") or row["status"]).split(":", 1)[0]
        for row in rows if row["status"] != "ok"))
    for metric in ("macro_f1", "uar", "wa"):
        result[f"{metric}_utterance_ci"] = bootstrap(rows, lambda sample: classification(sample)[metric],
                                                      replicates, seed)
        result[f"{metric}_speaker_ci"] = bootstrap(rows, lambda sample: classification(sample)[metric],
                                                    replicates, seed, group_key="speaker_id")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["speaker_id"]].append(row)
    result["by_speaker"] = {speaker: classification(sample) for speaker, sample in sorted(groups.items())}
    result["runtime"] = runtime(rows)
    return result


def runtime(rows: list[dict]) -> dict:
    latencies = [row["latency_s"] for row in rows if row["latency_s"] is not None]
    peaks = [row["peak_cuda_allocated_mib"] for row in rows if row["peak_cuda_allocated_mib"] is not None]
    return {"latency_p50_s": percentile(latencies, 0.5) if latencies else None,
            "latency_p95_s": percentile(latencies, 0.95) if latencies else None,
            "peak_cuda_allocated_mib": max(peaks) if peaks else None}


def legacy_correction(legacy: Path, leakage: Path) -> dict:
    selected = json.loads((legacy / "av_attribution_v16/artifacts/selected_chinese.json").read_text(encoding="utf-8"))
    leak = json.loads(leakage.read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in leak["overlap_details"]}
    root = legacy / "av_attribution_v16/artifacts/chinese/predictions/qwen_old"
    output = {"leakage_counts": leak["overlap_counts"], "groups": {}}
    for condition in ("audio", "video", "av"):
        groups = {"clean_all_levels": [], "leaked_any_level": [],
                  "same_utterance": [], "speaker_only": []}
        for source in selected:
            sample_id = source["id"]
            path = root / f"{sample_id}__{condition}.json"
            receipt = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            row = {"id": sample_id, "truth": source["references"]["av"]["truth"],
                   "pred": receipt.get("pred") if receipt.get("status") == "ok" else None}
            hits = by_id.get(sample_id)
            if hits:
                groups["leaked_any_level"].append(row)
                if hits["utterance"]:
                    groups["same_utterance"].append(row)
                elif hits["speaker"]:
                    groups["speaker_only"].append(row)
            else:
                groups["clean_all_levels"].append(row)
        output["groups"][condition] = {name: classification(rows) for name, rows in groups.items()}
    return output


def analyze(paths_config: Path, results: Path, replicates: int, seed: int) -> None:
    paths = load_paths(paths_config)
    all_rows = {}
    metrics = {"et": {}, "sims_valid": {}, "paired": {}, "native_labels": {}, "run_meta": {},
               "bootstrap_replicates": replicates, "seed": seed,
               "speaker_interval_note": "ET test has four speakers; speaker bootstrap intervals are descriptive."}
    prediction_paths = []
    for model in ("base", "old_lora"):
        for dataset in ("et_test", "et_val", "sims_valid"):
            conditions = ("av", "audio", "video") if dataset.startswith("et_") else ("av", "audio")
            for condition in conditions:
                name = f"{model}_{dataset}_{condition}"
                directory = paths["run_root"] / "inference" / name
                meta = json.loads((directory / "run_meta.json").read_text(encoding="utf-8"))
                if meta["ended_at"] is None or meta["completed"] != meta["planned"]:
                    raise RuntimeError(f"Incomplete run: {name}")
                prediction_dir = directory / "predictions"
                rows = load_predictions(results / f"{dataset}.jsonl", prediction_dir, dataset)
                if len(rows) != meta["planned"]:
                    raise RuntimeError(f"Manifest/receipt count mismatch: {name}")
                all_rows[name] = rows
                if dataset.startswith("et_"):
                    metrics["et"][name] = et_result(rows, replicates, seed)
                    if condition in ("audio", "video"):
                        truth_kind = "A" if condition == "audio" else "V"
                        native = load_predictions(results / f"{dataset}.jsonl", prediction_dir,
                                                  dataset, truth_kind)
                        metrics["native_labels"][name] = classification(native)
                    metrics["et"][name]["with_other"] = classification(rows, pred_key="pred_with_other")
                else:
                    metrics["sims_valid"][name] = sims_metrics(rows)
                    metrics["sims_valid"][name]["runtime"] = runtime(rows)
                    for metric in ("pearson", "spearman", "acc_2_nonzero"):
                        if metrics["sims_valid"][name][metric] is not None:
                            metrics["sims_valid"][name][f"{metric}_utterance_ci"] = bootstrap(
                                rows, lambda sample, m=metric: sims_metrics(sample)[m] or 0.0,
                                replicates, seed)
                    metrics["sims_valid"][name]["failure_reasons"] = dict(collections.Counter(
                        (row.get("error") or row["status"]).split(":", 1)[0]
                        for row in rows if row["status"] != "ok"))
                metrics["run_meta"][name] = {"path": str(directory / "run_meta.json"),
                                             "sha256": sha256_file(directory / "run_meta.json")}
                for path in sorted(prediction_dir.glob("*.json")):
                    prediction_paths.append({"path": str(path), "sha256": sha256_file(path)})
    for dataset in ("et_test", "et_val"):
        for model in ("base", "old_lora"):
            for left, right in (("av", "audio"), ("av", "video"), ("audio", "video")):
                a, b = f"{model}_{dataset}_{left}", f"{model}_{dataset}_{right}"
                metrics["paired"][f"{a}_minus_{b}"] = paired_bootstrap(all_rows[a], all_rows[b],
                                                                          repetitions=replicates, seed=seed)
        for condition in ("av", "audio", "video"):
            a, b = f"old_lora_{dataset}_{condition}", f"base_{dataset}_{condition}"
            metrics["paired"][f"{a}_minus_{b}"] = paired_bootstrap(all_rows[a], all_rows[b],
                                                                      repetitions=replicates, seed=seed)
    write_json_atomic(results / "metrics.json", metrics)
    write_json_atomic(results / "prediction_manifest.json", prediction_paths)
    write_json_atomic(results / "legacy_correction.json",
                      legacy_correction(paths["legacy_root"], results / "leakage_v16.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    analyze(args.paths, args.results, args.replicates, args.seed)


if __name__ == "__main__":
    main()
