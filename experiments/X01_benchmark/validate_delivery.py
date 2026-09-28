"""Check X01's frozen inputs, completed runs, and report before review."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from emotioncore.utils.common import load_paths, read_jsonl, sha256_file, write_json_atomic


def validate(paths_config: Path, results: Path, report: Path) -> dict:
    paths = load_paths(paths_config)
    checks: dict[str, bool] = {}
    read = lambda name: json.loads((results / name).read_text(encoding="utf-8"))
    et = read("et_audit.json")
    sims = read("chsims_audit.json")
    decode = read("chsims_decode_audit.json")
    freeze = read("benchmark_freeze.json")
    d3 = read("d3_consistency.json")
    v16 = read("leakage_v16.json")
    metrics = read("metrics.json")
    faces = read("face_audit.json")
    predictions = read("prediction_manifest.json")
    tests = read("test_results.json")
    legacy = read("legacy_snapshot_audit.json")
    environment = read("environment_compare.json")
    new_environment = read("new_venv_verification.json")
    incidents = read("d2_incidents.json")

    checks["et_rows_and_official_counts"] = et["rows"] == 19250 and not et["official_differences"]
    checks["sims_rows_and_decode_coverage"] = (
        sims["rows"] == decode["rows"] == 4403
        and sum(sum(counts.values()) for counts in decode["counts_by_mode"].values()) == 4403
        and sum(counts.get("missing", 0) for counts in decode["counts_by_mode"].values()) == sims["missing_count"]
    )
    checks["face_audit_coverage"] = (
        faces["emotiontalk"]["clips"] == 5837
        and faces["chsims_v2"]["clips"] == 1647
        and all(abs(sum(faces[name]["frames"]["fractions"].values()) - 1) < 1e-9
                for name in ("emotiontalk", "chsims_v2"))
    )
    checks["legacy_leak_reproduction"] = (v16["overlap_counts"]["utterance"] == 161
                                          and v16["overlap_counts"]["speaker"] == 168)
    checks["et_splits_clean"] = all(
        all(value == 0 for value in read(f"leakage_{split}.json")["overlap_counts"].values())
        for split in ("et_test", "et_val")
    )
    checks["d3_passed"] = d3["passed"] and d3["completed"] == d3["required"] == 40
    checks["definition_unchanged"] = freeze["definition_sha256"] == sha256_file(Path(freeze["definition"]))
    checks["manifests_unchanged"] = all(
        entry["sha256"] == sha256_file(Path(entry["path"]))
        and entry["rows"] == sum(1 for _ in read_jsonl(Path(entry["path"])))
        for entry in freeze["manifests"].values()
    )
    checks["lockbox_has_no_labels"] = all(
        set(row) == {"video_id", "clip_id", "media"}
        and set(row["media"]) == {"path", "sha256"}
        for row in read_jsonl(results / "sims_test_lockbox.jsonl")
    )
    checks["no_lockbox_inference"] = not any(
        "sims_test" in path.name for path in (paths["run_root"] / "inference").glob("*")
    )
    checks["all_16_runs"] = len(metrics["et"]) == 12 and len(metrics["sims_valid"]) == 4
    checks["prediction_manifest_complete"] = len(predictions) == sum(
        row["n"] for section in ("et", "sims_valid") for row in metrics[section].values()
    )
    expected_modality = {
        "audio": {"processor": False, "forward": False, "has_audio": True, "has_video": False},
        "video": {"processor": False, "forward": False, "has_audio": False, "has_video": True},
        "av": {"processor": True, "forward": True, "has_audio": True, "has_video": True},
    }
    seven_labels = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise")
    hashes_valid = True
    modality_valid = True
    for entry in predictions:
        payload = Path(entry["path"]).read_bytes()
        hashes_valid &= hashlib.sha256(payload).hexdigest() == entry["sha256"]
        receipt = json.loads(payload)
        if receipt["status"] == "ok":
            modality_valid &= (len(receipt["raw_logprobs"]) == 8
                               and len(receipt["scores"]) == 8
                               and abs(sum(receipt["scores"]) - 1) < 1e-5
                               and receipt["modality_switch"] == expected_modality.get(receipt["condition"])
                               and receipt["pred"] in seven_labels
                               and receipt["pred_with_other"] in (*seven_labels, "other"))
    checks["prediction_hashes_valid"] = hashes_valid
    checks["modality_and_eight_scores_valid"] = modality_valid
    def metadata_complete(item: dict) -> bool:
        row = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        return row["completed"] == row["planned"] and row["ended_at"] is not None

    checks["run_metadata_complete"] = len(metrics["run_meta"]) == 16 and all(
        metadata_complete(item) for item in metrics["run_meta"].values()
    )
    checks["tests_passed"] = tests["exit_code"] == 0 and tests["failed"] == 0
    checks["environment_locked"] = (environment["mismatch_count"] == 0
                                    and new_environment["mismatch_count"] == 0
                                    and new_environment["cuda_available"])
    checks["incident_receipts_verified"] = all(
        sha256_file(Path(item["receipt_path"])) == item["receipt_sha256"]
        for item in incidents["incidents"]
    )
    headings = [f"## {n}." for n in range(1, 9)]
    checks["report_sections"] = all(heading in report.read_text(encoding="utf-8") for heading in headings)
    forbidden = {"text", "transcript", "utterance_text"}
    checks["manifests_without_transcripts"] = all(
        not forbidden.intersection(row)
        for name in ("et_utterances", "chsims_v2_clips", "et_test", "et_val", "sims_valid")
        for row in read_jsonl(results / f"{name}.jsonl")
    )
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    blocked_extensions = {".mp4", ".mov", ".avi", ".wav", ".flac", ".mp3",
                          ".zip", ".pt", ".pth", ".safetensors", ".bin"}
    checks["no_media_or_weights_tracked"] = all(Path(name).suffix.lower() not in blocked_extensions
                                                   for name in tracked)
    boundary_passed = all(legacy[key] == 0 for key in
                          ("added_count", "removed_count", "changed_existing_count"))
    return {"checks": checks, "functional_checks_passed": all(checks.values()),
            "legacy_boundary_passed": boundary_passed,
            "review_required": not boundary_passed or not all(checks.values()),
            "legacy_added_files": legacy["added_count"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate(args.paths, args.results, args.report)
    write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not receipt["functional_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
