"""Idempotent X01 pipeline; only approved cohorts reach Qwen."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from emotioncore.data.benchmark import build as build_benchmark
from emotioncore.data.chsims import build as build_chsims
from emotioncore.data.decode_audit import run as run_decode_audit
from emotioncore.data.emotiontalk import build as build_et
from emotioncore.eval.analyze import analyze
from emotioncore.eval.consistency import run as run_consistency
from emotioncore.eval.run_baselines import SAMPLE_TIMEOUT_S, run as run_baselines
from emotioncore.tools.leakage_check import check
from emotioncore.utils.common import load_paths, sha256_file, write_json_atomic


def run(paths_config: Path, experiment_config: Path, with_face_audit: bool) -> None:
    import yaml

    paths = load_paths(paths_config)
    results = Path("experiments/X01_benchmark/results")
    results.mkdir(parents=True, exist_ok=True)
    write_json_atomic(results / "execution_config.json",
                      {"source": str(experiment_config),
                       "sha256": sha256_file(experiment_config),
                       "per_sample_timeout_s": SAMPLE_TIMEOUT_S,
                       "values": yaml.safe_load(experiment_config.read_text(encoding="utf-8"))})
    extracted = paths["derived_root"] / "chsims_v2/raw"
    if not (extracted / ".extract_complete").is_file():
        subprocess.run(["bash", "experiments/X01_benchmark/extract_chsims.sh",
                        str(paths["source_data_root"] / "CH-SIMSV2"),
                        str(paths["derived_root"] / "chsims_v2/raw")], check=True)
    if not (results / "et_utterances.jsonl").is_file():
        build_et(paths_config, results / "et_utterances.jsonl", workers=8)
    if not (results / "chsims_v2_clips.jsonl").is_file():
        build_chsims(paths_config, results / "chsims_v2_clips.jsonl", workers=8)
    if not (results / "chsims_decode_audit.json").is_file():
        run_decode_audit(paths_config, results / "chsims_v2_clips.jsonl",
                         results / "chsims_decode_audit.json")
    if not (results / "benchmark_manifests.json").is_file():
        build_benchmark(results)
    write_json_atomic(results / "benchmark_freeze.json",
                      {"definition": "docs/BENCHMARK_v1.md",
                       "definition_sha256": sha256_file(Path("docs/BENCHMARK_v1.md")),
                       "manifests": json.loads(
                           (results / "benchmark_manifests.json").read_text(encoding="utf-8"))})
    train = [paths["legacy_root"] / "data/manifests_t0/swift/train_e3_core_av.jsonl",
             paths["legacy_root"] / "data/manifests_t0/swift/train_e3_core_audio.jsonl"]
    legacy_chinese = paths["legacy_root"] / "av_attribution_v16/artifacts/selected_chinese.json"
    if not (results / "leakage_v16.json").is_file():
        write_json_atomic(results / "leakage_v16.json", check(train, legacy_chinese))
    for split in ("et_test", "et_val"):
        target = results / f"leakage_{split}.json"
        if not target.is_file():
            write_json_atomic(target, check(train, results / f"{split}.jsonl",
                                            eval_media_root=paths["processed_root"] / "emotiontalk"))
    if not (results / "d3_consistency.json").is_file():
        run_consistency(paths_config, results / "d3_consistency.json")
    run_baselines(paths_config, experiment_config, results)
    analyze(paths_config, results, replicates=2000, seed=20260928)
    if with_face_audit:
        from emotioncore.data.face_audit import run as run_face_audit
        if not (results / "face_audit.json").is_file():
            run_face_audit(paths_config, results, results / "face_audit.json")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/x01.yaml"))
    parser.add_argument("--with-face-audit", action="store_true")
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VIDEO_MAX_PIXELS=str(experiment["video_max_pixels"]),
                      FPS_MAX_FRAMES=str(experiment["fps_max_frames"]),
                      MAX_PIXELS=str(experiment["max_pixels"]), TOKENIZERS_PARALLELISM="false")
    run(args.paths, args.config, args.with_face_audit)


if __name__ == "__main__":
    main()
