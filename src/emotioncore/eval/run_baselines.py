from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from contextlib import contextmanager

from emotioncore.models.qwen_omni import QwenOmniScorer
from emotioncore.utils.common import load_paths, read_jsonl, sha256_file, write_json_atomic


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextmanager
def sample_deadline(seconds: int = 300):
    """Bound a single sample's media preprocessing and forward pass."""
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError(f"Sample processing exceeded {seconds} seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def tree_listing_sha256(root: Path | None) -> str | None:
    if root is None:
        return None
    digest = hashlib.sha256()
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}\t{stat.st_size}\n".encode())
    return digest.hexdigest()


def git_state(root: Path) -> dict:
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def prompt_pair(legacy_root: Path, condition: str) -> tuple[str, str]:
    name = "train_e3_core_audio.jsonl" if condition == "audio" else "train_e3_core_av.jsonl"
    path = legacy_root / "data/manifests_t0/swift" / name
    row = json.loads(path.open(encoding="utf-8").readline())
    return row["messages"][0]["content"], row["messages"][1]["content"]


def sims_audio(source: Path, output: Path) -> Path:
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.stem + ".tmp.wav")
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                             "-vn", "-ac", "1", "-ar", "16000", str(tmp)],
                            capture_output=True, text=True)
    if result.returncode:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip()[:500] or "ffmpeg audio extraction failed")
    os.replace(tmp, output)
    return output


def media_for(row: dict, dataset: str, condition: str, paths: dict[str, Path]) -> Path:
    if dataset.startswith("et_"):
        root = paths["processed_root"] / "emotiontalk"
        return root / row["wav" if condition == "audio" else "mp4"]["path"]
    video = paths["derived_root"] / "chsims_v2/raw" / row["media"]["path"]
    if condition == "audio":
        audio = paths["derived_root"] / "chsims_v2/audio_valid" / row["video_id"] / f"{row['clip_id']}.wav"
        return sims_audio(video, audio)
    return video


def run_one(model: QwenOmniScorer, model_name: str, dataset: str, condition: str,
            manifest: Path, paths: dict[str, Path], config: Path, root: Path,
            model_hash: str, adapter_hash: str | None) -> dict:
    import torch

    rows = list(read_jsonl(manifest))
    if dataset == "sims_valid" and any(r["mode"] != "valid" for r in rows):
        raise ValueError("SIMS test lock-box row reached inference")
    if dataset not in ("et_test", "et_val", "sims_valid"):
        raise ValueError(f"Dataset not approved for D2: {dataset}")
    system, user = prompt_pair(paths["legacy_root"], condition)
    directory = paths["run_root"] / "inference" / f"{model_name}_{dataset}_{condition}"
    predictions = directory / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "run_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["manifest_sha256"] != sha256_file(manifest):
            raise RuntimeError("Resume manifest SHA-256 mismatch")
    else:
        meta = {"git": git_state(root), "command": sys.argv,
                "configuration": str(config), "configuration_sha256": sha256_file(config),
                "model_listing_sha256": model_hash, "adapter_listing_sha256": adapter_hash,
                "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                "torch": torch.__version__, "transformers": __import__("transformers").__version__,
                "gpu": torch.cuda.get_device_name(0), "started_at": now(), "ended_at": None,
                "model": model_name, "dataset": dataset, "condition": condition,
                "planned": len(rows), "completed": 0, "failures": 0,
                "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
                "user_prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
                "adapter_info": model.adapter_info}
        write_json_atomic(meta_path, meta)
    failures = 0
    for index, row in enumerate(rows):
        sample_id = row["utt_id"] if dataset.startswith("et_") else f"{row['video_id']}__{row['clip_id']}"
        target = predictions / f"{sample_id}.json"
        if target.is_file():
            continue
        start = time.monotonic()
        receipt = {"id": sample_id, "model": model_name, "dataset": dataset,
                   "condition": condition, "status": "ok", "at": now()}
        try:
            with sample_deadline():
                media = media_for(row, dataset, condition, paths)
                if not media.is_file():
                    raise FileNotFoundError(str(media))
                receipt.update(model.score_file(system, user, media, condition))
        except Exception as exc:
            failures += 1
            receipt.update(status="failed", pred=None, pred_with_other=None,
                           error=f"{type(exc).__name__}: {str(exc)[:500]}")
        receipt["latency_s"] = time.monotonic() - start
        write_json_atomic(target, receipt)
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            meta.update(completed=sum(p.is_file() for p in predictions.glob("*.json")),
                        failures=failures, updated_at=now())
            write_json_atomic(meta_path, meta)
    meta.update(completed=sum(p.is_file() for p in predictions.glob("*.json")),
                failures=sum(json.loads(p.read_text(encoding="utf-8"))["status"] != "ok"
                             for p in predictions.glob("*.json")), ended_at=now())
    write_json_atomic(meta_path, meta)
    return meta


def run(config: Path, x01_config: Path, results: Path) -> None:
    import torch

    paths = load_paths(config)
    d3 = json.loads((results / "d3_consistency.json").read_text(encoding="utf-8"))
    if not d3.get("passed") or d3.get("completed") != 40:
        raise RuntimeError("D3 consistency gate has not passed 40/40")
    for split in ("et_test", "et_val"):
        leak = json.loads((results / f"leakage_{split}.json").read_text(encoding="utf-8"))
        if any(leak["overlap_counts"][key] for key in ("utterance", "media_sha256", "speaker", "dialogue")):
            raise RuntimeError(f"Training leakage in {split}; do not run D2")
    model_hash = tree_listing_sha256(paths["model_root"])
    adapter_hash = tree_listing_sha256(paths["adapter_root"])
    root = Path.cwd()
    for model_name in ("base", "old_lora"):
        adapter = None if model_name == "base" else paths["adapter_root"]
        model = QwenOmniScorer(paths["model_root"], adapter)
        for dataset in ("et_test", "et_val", "sims_valid"):
            conditions = ("av", "audio", "video") if dataset.startswith("et_") else ("av", "audio")
            for condition in conditions:
                run_one(model, model_name, dataset, condition, results / f"{dataset}.jsonl",
                        paths, x01_config, root, model_hash, adapter_hash if adapter else None)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VIDEO_MAX_PIXELS=str(experiment["video_max_pixels"]),
                      FPS_MAX_FRAMES=str(experiment["fps_max_frames"]),
                      MAX_PIXELS=str(experiment["max_pixels"]),
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    run(args.paths, args.config, args.results)


if __name__ == "__main__":
    main()
