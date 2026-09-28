"""Load only the local Qwen2.5-Omni audio tower and cache frame states."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


ENCODER = "A2_qwen2.5_omni_audio_tower"
PREPROCESS_VERSION = "1"
TARGET_SAMPLE_RATE = 16000
EXCLUDED_SPEAKERS = frozenset({"02", "13", "20", "21"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audio_shards(model_dir: Path) -> list[Path]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    names = {value for key, value in index["weight_map"].items()
             if key.startswith("thinker.audio_tower.")}
    if not names:
        raise ValueError("No Qwen audio tower tensors in local model index")
    return [model_dir / name for name in sorted(names)]


def load_audio_tower(model_dir: Path):
    import torch
    from safetensors import safe_open
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniProcessor
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder

    config = Qwen2_5OmniConfig.from_pretrained(str(model_dir), local_files_only=True)
    tower = Qwen2_5OmniAudioEncoder(config.thinker_config.audio_config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    audio = {key: shard for key, shard in index.items()
             if key.startswith("thinker.audio_tower.")}
    state = {}
    for shard in sorted(set(audio.values())):
        with safe_open(model_dir / shard, framework="pt", device="cpu") as stream:
            for key, source in audio.items():
                if source == shard:
                    state[key.removeprefix("thinker.audio_tower.")] = stream.get_tensor(key)
    tower.load_state_dict(state, strict=True)
    del state
    tower = tower.to(device="cuda:0", dtype=torch.bfloat16).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
    return tower, processor.feature_extractor


def worker_main(model_dir: Path) -> None:
    import numpy as np
    import scipy.signal
    import soundfile as sf
    import torch

    tower, extractor = load_audio_tower(model_dir)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            started = time.monotonic()
            wav = Path(task["wav"])
            if sha256(wav) != task["input_sha256"]:
                raise ValueError("Input WAV sha256 differs from frozen manifest")
            signal, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
            signal = signal.mean(axis=1)
            if sample_rate != TARGET_SAMPLE_RATE:
                divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
                signal = scipy.signal.resample_poly(signal,
                                                    TARGET_SAMPLE_RATE // divisor,
                                                    sample_rate // divisor)
            if not len(signal) or not np.isfinite(signal).all():
                raise ValueError("Empty or non-finite audio")
            features = extractor(signal, sampling_rate=TARGET_SAMPLE_RATE,
                                 return_tensors="pt", return_attention_mask=True,
                                 padding="max_length")
            mask = features["attention_mask"].bool()
            mel_length = mask.sum(dim=1).to("cuda:0")
            mel = features["input_features"].permute(0, 2, 1)[mask].permute(1, 0)
            mel = mel.to("cuda:0")
            aftercnn_lens, _ = tower._get_feat_extract_output_lengths(mel_length)
            with torch.inference_mode():
                hidden = tower(mel, feature_lens=mel_length,
                               aftercnn_lens=aftercnn_lens).last_hidden_state
            frames = hidden.float().cpu().numpy()
            if frames.ndim != 2 or frames.shape[1] != 3584 or not len(frames) or not np.isfinite(frames).all():
                raise ValueError(f"Unexpected Qwen audio shape: {frames.shape}")
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.npz")
            np.savez(temporary, frames=frames.astype(np.float16))
            os.replace(temporary, output)
            response = {"status": "ok", "feature_path": str(output),
                        "feature_sha256": sha256(output),
                        "frame_shape": list(frames.shape), "dtype": "float16",
                        "sample_rate_hz": TARGET_SAMPLE_RATE,
                        "model_seconds": time.monotonic() - started}
        except Exception as exc:
            response = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    provenance = git_provenance(root, args.allow_dirty)
    manifest_hash = sha256(args.manifest)
    shard_hashes = {str(path): sha256(path) for path in audio_shards(args.model_dir)}
    model_hash = sha256(args.model_dir / "model.safetensors.index.json")
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if args.split == "cv" and row["speaker_id"] in EXCLUDED_SPEAKERS:
                continue
            if args.split == "test" and row["speaker_id"] not in EXCLUDED_SPEAKERS:
                continue
            rows.append(row)
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated utterance IDs in manifest")
    if args.split == "cv" and len(rows) != 16366 and args.limit is None:
        raise ValueError(f"Expected 16,366 CV utterances, got {len(rows)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    base = args.run_root / "features" / "A2"
    receipts_dir = base / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base / f"run_meta_{args.split}.json"
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest": str(args.manifest), "manifest_sha256": manifest_hash,
            "model": str(args.model_dir), "model_index_sha256": model_hash,
            "audio_weight_shards_sha256": shard_hashes,
            "encoder": ENCODER, "preprocess_version": PREPROCESS_VERSION,
            "sample_rate_hz": TARGET_SAMPLE_RATE, "inference_dtype": "bfloat16",
            "split": args.split, "count": len(rows), "timeout_s": args.timeout}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "model_index_sha256", "audio_weight_shards_sha256",
                    "encoder", "preprocess_version", "split"):
            if old[key] != meta[key]:
                raise RuntimeError(f"Cannot resume A2 with changed {key}")
        if old["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume A2 from a different Git commit")
        meta["started_at"] = old["started_at"]
        meta["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(meta_path, meta)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    worker = PersistentJSONWorker([args.worker_python, "-u", "-m",
                                   "emotioncore.x02.audio_a2", "--worker",
                                   "--model-dir", str(args.model_dir)],
                                  cwd=root, env=env, stderr_path=base / "worker_errors.log")
    status = {"ok": 0, "failed": 0}
    try:
        for index, row in enumerate(rows, 1):
            receipt_path = receipts_dir / f"{row['utt_id']}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (receipt["input_sha256"] != row["wav"]["sha256"] or
                        receipt["model_index_sha256"] != model_hash or
                        receipt["preprocess_version"] != PREPROCESS_VERSION):
                    raise RuntimeError(f"A2 provenance changed: {row['utt_id']}")
            else:
                result = worker.request({"wav": str(args.media_root / row["wav"]["path"]),
                      "input_sha256": row["wav"]["sha256"],
                      "output": str(base / "data" / f"{row['utt_id']}.npz")}, args.timeout)
                receipt = {"id": row["utt_id"], "speaker_id": row["speaker_id"],
                           "official_split": row["official_split"],
                           "encoder": ENCODER, "model_index_sha256": model_hash,
                           "preprocess_version": PREPROCESS_VERSION,
                           "input_sha256": row["wav"]["sha256"], **result}
                write_json_atomic(receipt_path, receipt)
            status[receipt["status"]] += 1
            if index % 100 == 0 or index == len(rows):
                write_json_atomic(base / f"progress_{args.split}.json",
                                  {"done": index, "total": len(rows), "status": status})
    finally:
        worker.close()
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["status"] = status
    write_json_atomic(meta_path, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--split", choices=("cv", "test"), default="cv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.model_dir)
    else:
        for name in ("manifest", "media_root", "run_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
