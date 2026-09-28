"""BF16 Qwen2.5-Omni zero-shot ET-val probe with isolated media decoding."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification, paired_bootstrap, percentile
from emotioncore.eval.run_baselines import prompt_pair
from emotioncore.models.qwen_omni import QwenOmniScorer, conversation
from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


EXPECTED_VAL_COUNT = 1908


def media_worker() -> None:
    import torch
    from qwen_omni_utils import process_mm_info

    for line in sys.stdin:
        try:
            task = json.loads(line)
            media = Path(task["media"])
            if sha256(media) != task["input_sha256"]:
                raise ValueError("Media differs from frozen ET-val manifest")
            condition = task["condition"]
            audios, images, videos = process_mm_info(
                conversation("system", "user", str(media), condition == "av"),
                use_audio_in_video=condition == "av")
            if images is not None or not audios or \
               bool(videos) != (condition == "av"):
                raise ValueError("Decoded modality does not match probe condition")
            values = {"audio": np.asarray(audios[0], dtype=np.float32)}
            if condition == "av":
                frames = videos[0]
                values["video"] = (frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor)
                                   else np.asarray(frames, dtype=np.float32))
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.npz")
            np.savez(temporary, **values)
            os.replace(temporary, output)
            result = {"status": "ok", "output": str(output), "sha256": sha256(output),
                      "audio_samples": len(values["audio"]),
                      "video_shape": list(values["video"].shape) if "video" in values else None}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(result), flush=True)


class BF16Scorer:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        torch.set_num_threads(4)
        self.device = "cuda"
        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir), local_files_only=True)
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        config.enable_audio_output = False
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(model_dir), config=config, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa", local_files_only=True).eval()
        self.thinker = getattr(self.model, "thinker", self.model)
        self.first_ids = [self.processor.tokenizer(label, add_special_tokens=False).input_ids[0]
                          for label in (*LABELS, "other")]
        if len(set(self.first_ids)) != 8:
            raise RuntimeError("Emotion labels do not have unique first tokens")

    def score(self, system: str, user: str, media: str, condition: str,
              audio: np.ndarray, video: np.ndarray | None) -> dict:
        frames = self.torch.from_numpy(video) if video is not None else None
        result = QwenOmniScorer.score_prepared(self, system, user, media, condition,
                                               audio=audio, frames=frames)
        scores = result["raw_logprobs"]
        result["pred"] = LABELS[int(np.argmax(scores[:7]))]
        result["pred_with_other"] = (*LABELS, "other")[int(np.argmax(scores))]
        return result


def load_val(manifest: Path, limit: int | None) -> list[dict]:
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != EXPECTED_VAL_COUNT or any(row["official_split"] != "val" for row in rows):
        raise ValueError("P1 accepts only the frozen ET-val manifest")
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated ET-val IDs")
    return rows[:limit] if limit else rows


def analyze(rows: list[dict], conditions: tuple[str, ...], output_dir: Path,
            x01_root: Path, *, smoke: bool) -> dict:
    result = {}
    for condition in conditions:
        saved = [json.loads((output_dir / condition / "receipts" /
                            f"{row['utt_id']}.json").read_text(encoding="utf-8")) for row in rows]
        scored = [{"id": row["utt_id"], "truth": row["labels"]["M"],
                   "pred": receipt.get("pred") if receipt["status"] == "ok" else None}
                  for row, receipt in zip(rows, saved, strict=True)]
        latencies = [receipt["latency_s"] for receipt in saved]
        result[condition] = {"M": classification(scored),
                             "latency_p50_s": percentile(latencies, 0.5),
                             "latency_p95_s": percentile(latencies, 0.95),
                             "peak_cuda_allocated_mib": max(
                                 (receipt.get("peak_cuda_allocated_mib", 0) for receipt in saved),
                                 default=0),
                             "failed": sum(receipt["status"] != "ok" for receipt in saved)}
        if not smoke:
            prior_dir = x01_root / "inference" / f"base_et_val_{condition}" / "predictions"
            old = []
            for row in rows:
                receipt = json.loads((prior_dir / f"{row['utt_id']}.json").read_text(encoding="utf-8"))
                old.append({"id": row["utt_id"], "truth": row["labels"]["M"],
                            "pred": receipt.get("pred") if receipt["status"] == "ok" else None})
            result[condition]["x01_nf4_M"] = classification(old)
            result[condition]["bf16_minus_nf4_macro_f1"] = paired_bootstrap(scored, old)
    return result


def run(args: argparse.Namespace) -> None:
    import torch

    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    os.environ.update(VIDEO_MAX_PIXELS="50176", FPS_MAX_FRAMES="8", MAX_PIXELS="401408",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    rows = load_val(args.manifest, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems = {}
    for condition in args.conditions:
        system, user = prompt_pair(args.legacy_root, condition)
        systems[condition] = (system, user)
    model_index = args.model_dir / "model.safetensors.index.json"
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest), "model_index_sha256": sha256(model_index),
            "model_dir": str(args.model_dir), "dtype": "bfloat16", "quantization": None,
            "conditions": args.conditions, "rows": len(rows), "preprocess_timeout_s": args.timeout,
            "video_max_pixels": 50176, "fps_max_frames": 8, "max_pixels": 401408,
            "prompts_sha256": {condition: [hashlib.sha256(text.encode()).hexdigest()
                                           for text in pair] for condition, pair in systems.items()}}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    model = BF16Scorer(args.model_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    worker = PersistentJSONWorker([sys.executable, "-u", "-m", "emotioncore.x02.probe_p1",
                                   "--worker"], cwd=Path.cwd(), env=env,
                                  stderr_path=args.output_dir / "worker_errors.log")
    try:
        for condition in args.conditions:
            system, user = systems[condition]
            receipts = args.output_dir / condition / "receipts"
            receipts.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(rows, 1):
                target = receipts / f"{row['utt_id']}.json"
                if target.is_file():
                    prior = json.loads(target.read_text(encoding="utf-8"))
                    if prior["input_sha256"] != row["wav" if condition == "audio" else "mp4"]["sha256"]:
                        raise ValueError("P1 resume input SHA mismatch")
                    continue
                start = time.monotonic()
                media = args.media_root / row["wav" if condition == "audio" else "mp4"]["path"]
                input_sha = row["wav" if condition == "audio" else "mp4"]["sha256"]
                prepared = args.output_dir / "prepared" / f"{row['utt_id']}_{condition}.npz"
                decoded = worker.request({"media": str(media), "input_sha256": input_sha,
                                          "condition": condition, "output": str(prepared)}, args.timeout)
                receipt = {"id": row["utt_id"], "condition": condition,
                           "input_sha256": input_sha, "status": decoded["status"],
                           "media_decode_s": decoded["elapsed_s"]}
                try:
                    if decoded["status"] != "ok":
                        raise RuntimeError(decoded.get("error", "Media preprocessing failed"))
                    if sha256(prepared) != decoded["sha256"]:
                        raise ValueError("Prepared media SHA mismatch")
                    with np.load(prepared) as data:
                        audio = data["audio"].copy()
                        video = data["video"].copy() if "video" in data else None
                    receipt.update(model.score(system, user, str(media), condition, audio, video))
                except Exception as exc:
                    receipt.update(status="failed", pred=None, pred_with_other=None,
                                   error=f"{type(exc).__name__}: {str(exc)[:500]}")
                finally:
                    prepared.unlink(missing_ok=True)
                receipt["latency_s"] = time.monotonic() - start
                write_json_atomic(target, receipt)
                if index % 100 == 0 or index == len(rows):
                    write_json_atomic(args.output_dir / condition / "progress.json",
                                      {"done": index, "total": len(rows)})
    finally:
        worker.close()
    result = {"probe": "P1", "conditions": analyze(rows, tuple(args.conditions),
                 args.output_dir, args.x01_root, smoke=args.limit is not None),
              "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["elapsed_s"] = result["elapsed_s"]
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--x01-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=("audio", "av"), default=("audio", "av"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.worker:
        media_worker()
    else:
        for name in ("manifest", "media_root", "model_dir", "legacy_root", "x01_root", "output_dir"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
