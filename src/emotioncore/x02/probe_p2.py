"""Qwen3-Omni text-only first-token ET-val probe with isolated media decoding."""
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

from emotioncore.eval.metrics import LABELS, classification, percentile
from emotioncore.eval.run_baselines import prompt_pair
from emotioncore.models.qwen_omni import ASSISTANT_PREFIX, conversation
from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.probe_p1 import EXPECTED_VAL_COUNT
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


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
            if condition not in ("audio", "video", "av"):
                raise ValueError(condition)
            audios, images, videos = process_mm_info(
                conversation("system", "user", str(media), condition != "audio"),
                use_audio_in_video=condition == "av")
            if images is not None or bool(audios) != (condition != "video") or \
               bool(videos) != (condition != "audio"):
                raise ValueError("Decoded modality does not match probe condition")
            values = {}
            if audios:
                values["audio"] = np.asarray(audios[0], dtype=np.float32)
            if videos:
                frames = videos[0]
                values["video"] = (frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor)
                                   else np.asarray(frames, dtype=np.float32))
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.npz")
            np.savez(temporary, **values)
            os.replace(temporary, output)
            result = {"status": "ok", "output": str(output), "sha256": sha256(output)}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(result), flush=True)


class P2Scorer:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import (BitsAndBytesConfig, Qwen3OmniMoeProcessor,
                                  Qwen3OmniMoeThinkerForConditionalGeneration)

        self.torch = torch
        self.device = "cuda:0"
        torch.set_num_threads(4)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            str(model_dir), local_files_only=True)
        self.first_ids = [self.processor.tokenizer(label, add_special_tokens=False).input_ids[0]
                          for label in LABELS]
        if len(set(self.first_ids)) != len(LABELS):
            raise RuntimeError("Seven emotion labels do not have unique first tokens")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            str(model_dir), quantization_config=quant, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa",
            local_files_only=True).eval()

    def score(self, system: str, user: str, media: str, condition: str,
              audio: np.ndarray | None, video: np.ndarray | None) -> dict:
        torch = self.torch
        conv = conversation(system, user, media, condition != "audio")
        prompt = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False) + ASSISTANT_PREFIX
        start = time.monotonic()
        frames = torch.from_numpy(video) if video is not None else None
        inputs = self.processor(text=prompt, audio=[audio] if audio is not None else None,
                                videos=[frames] if frames is not None else None,
                                return_tensors="pt", padding=True,
                                use_audio_in_video=condition == "av", fps=2.0)
        has_audio = inputs.get("input_features") is not None
        has_video = inputs.get("pixel_values_videos") is not None
        if has_audio != (condition != "video") or has_video != (condition != "audio"):
            raise RuntimeError("Processor modality receipt mismatch")
        preprocessing_s = time.monotonic() - start
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        torch.cuda.synchronize()
        forward_start = time.monotonic()
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, return_dict=True,
                                use_audio_in_video=condition == "av")
            logprobs = output.logits[0, -1].float().log_softmax(-1)[self.first_ids]
            probabilities = logprobs.softmax(0)
        torch.cuda.synchronize()
        return {"score_labels": list(LABELS), "raw_logprobs": logprobs.tolist(),
                "scores": probabilities.tolist(),
                "pred": LABELS[int(logprobs.argmax())],
                "preprocess_s": preprocessing_s,
                "forward_s": time.monotonic() - forward_start,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "modality_switch": {"processor": condition == "av",
                                    "forward": condition == "av",
                                    "has_audio": has_audio, "has_video": has_video}}


def load_val(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != EXPECTED_VAL_COUNT or any(row["official_split"] != "val" for row in rows):
        raise ValueError("P2 accepts only the frozen ET-val manifest")
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated ET-val IDs")
    return rows


def analyze(rows: list[dict], conditions: tuple[str, ...], output_dir: Path) -> dict:
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
    return result


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    os.environ.update(VIDEO_MAX_PIXELS="50176", FPS_MAX_FRAMES="8", MAX_PIXELS="401408",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    rows = load_val(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    systems = {condition: prompt_pair(args.legacy_root, condition)
               for condition in args.conditions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_index = args.model_dir / "model.safetensors.index.json"
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_sha256": sha256(args.manifest), "model_index_sha256": sha256(model_index),
            "model_dir": str(args.model_dir), "dtype": "bfloat16",
            "quantization": "bitsandbytes_nf4_double_quant_thinker_only",
            "first_token_ids": None, "conditions": args.conditions, "rows": len(rows),
            "preprocess_timeout_s": args.timeout, "video_max_pixels": 50176,
            "fps_max_frames": 8, "max_pixels": 401408,
            "prompts_sha256": {condition: [hashlib.sha256(text.encode()).hexdigest()
                                           for text in pair] for condition, pair in systems.items()}}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    model = P2Scorer(args.model_dir)
    meta["first_token_ids"] = model.first_ids
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    worker = PersistentJSONWorker([sys.executable, "-u", "-m", "emotioncore.x02.probe_p2",
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
                    expected = row["wav" if condition == "audio" else "mp4"]["sha256"]
                    if prior["input_sha256"] != expected:
                        raise ValueError("P2 resume input SHA mismatch")
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
                        audio = data["audio"].copy() if "audio" in data else None
                        video = data["video"].copy() if "video" in data else None
                    receipt.update(model.score(system, user, str(media), condition, audio, video))
                except Exception as exc:
                    receipt.update(status="failed", pred=None,
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
    result = {"probe": "P2", "conditions": analyze(rows, tuple(args.conditions), args.output_dir),
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=("audio", "video", "av"),
                        default=("audio", "video", "av"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.worker:
        media_worker()
    else:
        for name in ("manifest", "media_root", "model_dir", "legacy_root", "output_dir"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
