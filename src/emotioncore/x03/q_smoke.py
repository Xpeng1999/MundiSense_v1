"""Q arm modality and NF4 LoRA step-time feasibility probe."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.asr_cer import sha256
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


MODEL = Path("/home/administrator/projects/emotion/omni_models/Qwen2.5-Omni-7B")
MEDIA_ROOT = Path("/home/administrator/emotion-data/emotionAI_processed/phase1_v1/emotiontalk")
PROMPT = "请根据提供的声音和画面判断可观察到的情绪，只输出指定 JSON。"
PREFIX = '{"coarse_emotion":"'


def assert_modality_receipt(condition: str, keys: set[str]) -> None:
    if condition not in ("A", "V", "AV"):
        raise ValueError(condition)
    has_audio = "input_features" in keys
    has_video = "pixel_values_videos" in keys
    if has_audio != (condition in ("A", "AV")) or has_video != (condition in ("V", "AV")):
        raise RuntimeError(f"Processor/forward modality mismatch: {condition}, {sorted(keys)}")


def worker() -> None:
    import cv2
    import numpy as np
    import scipy.signal
    import soundfile as sf
    from transformers import Qwen2_5OmniProcessor

    cv2.setNumThreads(1)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    first_ids = [processor.tokenizer(label, add_special_tokens=False).input_ids[0]
                 for label in LABELS]
    if len(set(first_ids)) != 7:
        raise RuntimeError("Seven emotion first tokens are not unique")
    for line in sys.stdin:
        try:
            task = json.loads(line)
            condition = task["condition"]
            content = []
            audio = None
            frames = None
            fps = None
            if condition in ("V", "AV"):
                video = Path(task["video"])
                if sha256(video) != task["video_sha256"]:
                    raise ValueError("Video SHA differs from X03 receipt")
                capture = cv2.VideoCapture(str(video))
                try:
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    frames = []
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                finally:
                    capture.release()
                if len(frames) != 8 or any(frame.shape != (336, 336, 3) for frame in frames):
                    raise ValueError("Invalid eight-frame face video")
                content.append({"type": "video", "video": str(video)})
            if condition in ("A", "AV"):
                wav = Path(task["wav"])
                if sha256(wav) != task["wav_sha256"]:
                    raise ValueError("WAV SHA differs from frozen ET manifest")
                signal, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
                audio = signal.mean(axis=1)
                if sample_rate != 16000:
                    divisor = math.gcd(sample_rate, 16000)
                    audio = scipy.signal.resample_poly(audio, 16000 // divisor,
                                                       sample_rate // divisor)
                audio = np.asarray(audio, dtype=np.float32)
                content.append({"type": "audio", "audio": str(wav)})
            content.append({"type": "text", "text": PROMPT})
            conversation = [{"role": "system", "content": [{"type": "text", "text":
                             "你是外显情绪识别模型，只从可听和可见信息判断。"}]},
                            {"role": "user", "content": content}]
            text = processor.apply_chat_template(conversation, tokenize=False,
                                                 add_generation_prompt=True) + PREFIX
            kwargs = {"text": text, "audio": [audio] if audio is not None else None,
                      "videos": [frames] if frames is not None else None,
                      "return_tensors": "pt", "padding": True,
                      "use_audio_in_video": False}
            if fps is not None:
                kwargs["fps"] = fps
            values = processor(**kwargs)
            actual = {key for key, value in values.items() if value is not None}
            assert_modality_receipt(condition, actual)
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.{os.getpid()}.npz")
            np.savez(temporary, **{key: value.detach().cpu().numpy()
                                   for key, value in values.items() if value is not None})
            os.replace(temporary, output)
            result = {"status": "ok", "path": str(output), "sha256": sha256(output),
                      "condition": condition, "processor_has_audio": "input_features" in actual,
                      "processor_has_video": "pixel_values_videos" in actual,
                      "shapes": {key: list(value.shape) for key, value in values.items()
                                 if value is not None}}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)


def model_and_optimizer():
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, BitsAndBytesConfig, Qwen2_5OmniForConditionalGeneration

    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    config.enable_audio_output = False
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(MODEL), config=config, quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", local_files_only=True)
    thinker = prepare_model_for_kbit_training(model.thinker, use_gradient_checkpointing=True)
    targets = []
    visual = []
    for name, module in thinker.named_modules():
        if not isinstance(module, torch.nn.Module):
            continue
        if name.startswith("model.layers.") and name.endswith((".q_proj", ".k_proj", ".v_proj", ".o_proj")):
            targets.append(name)
        elif name.startswith("visual.blocks.") and name.endswith((".attn.q", ".attn.k",
                                                                  ".attn.v", ".attn.proj")):
            targets.append(name)
            visual.append(name)
    if not targets or not visual or any("audio_tower" in name for name in targets):
        examples = [name for name, _ in thinker.named_modules()
                    if "visual" in name and "attn" in name][:12]
        raise RuntimeError(f"LLM/vision LoRA target discovery failed: "
                           f"targets={len(targets)} visual={len(visual)} examples={examples}")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=targets,
                      rank_pattern={name: 8 for name in visual},
                      alpha_pattern={name: 16 for name in visual})
    thinker = get_peft_model(thinker, lora)
    named_trainable = [(name, p) for name, p in thinker.named_parameters() if p.requires_grad]
    vision = [(name, p) for name, p in named_trainable if "visual.blocks." in name]
    llm = [(name, p) for name, p in named_trainable if "visual.blocks." not in name]
    if not vision or not llm or any("audio_tower" in name for name, _ in named_trainable):
        raise RuntimeError("Trainable parameter groups do not match Q protocol")
    ranks = {"llm": set(), "vision": set()}
    for name, module in thinker.named_modules():
        if name.endswith("lora_A.default") and hasattr(module, "weight"):
            ranks["vision" if "visual.blocks." in name else "llm"].add(module.weight.shape[0])
    if ranks != {"llm": {16}, "vision": {8}}:
        raise RuntimeError(f"LoRA rank mismatch: {ranks}")
    optimizer = torch.optim.AdamW([{"params": [p for _, p in llm], "lr": 5e-5},
                                   {"params": [p for _, p in vision], "lr": 1e-5}],
                                  weight_decay=0.01)
    return thinker, optimizer, {"target_modules": len(targets),
                                "visual_target_modules": len(visual),
                                "llm_trainable_parameters": sum(p.numel() for _, p in llm),
                                "vision_trainable_parameters": sum(p.numel() for _, p in vision)}


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from transformers import Qwen2_5OmniProcessor

    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": sys.argv, "model": str(MODEL), "model_index_sha256":
            sha256(MODEL / "model.safetensors.index.json")}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8")]
    if len(rows) != 19_250:
        raise ValueError("Frozen ET manifest count differs")
    row = next(r for r in rows if r["utt_id"] == args.utt_id)
    video_receipt = json.loads((args.media_root / "receipts" /
                                f"{args.utt_id}.json").read_text(encoding="utf-8"))
    if video_receipt["status"] != "ok":
        raise RuntimeError("Selected smoke video failed fixed face filter")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = PersistentJSONWorker([sys.executable, "-u", "-m", "emotioncore.x03.q_smoke", "--worker"],
                                 cwd=Path.cwd(), env=env, stderr_path=args.output_dir / "worker_errors.log")
    receipts = {}
    try:
        for condition in ("A", "V", "AV"):
            task = {"condition": condition, "wav": str(MEDIA_ROOT / row["wav"]["path"]),
                    "wav_sha256": row["wav"]["sha256"], "video": video_receipt["output"],
                    "video_sha256": video_receipt["output_sha256"],
                    "output": str(args.output_dir / f"{condition}.npz")}
            receipt = child.request(task, args.timeout)
            if receipt["status"] != "ok":
                raise RuntimeError(f"{condition} media preparation failed: {receipt}")
            receipts[condition] = receipt
    finally:
        child.close()
    write_json_atomic(args.output_dir / "modality_receipts.json", receipts)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    first_ids = [processor.tokenizer(label, add_special_tokens=False).input_ids[0]
                 for label in LABELS]
    if len(set(first_ids)) != 7:
        raise RuntimeError("Seven class first token collision")
    torch.set_num_threads(4)
    thinker, optimizer, parameter_info = model_and_optimizer()
    thinker.train()
    results = {}
    prepared = {}
    for condition in ("A", "V", "AV"):
        arrays = np.load(args.output_dir / f"{condition}.npz")
        assert_modality_receipt(condition, set(arrays.files))
        values = {key: torch.from_numpy(arrays[key]).to("cuda:0") for key in arrays.files}
        ids = values["input_ids"]
        suffix = row["labels"]["M"] + '"}'
        answer = processor.tokenizer(suffix, add_special_tokens=False,
                                     return_tensors="pt").input_ids.to("cuda:0")
        values["input_ids"] = torch.cat((ids, answer), dim=1)
        values["attention_mask"] = torch.ones_like(values["input_ids"])
        values["labels"] = torch.cat((torch.full_like(ids, -100), answer), dim=1)
        prepared[condition] = {key: value for key, value in values.items() if key != "labels"}
        times = []
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            output = thinker(**values, use_cache=False, use_audio_in_video=False)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite Q smoke loss: {condition}")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - started)
            losses.append(float(loss.detach().cpu()))
        results[condition] = {"step_s": statistics.median(times[1:]),
                              "step_repeats_s": times,
                              "loss": float(loss.detach().cpu()),
                              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                              "forward_has_audio": "input_features" in values,
                              "forward_has_video": "pixel_values_videos" in values}
        assert_modality_receipt(condition, set(values))
        write_json_atomic(args.output_dir / "steps_partial.json", results)
    thinker.eval()
    with torch.inference_mode():
        for condition, values in prepared.items():
            times = []
            for _ in range(4):
                torch.cuda.synchronize()
                started = time.monotonic()
                thinker(**values, use_cache=False, use_audio_in_video=False)
                torch.cuda.synchronize()
                times.append(time.monotonic() - started)
            results[condition]["inference_s"] = statistics.median(times[1:])
    weighted_step = .2 * results["A"]["step_s"] + .2 * results["V"]["step_s"] + .6 * results["AV"]["step_s"]
    folds = json.loads(Path("configs/folds_v1.json").read_text(encoding="utf-8"))
    train_examples_per_epoch = (sum(f["inner_train_count"] for f in folds["folds"]) +
                                folds["cv_pool_count"] - folds["folds"][0]["inner_validation_count"])
    validation_examples = 8 * (sum(f["inner_validation_count"] for f in folds["folds"]) +
                               folds["folds"][0]["inner_validation_count"])
    projected_training_h = 2 * train_examples_per_epoch * weighted_step / 3600
    projected_validation_h = validation_examples * results["AV"]["inference_s"] / 3600
    projected_prediction_h = (16_366 + 1_929 + 647) * sum(
        results[c]["inference_s"] for c in ("A", "V", "AV")) / 3600
    projected_total_h = projected_training_h + projected_validation_h + projected_prediction_h
    summary = {"steps": results, "weighted_step_s": weighted_step,
               "train_examples_per_epoch": train_examples_per_epoch,
               "validation_examples": validation_examples,
               "projected_training_h": projected_training_h,
               "projected_validation_h": projected_validation_h,
               "projected_prediction_h": projected_prediction_h,
               "projected_total_h_without_loading_or_preprocessing": projected_total_h,
               "gpu_45h_gate": "stop" if projected_total_h > 45 else "within_budget_estimate",
               "parameter_info": parameter_info, "first_token_ids": first_ids}
    write_json_atomic(args.output_dir / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--utt-id", default="G00001_02_01_002")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        if not all((args.manifest, args.media_root, args.output_dir)):
            parser.error("manifest, media-root, and output-dir are required")
        run(args)


if __name__ == "__main__":
    main()
