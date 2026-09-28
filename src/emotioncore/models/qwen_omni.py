from __future__ import annotations

import time
from pathlib import Path

LABELS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise", "other")
ASSISTANT_PREFIX = '{"coarse_emotion":"'


def conversation(system: str, user: str, media: str, video: bool) -> list[dict]:
    element = {"type": "video", "video": media} if video else {"type": "audio", "audio": media}
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [element, {"type": "text", "text":
             user.replace("<video>", "").replace("<audio>", "")}]}]


class QwenOmniScorer:
    def __init__(self, model_dir: Path, adapter_dir: Path | None = None):
        import torch
        from transformers import (AutoConfig, BitsAndBytesConfig,
                                  Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor)

        self.torch = torch
        torch.set_num_threads(4)
        self.device = "cuda"
        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(model_dir))
        config = AutoConfig.from_pretrained(str(model_dir))
        config.enable_audio_output = False
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            str(model_dir), config=config, quantization_config=quant,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
        self.model.eval()
        self.adapter_info = None
        if adapter_dir:
            from peft import PeftModel
            from safetensors import safe_open
            weight = adapter_dir / "adapter_model.safetensors"
            with safe_open(str(weight), framework="pt") as source:
                expected = sum("lora_A" in key for key in source.keys())
            peft_model = PeftModel.from_pretrained(self.model, str(adapter_dir), is_trainable=False)
            peft_model.eval()
            actual = sum(name.endswith("lora_A.default") for name, _ in peft_model.named_modules())
            nonzero = sum(name.endswith("lora_B.default") and hasattr(module, "weight")
                          and bool(module.weight.detach().abs().sum().item() > 0)
                          for name, module in peft_model.named_modules())
            if actual != expected or nonzero == 0:
                raise RuntimeError(f"LoRA mismatch: expected {expected}, injected {actual}, nonzero B {nonzero}")
            self.adapter_info = {"expected_lora_A": expected, "injected_lora_A": actual,
                                 "nonzero_lora_B": nonzero}
        self.thinker = getattr(self.model, "thinker", self.model)
        tokenizer = self.processor.tokenizer
        self.first_ids = [tokenizer(label, add_special_tokens=False).input_ids[0] for label in LABELS]
        if len(set(self.first_ids)) != len(LABELS):
            raise RuntimeError("Emotion labels do not have unique first tokens")

    def score_prepared(self, system: str, user: str, media: str, condition: str,
                       audio=None, frames=None, fps: float = 2.0) -> dict:
        """Score already decoded inputs; this path reproduces the v16 D3 protocol."""
        torch = self.torch
        if condition not in ("audio", "video", "av"):
            raise ValueError(condition)
        use_audio_in_video = condition == "av"
        conv = conversation(system, user, media, condition != "audio")
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True,
                                                  tokenize=False) + ASSISTANT_PREFIX
        start = time.monotonic()
        inputs = self.processor(text=text, audio=[audio] if audio is not None else None,
                                videos=[frames] if frames is not None else None,
                                return_tensors="pt", padding=True,
                                use_audio_in_video=use_audio_in_video, fps=fps)
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
            out = self.thinker(**inputs, use_cache=False, return_dict=True,
                               use_audio_in_video=use_audio_in_video)
            scores = out.logits[0, -1].float().log_softmax(-1)[self.first_ids]
            probabilities = scores.softmax(0)
        torch.cuda.synchronize()
        return {"score_labels": list(LABELS), "raw_logprobs": scores.tolist(),
                "scores": probabilities.tolist(), "pred": LABELS[int(scores.argmax())],
                "preprocess_s": preprocessing_s, "forward_s": time.monotonic() - forward_start,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "modality_switch": {"processor": use_audio_in_video,
                                    "forward": use_audio_in_video,
                                    "has_audio": has_audio, "has_video": has_video}}

    def score_file(self, system: str, user: str, media: Path, condition: str) -> dict:
        """Decode the configured media once and score the first label token."""
        from qwen_omni_utils import process_mm_info

        if condition not in ("audio", "video", "av"):
            raise ValueError(condition)
        use_audio_in_video = condition == "av"
        conv = conversation(system, user, str(media), condition != "audio")
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True,
                                                  tokenize=False) + ASSISTANT_PREFIX
        start = time.monotonic()
        audios, images, videos = process_mm_info(conv, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True,
                                use_audio_in_video=use_audio_in_video)
        has_audio = inputs.get("input_features") is not None
        has_video = inputs.get("pixel_values_videos") is not None
        if has_audio != (condition != "video") or has_video != (condition != "audio"):
            raise RuntimeError("Processor modality receipt mismatch")
        preprocessing_s = time.monotonic() - start
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value
                  for key, value in inputs.items()}
        self.torch.cuda.synchronize()
        forward_start = time.monotonic()
        with self.torch.inference_mode():
            output = self.thinker(**inputs, use_cache=False, return_dict=True,
                                  use_audio_in_video=use_audio_in_video)
            scores = output.logits[0, -1].float().log_softmax(-1)[self.first_ids]
            probabilities = scores.softmax(0)
        self.torch.cuda.synchronize()
        return {"score_labels": list(LABELS), "raw_logprobs": scores.tolist(),
                "scores": probabilities.tolist(),
                "pred": LABELS[int(scores[:7].argmax())],
                "pred_with_other": LABELS[int(scores.argmax())],
                "preprocess_s": preprocessing_s,
                "forward_s": time.monotonic() - forward_start,
                "peak_cuda_allocated_mib": self.torch.cuda.max_memory_allocated() / 2**20,
                "modality_switch": {"processor": use_audio_in_video,
                                    "forward": use_audio_in_video,
                                    "has_audio": has_audio, "has_video": has_video}}
