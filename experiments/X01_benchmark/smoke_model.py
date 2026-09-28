"""Small no-gate Qwen media path check; D2 still requires D3."""
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VIDEO_MAX_PIXELS="50176",
                  FPS_MAX_FRAMES="8", MAX_PIXELS="401408", TOKENIZERS_PARALLELISM="false")

from emotioncore.eval.run_baselines import prompt_pair
from emotioncore.models.qwen_omni import QwenOmniScorer
from emotioncore.utils.common import load_paths, write_json_atomic

paths = load_paths(Path("configs/paths.server.yaml"))
row = json.loads(Path("experiments/X01_benchmark/results/et_test.jsonl").open().readline())
model = QwenOmniScorer(paths["model_root"])
results = {}
for condition in ("audio", "video", "av"):
    system, user = prompt_pair(paths["legacy_root"], condition)
    kind = "wav" if condition == "audio" else "mp4"
    media = paths["processed_root"] / "emotiontalk" / row[kind]["path"]
    result = model.score_file(system, user, media, condition)
    assert len(result["raw_logprobs"]) == 8
    assert result["modality_switch"]["has_audio"] == (condition != "video")
    assert result["modality_switch"]["has_video"] == (condition != "audio")
    results[condition] = {key: result[key] for key in ("pred", "pred_with_other", "modality_switch")}
write_json_atomic(Path("experiments/X01_benchmark/results/media_smoke.json"), results)
