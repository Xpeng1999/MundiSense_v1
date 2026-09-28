from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from emotioncore.models.qwen_omni import QwenOmniScorer
from emotioncore.utils.common import load_paths, write_json_atomic

LEGACY_USER = "判断当前素材中唯一目标人物在所提供信息下可观察到的情绪，严格按给定字段输出 JSON。"


def run(config: Path, output: Path, per_condition: int = 20) -> dict:
    import numpy as np
    import soundfile as sf
    import torch

    paths = load_paths(config)
    legacy = paths["legacy_root"]
    source = legacy / "av_attribution_v16/artifacts/chinese"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    predictions = source / "predictions/qwen_old"
    selected = []
    for condition in ("audio", "av"):
        candidates = []
        for row in manifest:
            old_file = predictions / f"{row['id']}__{condition}.json"
            if old_file.is_file():
                old = json.loads(old_file.read_text(encoding="utf-8"))
                if old.get("status") == "ok" and old.get("raw_logprobs"):
                    candidates.append((row, old))
        selected.extend((condition, row, old) for row, old in sorted(candidates, key=lambda x: x[0]["id"])[:per_condition])
    if len(selected) != 2 * per_condition:
        raise ValueError(f"Only {len(selected)} eligible D3 records")
    train = legacy / "data/manifests_t0/swift/train_e3_core_av.jsonl"
    system = json.loads(train.open(encoding="utf-8").readline())["messages"][0]["content"]
    model = QwenOmniScorer(paths["model_root"], paths["adapter_root"])
    details = []
    for condition, row, old in selected:
        audio = sf.read(row["audio"], dtype="float32")[0]
        frames = torch.from_numpy(np.load(row["frames"])).permute(0, 3, 1, 2) if condition == "av" else None
        fps = 3 / (row["frames_receipt"][-1]["time_s"] - row["frames_receipt"][0]["time_s"]) if frames is not None else 2.0
        new = model.score_prepared(system, LEGACY_USER,
                                   row["audio"] if condition == "audio" else row["source"],
                                   condition, audio=audio, frames=frames, fps=fps)
        difference = max(abs(a - b) for a, b in zip(new["raw_logprobs"], old["raw_logprobs"]))
        details.append({"id": row["id"], "condition": condition, "old_pred": old["pred"],
                        "new_pred": new["pred"], "max_abs_logprob_difference": difference,
                        "passed": new["pred"] == old["pred"] and difference <= 1e-3})
        write_json_atomic(output, {"passed": all(x["passed"] for x in details),
                                   "completed": len(details), "required": 2 * per_condition,
                                   "details": details})
    return {"passed": all(x["passed"] for x in details), "completed": len(details),
            "max_abs_logprob_difference": max(x["max_abs_logprob_difference"] for x in details),
            "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-condition", type=int, default=20)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    result = run(args.config, args.output, args.per_condition)
    if not result["passed"]:
        raise SystemExit("D3 failed: do not start D2")


if __name__ == "__main__":
    main()
