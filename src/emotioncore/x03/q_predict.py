"""Score frozen Q fold/target A, V and AV manifests with seven first tokens."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification, sims_metrics
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x03.q_smoke import MODEL, assert_modality_receipt, model_and_optimizer
from emotioncore.x03.q_train import load_input


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import Qwen2_5OmniProcessor

    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != "expandable_segments:True":
        raise RuntimeError("Q prediction requires expandable CUDA segments")
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(Path("configs/x03_face_missing_policy.json")) != FACE_SHA:
        raise ValueError("Frozen face policy changed")
    manifest_meta = json.loads(args.manifest_meta.read_text(encoding="utf-8"))
    if manifest_meta.get("face_policy_sha256", FACE_SHA) != FACE_SHA or manifest_meta["git"]["dirty"]:
        raise ValueError("Inference manifest provenance mismatch")
    if args.run_name.startswith("fold"):
        run_id = args.run_name
        split = "oof"
    elif args.run_name == "final_et_test":
        run_id = args.run_name
        split = "et_test"
    elif args.run_name == "final_sims_valid":
        run_id = args.run_name
        split = "sims_valid"
    else:
        raise ValueError("Unsupported Q prediction run")
    manifests = {}
    for form in ("A", "V", "AV"):
        name = f"{run_id}_{form}"
        spec = manifest_meta["outputs"][name]
        path = Path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Inference manifest SHA changed: {name}")
        rows = read_jsonl(path)
        if len(rows) != spec["n"]:
            raise ValueError(f"Inference manifest count changed: {name}")
        manifests[form] = rows
    ids = [r["utt_id"] for r in manifests["A"]]
    if len(ids) != len(set(ids)) or any([r["utt_id"] for r in manifests[f]] != ids for f in ("V", "AV")):
        raise ValueError("A/V/AV ID ordering differs")
    checkpoint = args.adapter / "adapter_model.safetensors"
    if sha256(checkpoint) != args.adapter_sha256:
        raise ValueError("Best Q checkpoint SHA changed")
    for form, rows in manifests.items():
        for row in rows:
            if row["effective_form"] == "uniform":
                allowed = (form == "V" and row["face_missing"]) or \
                          (split == "sims_valid" and row.get("audio_missing"))
                if not allowed:
                    raise ValueError("Uniform score outside frozen missing-face V rule")
            elif not (args.cache_root / row["utt_id"] / f"{row['effective_form']}.json").is_file():
                raise FileNotFoundError(f"Uncached Q input: {row['utt_id']}/{form}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_name": args.run_name, "split": split,
            "manifest_meta_sha256": sha256(args.manifest_meta), "adapter": str(args.adapter),
            "adapter_sha256": args.adapter_sha256,
            "base_index_sha256": sha256(MODEL / "model.safetensors.index.json"),
            "face_policy_sha256": FACE_SHA,
            "cuda_allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"]}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    torch.set_num_threads(4)
    processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
    first_ids = [processor.tokenizer(label, add_special_tokens=False).input_ids[0] for label in LABELS]
    if len(set(first_ids)) != 7:
        raise RuntimeError("Seven label first-token IDs collide")
    thinker, optimizer, _ = model_and_optimizer()
    del optimizer
    thinker.load_adapter(str(args.adapter), adapter_name="best")
    thinker.set_adapter("best")
    thinker.eval()
    torch.cuda.reset_peak_memory_stats()
    summaries = {}
    with torch.inference_mode():
        for form in ("A", "V", "AV"):
            output = args.output_dir / f"{form}.jsonl"
            scored = []
            sims_scored = []
            durations = []
            uniform = 0
            fallback = 0
            with output.open("w", encoding="utf-8") as stream:
                for index, row in enumerate(manifests[form], 1):
                    started = time.monotonic()
                    if row["effective_form"] == "uniform":
                        probabilities = [1 / 7] * 7
                        pred = None  # Count the row as incorrect, even if its true label is neutral.
                        has_audio = has_video = False
                        status = "missing_media" if row.get("audio_missing") else "face_missing"
                        uniform += 1
                    else:
                        values = load_input(args.cache_root, row)
                        assert_modality_receipt(row["effective_form"], set(values))
                        has_audio = "input_features" in values
                        has_video = "pixel_values_videos" in values
                        result = thinker(**values, use_cache=False, use_audio_in_video=False)
                        logits = result.logits[0, -1].float()[first_ids]
                        probabilities = logits.softmax(0).cpu().tolist()
                        pred = LABELS[int(np.argmax(probabilities))]
                        status = "face_missing_fallback" if row["face_missing_fallback"] else "ok"
                        fallback += int(row["face_missing_fallback"])
                        del values, result
                    elapsed = time.monotonic() - started
                    durations.append(elapsed)
                    item = {"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                            "fold": row["fold"], "form": form,
                            "effective_form": row["effective_form"], "face_missing": row["face_missing"],
                            "face_missing_fallback": row["face_missing_fallback"], "status": status,
                            "probabilities": probabilities, "pred": pred,
                            "forward_has_audio": has_audio, "forward_has_video": has_video,
                            "latency_s": elapsed}
                    if split == "sims_valid":
                        item["label"] = row["label"]
                        item["s"] = None if row["effective_form"] == "uniform" else float(
                            probabilities[1] - sum(probabilities[2:6]))
                        sims_scored.append({"id": row["utt_id"], "label": row["label"],
                                            "s": item["s"]})
                    else:
                        item["truth_M"] = row["truth_M"]
                        scored.append({"id": row["utt_id"], "speaker_id": row["speaker_id"],
                                       "truth": row["truth_M"], "pred": pred})
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    if index % 100 == 0:
                        write_json_atomic(args.output_dir / "progress.json",
                                          {"form": form, "done": index, "total": len(ids)})
            summaries[form] = {"n": len(ids), "uniform": uniform, "face_missing_fallback": fallback,
                               "metrics": sims_metrics(sims_scored) if split == "sims_valid" else classification(scored),
                               "latency_p50_s": float(np.percentile(durations, 50)),
                               "latency_p95_s": float(np.percentile(durations, 95)),
                               "path": str(output), "sha256": sha256(output)}
            write_json_atomic(args.output_dir / "summary_partial.json", summaries)
    summaries["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    write_json_atomic(args.output_dir / "summary.json", summaries)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest-meta", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
