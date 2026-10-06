"""Measure a steady-state, serial E2 pipeline from frozen ET-test receipts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.e2 import E2Model, batch, condition_keep
from emotioncore.x03.e2_target import load_target, target_rows


STAGES = ("face", "A3", "V2", "SenseVoice", "T1_base", "E2_head")


def receipt_seconds(path: Path, uid: str) -> float:
    receipt = json.loads((path / f"{uid}.json").read_text(encoding="utf-8"))
    if receipt["id"] != uid:
        raise ValueError(f"Timing receipt ID mismatch for {uid}")
    elapsed = float(receipt["elapsed_s"])
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"Invalid timing receipt for {uid}")
    return elapsed


def summarize(timings: list[dict]) -> dict:
    summary = {}
    for key in (*STAGES, "serial_total"):
        values = np.asarray([row[key] for row in timings], dtype=float)
        summary[key] = {"n": len(values), "p50_s": float(np.percentile(values, 50)),
                        "p95_s": float(np.percentile(values, 95)),
                        "mean_s": float(np.mean(values))}
    return summary


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel

    started = time.monotonic()
    git = git_provenance(Path.cwd(), args.allow_dirty)
    rows = target_rows(args.manifest, "et_test")
    data = load_target(SimpleNamespace(target_pack=args.target_pack,
                                       target_tokens=args.t1_tokens,
                                       target_cls=args.t1_cls,
                                       target_cls_meta=args.t1_cls_meta,
                                       face_policy=args.face_policy,
                                       target_manifest=args.manifest,
                                       split="et_test"), rows)
    et_meta = json.loads(args.et_meta.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != et_meta["checkpoint_sha256"]:
        raise ValueError("Final E2 checkpoint differs from ET-test inference receipt")
    cls_meta = json.loads(args.t1_cls_meta.read_text(encoding="utf-8"))
    if sha256(args.t1_model / "pytorch_model.bin") != cls_meta["model_sha256"]:
        raise ValueError("T1 base model differs from frozen [CLS] cache")
    with np.load(args.t1_tokens) as archive:
        ids = archive["input_ids"].copy()
        masks = archive["attention_mask"].copy()
        failed = archive["failed"].copy()
        token_utt_ids = [str(uid) for uid in archive["utt_ids"]]
    if token_utt_ids != [row["utt_id"] for row in rows]:
        raise ValueError("ET-test token order differs from manifest")
    torch.set_num_threads(args.threads)
    t1 = AutoModel.from_pretrained(str(args.t1_model), local_files_only=True).to("cpu").eval()
    head = E2Model.build(et_meta["final_config"]["dropout"]).to("cpu").eval()
    head.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))

    def infer(index: int) -> tuple[float, float]:
        if failed[index]:
            cls = torch.zeros((1, 1024), dtype=torch.float32)
            t1_seconds = 0.0
        else:
            length = int(masks[index].sum())
            input_ids = torch.from_numpy(ids[index:index + 1, :length].astype(np.int64))
            attention_mask = torch.from_numpy(masks[index:index + 1, :length].astype(np.int64))
            start = time.monotonic()
            cls = t1(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            t1_seconds = time.monotonic() - start
        start = time.monotonic()
        inputs = batch(data, np.asarray([index]), "cpu")
        keep = condition_keep(inputs[-1], "AVT")
        if not keep.any():
            return t1_seconds, time.monotonic() - start
        probability = torch.softmax(head(*inputs[:4], cls.float(), keep), dim=1)
        head_seconds = time.monotonic() - start
        if not torch.isfinite(probability).all():
            raise ValueError(f"Nonfinite E2 probability for {rows[index]['utt_id']}")
        return t1_seconds, head_seconds

    with torch.inference_mode():
        for index in range(min(args.warmup, len(rows))):
            infer(index)
        timings = []
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "et_test_latency.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                uid = row["utt_id"]
                stages = {"face": receipt_seconds(args.face_receipts, uid),
                          "A3": receipt_seconds(args.a3_receipts, uid),
                          "V2": receipt_seconds(args.v2_receipts, uid),
                          "SenseVoice": receipt_seconds(args.asr_receipts, uid)}
                stages["T1_base"], stages["E2_head"] = infer(index)
                stages["serial_total"] = sum(stages[name] for name in STAGES)
                item = {"utt_id": uid, **stages}
                timings.append(item)
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"count": len(timings), "condition": "AVT",
               "method": "sum of frozen per-utterance stage elapsed_s and CPU single-sample T1/E2 forward times",
               "model_loading_excluded": True, "warmup_count": min(args.warmup, len(rows)),
               "stage_devices": {"face": "CPU/YuNet", "A3": "X02 CUDA feature run",
                                 "V2": "X02 CUDA feature run",
                                 "SenseVoice": "X02 CUDA ASR run",
                                 "T1_base": "CPU", "E2_head": "CPU"},
               "stage_statistics": summarize(timings),
               "sample_timings": {"path": str(output), "sha256": sha256(output)}}
    write_json_atomic(args.output_dir / "summary.json", summary)
    write_json_atomic(args.output_dir / "run_meta.json",
                      {"git": git, "command": sys.argv,
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "elapsed_s": time.monotonic() - started,
                       "et_meta_sha256": sha256(args.et_meta),
                       "target_pack_meta_sha256": sha256(args.target_pack / "target_pack_meta.json"),
                       "t1_cls_meta_sha256": sha256(args.t1_cls_meta),
                       "summary_sha256": sha256(args.output_dir / "summary.json")})


def main() -> None:
    parser = argparse.ArgumentParser()
    for flag in ("manifest", "face-policy", "target-pack", "t1-tokens",
                 "t1-cls", "t1-cls-meta", "t1-model", "checkpoint", "et-meta",
                 "face-receipts", "a3-receipts", "v2-receipts", "asr-receipts",
                 "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.warmup < 0:
        parser.error("threads must be positive and warmup nonnegative")
    run(args)


if __name__ == "__main__":
    main()
