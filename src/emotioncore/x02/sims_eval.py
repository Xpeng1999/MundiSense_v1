"""Apply frozen ET-trained heads to the CH-SIMS valid-only adapter."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, sims_metrics
from emotioncore.x02 import head_h1, head_h2
from emotioncore.x02.final_eval import load_test_frames
from emotioncore.x02.frames_pack import SPECS
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.sims_prepare import validate_adapter
from emotioncore.x02.text_t1 import make_model, predict as predict_t1


def load_rows(adapter: Path, valid_manifest: Path) -> tuple[list[dict], list[dict]]:
    with adapter.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    validate_adapter(rows)
    with valid_manifest.open(encoding="utf-8") as stream:
        original = [json.loads(line) for line in stream]
    if (len(original) != len(rows) or any(row["mode"] != "valid" for row in original) or
            [row["utt_id"] for row in rows] !=
            [f"{row['video_id']}__{row['clip_id']}" for row in original]):
        raise ValueError("SIMS adapter differs from frozen valid manifest")
    return rows, original


def load_checkpoint(result_path: Path, encoder: str, head: str,
                    folds_path: Path):
    import torch

    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    if receipt["encoder"] != encoder or receipt["head"] != head:
        raise ValueError("Final checkpoint result names the wrong head")
    path = Path(receipt["checkpoint_path"])
    if head_h1.sha256(path) != receipt["checkpoint_sha256"]:
        raise ValueError("Final checkpoint SHA differs from receipt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint["encoder"] != encoder or checkpoint["head"] != head or
            checkpoint["folds_sha256"] != head_h1.sha256(folds_path)):
        raise ValueError("Final checkpoint differs from frozen folds")
    return checkpoint, path


def predict_head(args: argparse.Namespace, rows: list[dict]):
    import torch

    checkpoint, path = load_checkpoint(args.final_result, args.encoder,
                                       args.head, args.folds)
    if args.head == "h1":
        x, available = head_h1.load_features(rows, args.receipts, args.encoder)
        if x.ndim == 3:
            class LayerWeightedLogistic(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer_logits = torch.nn.Parameter(torch.zeros(x.shape[1]))
                    self.linear = torch.nn.Linear(x.shape[2], len(LABELS))

                def forward(self, values):
                    weights = torch.softmax(self.layer_logits, dim=0)
                    return self.linear((values * weights.view(1, -1, 1)).sum(dim=1))
            model = LayerWeightedLogistic().to(args.device)
        else:
            model = torch.nn.Linear(x.shape[1], len(LABELS), device=args.device)
        model.load_state_dict(checkpoint["state_dict"])
        trained = {"model": model, "mean": checkpoint["mean"], "std": checkpoint["std"]}
        probabilities = head_h1.predict(trained, x, np.arange(len(rows)),
                                        available, args.device)
    else:
        frames, lengths = load_test_frames(rows, args.receipts, args.encoder)
        model = head_h2.build_model(SPECS[args.encoder][1], args.encoder == "A3",
                                    args.device)
        model.load_state_dict(checkpoint["state_dict"])
        probabilities = head_h2.predict(model, frames, lengths,
                                        np.arange(len(rows)), args.device, args.batch_size)
        available = lengths > 0
    return probabilities, available, path


def predict_t1_final(args: argparse.Namespace, rows: list[dict]):
    import torch

    result = json.loads(args.final_result.read_text(encoding="utf-8"))
    if result["encoder"] != "T1_chinese_roberta_wwm_ext_large":
        raise ValueError("Expected T1 CV result")
    path = Path(result["final_checkpoint"]["path"])
    if head_h1.sha256(path) != result["final_checkpoint"]["sha256"]:
        raise ValueError("T1 final checkpoint SHA mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != result["final_config"] or tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("T1 final checkpoint differs from CV selection")
    token_path = args.tokens_dir / f"T1_{args.t1_source}_sims_valid.npz"
    token_meta = json.loads((args.tokens_dir / f"T1_{args.t1_source}_sims_valid_meta.json").read_text(encoding="utf-8"))
    if (token_meta["source"] != args.t1_source or token_meta["split"] != "sims_valid" or
            token_meta["manifest_sha256"] != head_h1.sha256(args.adapter) or
            token_meta["folds_sha256"] != head_h1.sha256(args.folds) or
            token_meta["tokens_sha256"] != head_h1.sha256(token_path)):
        raise ValueError("T1 SIMS token cache provenance mismatch")
    with np.load(token_path) as data:
        tokens = {key: data[key].copy() for key in data.files}
    if list(tokens["utt_ids"]) != [row["utt_id"] for row in rows]:
        raise ValueError("T1 SIMS token IDs differ from adapter")
    model = make_model(args.model_dir, args.device, 20260928)
    model.load_state_dict(checkpoint["state_dict"])
    probabilities = predict_t1(model, tokens, np.arange(len(rows)),
                               args.device, args.batch_size)
    return probabilities, tokens["failed"] == 0, path


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    rows, original = load_rows(args.adapter, args.valid_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": provenance, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "adapter_sha256": head_h1.sha256(args.adapter),
            "valid_manifest_sha256": head_h1.sha256(args.valid_manifest),
            "folds_sha256": head_h1.sha256(args.folds),
            "final_result_sha256": head_h1.sha256(args.final_result),
            "encoder": args.encoder, "head": args.head,
            "receipts": str(args.receipts) if args.receipts else None}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    if args.encoder == "T1":
        probabilities, available, checkpoint = predict_t1_final(args, rows)
        statuses = ["ok" if value else "failed" for value in available]
    else:
        probabilities, available, checkpoint = predict_head(args, rows)
        statuses = []
        for row in rows:
            receipt = json.loads((args.receipts / f"{row['utt_id']}.json").read_text(encoding="utf-8"))
            statuses.append(receipt["status"])
    if probabilities.shape != (len(rows), 7) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid SIMS probability matrix")
    output = []
    for row, source, probability, valid, status in zip(rows, original, probabilities,
                                                        available, statuses, strict=True):
        # X01 Benchmark v1 maps seven-class probabilities to valence this way.
        s = float(probability[1] - probability[2] - probability[3] -
                  probability[4] - probability[5]) if valid or status == "no_face" else None
        output.append({"id": row["utt_id"], "label": source["label"],
                       "probabilities": probability.astype(float).tolist(),
                       "s": s, "status": status,
                       "feature_available": bool(valid)})
    path = args.output_dir / "sims_valid.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"encoder": args.encoder, "head": args.head,
              "metrics": sims_metrics(output),
              "feature_coverage": float(np.mean(available)),
              "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
              "checkpoint_path": str(checkpoint),
              "checkpoint_sha256": head_h1.sha256(checkpoint),
              "prediction_path": str(path), "prediction_sha256": head_h1.sha256(path),
              "elapsed_s": time.monotonic() - started}
    write_json_atomic(args.output_dir / "results.json", result)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["elapsed_s"] = result["elapsed_s"]
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--encoder", choices=("A1", "A2", "A3", "V1", "V2",
                                              "V3a", "V3b", "T1", "T2"), required=True)
    parser.add_argument("--head", choices=("h1", "h2", "full_finetune"), required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--tokens-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--t1-source", choices=("o", "r"), default="o")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.encoder == "T1":
        if args.head != "full_finetune" or args.tokens_dir is None or args.model_dir is None:
            parser.error("T1 requires full_finetune, --tokens-dir and --model-dir")
    elif args.head == "full_finetune" or args.receipts is None:
        parser.error("A/V/T2 require h1 or h2 and --receipts")
    run(args)


if __name__ == "__main__":
    main()
