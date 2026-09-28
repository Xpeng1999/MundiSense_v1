"""Cache server-only T1 token IDs for annotated and SenseVoice text."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from emotioncore.x02.head_h1 import load_rows, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.text_t2 import text_path


REVISION = "a25cc9e05974bd9687e528edd516f2cfdb3f5db9"
MAX_LENGTH = 128


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    if sha256(args.manifest) != folds["source_manifest_sha256"]:
        raise ValueError("Manifest differs from frozen folds")
    rows = load_rows(args.manifest, folds)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    ids = np.zeros((len(rows), MAX_LENGTH), dtype=np.int32)
    mask = np.zeros((len(rows), MAX_LENGTH), dtype=np.int8)
    status = np.zeros(len(rows), dtype=np.int8)
    sources = hashlib.sha256()
    max_tokens = 0
    truncated = 0
    for index, row in enumerate(rows):
        if args.source == "o":
            path = text_path(args.media_root, row["wav"]["path"])
            source = json.loads(path.read_text(encoding="utf-8"))
            content = source["content"]
        else:
            receipt_path = args.asr_receipts / f"{row['utt_id']}.json"
            if not receipt_path.is_file():
                raise ValueError(f"Missing ASR receipt: {row['utt_id']}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["input_sha256"] != row["wav"]["sha256"]:
                raise ValueError(f"ASR input sha256 mismatch: {row['utt_id']}")
            if receipt["status"] != "ok":
                status[index] = 1
                sources.update(f"{row['utt_id']}:failed\n".encode())
                continue
            path = Path(receipt["transcript_path"])
            if sha256(path) != receipt["transcript_sha256"]:
                raise ValueError(f"ASR transcript hash mismatch: {row['utt_id']}")
            source = json.loads(path.read_text(encoding="utf-8"))
            if source["id"] != row["utt_id"]:
                raise ValueError("ASR transcript ID mismatch")
            content = source["text"]
        digest = sha256(path)
        sources.update(f"{row['utt_id']}:{digest}\n".encode())
        if not isinstance(content, str):
            raise ValueError("Text content is not a string")
        tokenized = tokenizer(content, truncation=False)
        if len(tokenized["input_ids"]) > MAX_LENGTH:
            if args.source == "o":
                raise ValueError(f"Annotated text exceeds frozen 128-token limit: {row['utt_id']}")
            truncated += 1
            tokenized = tokenizer(content, truncation=True, max_length=MAX_LENGTH)
        max_tokens = max(max_tokens, len(tokenized["input_ids"]))
        width = len(tokenized["input_ids"])
        ids[index, :width] = tokenized["input_ids"]
        mask[index, :width] = tokenized["attention_mask"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"T1_{args.source}_cv.npz"
    np.savez(output, input_ids=ids, attention_mask=mask, failed=status,
             utt_ids=np.asarray([row["utt_id"] for row in rows]))
    meta = {"git": provenance, "command": sys.argv,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": args.source, "split": "cv", "revision": REVISION,
            "model_dir": str(args.model_dir),
            "model_sha256": sha256(args.model_dir / "pytorch_model.bin"),
            "tokenizer_sha256": sha256(args.model_dir / "tokenizer.json"),
            "manifest_sha256": sha256(args.manifest), "folds_sha256": sha256(args.folds),
            "input_aggregate_sha256": sources.hexdigest(),
            "max_length": MAX_LENGTH, "maximum_observed_tokens": max_tokens,
            "truncated": truncated,
            "samples": len(rows), "failed": int(status.sum()),
            "tokens_path": str(output), "tokens_sha256": sha256(output),
            "transcripts_server_only": True}
    write_json_atomic(args.output_dir / f"T1_{args.source}_cv_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asr-receipts", type=Path)
    parser.add_argument("--source", choices=("o", "r"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.source == "r" and args.asr_receipts is None:
        parser.error("--asr-receipts is required for source r")
    run(args)


if __name__ == "__main__":
    main()
