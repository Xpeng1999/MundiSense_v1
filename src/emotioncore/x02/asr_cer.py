"""Calculate annotation vs SenseVoice CER without exporting transcripts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.text_t2 import text_path


EXCLUDED_SPEAKERS = frozenset({"02", "13", "20", "21"})


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_char in enumerate(reference, 1):
        current = [row_index]
        for col_index, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[col_index] + 1,
                               previous[col_index - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(counter: dict) -> dict:
    result = dict(counter)
    result["cer_success_only"] = (counter["edit_distance"] / counter["reference_chars"]
                                  if counter["reference_chars"] else None)
    result["coverage"] = (counter["success"] / counter["total"] if counter["total"] else None)
    return result


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    by_speaker = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0,
                                      "reference_chars": 0, "edit_distance": 0,
                                      "empty_reference": 0, "empty_asr": 0})
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if args.split == "cv" and row["speaker_id"] in EXCLUDED_SPEAKERS:
                continue
            if args.split == "test" and row["speaker_id"] not in EXCLUDED_SPEAKERS:
                continue
            rows.append(row)
    if args.split == "cv" and len(rows) != 16366:
        raise ValueError("CV count differs from frozen protocol")
    for row in rows:
        counter = by_speaker[row["speaker_id"]]
        counter["total"] += 1
        annotated = json.loads(text_path(args.media_root, row["wav"]["path"]).read_text())
        reference = normalize(annotated["content"])
        if not reference:
            counter["empty_reference"] += 1
        receipt_path = args.asr_receipts / f"{row['utt_id']}.json"
        if not receipt_path.is_file():
            raise ValueError(f"Missing ASR receipt: {row['utt_id']}")
        receipt = json.loads(receipt_path.read_text())
        if receipt["input_sha256"] != row["wav"]["sha256"]:
            raise ValueError(f"ASR input hash mismatch: {row['utt_id']}")
        if receipt["status"] != "ok":
            counter["failed"] += 1
            continue
        transcript_path = Path(receipt["transcript_path"])
        if sha256(transcript_path) != receipt["transcript_sha256"]:
            raise ValueError(f"ASR output hash mismatch: {row['utt_id']}")
        decoded = json.loads(transcript_path.read_text())
        if decoded["id"] != row["utt_id"]:
            raise ValueError("ASR transcript ID mismatch")
        hypothesis = normalize(decoded["text"])
        if not hypothesis:
            counter["empty_asr"] += 1
        counter["success"] += 1
        counter["reference_chars"] += len(reference)
        counter["edit_distance"] += edit_distance(reference, hypothesis)
    total = {key: sum(item[key] for item in by_speaker.values())
             for key in next(iter(by_speaker.values()))}
    result = {"normalization": "Unicode NFC, strip leading and trailing whitespace",
              "split": args.split, "overall": summary(total),
              "by_speaker": {speaker: summary(item)
                             for speaker, item in sorted(by_speaker.items())}}
    write_json_atomic(args.output, result)
    write_json_atomic(args.output.with_name("cer_run_meta.json"),
                      {"git": provenance, "command": sys.argv,
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "manifest_sha256": sha256(args.manifest),
                       "result_sha256": sha256(args.output),
                       "transcripts_server_only": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--asr-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("cv", "test"), default="cv")
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
