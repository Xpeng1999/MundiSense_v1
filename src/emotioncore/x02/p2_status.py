"""Record the observed P2 download limit and first-token uniqueness."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


LABELS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise")


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    p2 = plan["models"]["P2"]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    ids = {label: tokenizer(label, add_special_tokens=False).input_ids[0] for label in LABELS}
    retries = [int(value) for value in re.findall(
        r"received (\d+) bytes", args.snapshot_log.read_text(encoding="utf-8", errors="replace"))]
    if args.sample_bytes <= 0 or args.sample_seconds <= 0:
        raise ValueError("A positive observed Range sample is required")
    speed = args.sample_bytes / args.sample_seconds
    estimate_h = p2["planned_bytes"] / speed / 3600
    result = {"probe": "P2", "status": "skipped_before_model_load",
              "reason": "Observed source transfer rate makes the 70.5 GB download exceed one workday",
              "repo_id": p2["repo_id"], "revision": p2["revision"],
              "source_url": p2["source_url"], "planned_bytes": p2["planned_bytes"],
              "x02_total_planned_bytes": plan["total_planned_bytes"],
              "range_sample_bytes": args.sample_bytes,
              "range_sample_seconds": args.sample_seconds,
              "range_sample_bytes_per_second": speed,
              "estimated_download_hours_at_observed_rate": estimate_h,
              "snapshot_failed_response_count": len(retries),
              "snapshot_failed_response_received_bytes": sum(retries),
              "label_first_token_ids": ids,
              "label_first_tokens_unique": len(set(ids.values())) == len(ids),
              "model_load_attempted": False,
              "scoring_attempted": False,
              "note": "32 GB load feasibility is unverified because model weights were not fully downloaded.",
              "git": provenance, "command": sys.argv,
              "plan_sha256": sha256(args.plan),
              "snapshot_log_sha256": sha256(args.snapshot_log),
              "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json_atomic(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--snapshot-log", type=Path, required=True)
    parser.add_argument("--sample-bytes", type=int, required=True)
    parser.add_argument("--sample-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
