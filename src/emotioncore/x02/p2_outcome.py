"""Record the ModelScope Qwen3-Omni load and ET-val probe outcome."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


def run(args: argparse.Namespace) -> None:
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    old = json.loads(args.prior.read_text(encoding="utf-8"))
    downloads = json.loads(args.download_receipt.read_text(encoding="utf-8"))
    migration = json.loads(args.migration_receipt.read_text(encoding="utf-8"))
    model = downloads["models"]["P2_modelscope"]
    if (model["status"] != "verified" or
            migration["verified_bytes"] != model["verified_bytes"] or
            migration["target_path"] != str(args.model_dir.resolve()) or
            len(set(old["label_first_token_ids"])) != 7):
        raise ValueError("P2 download, migration, or first-token verification failed")
    output = dict(old)
    output.update(download_source="ModelScope", model_dir=str(args.model_dir),
                  model_storage_target=migration["target_path"],
                  modelscope_revision=model["revision"],
                  modelscope_verified_bytes=model["verified_bytes"],
                  modelscope_receipt_sha256=sha256(args.download_receipt),
                  migration_receipt_sha256=sha256(args.migration_receipt),
                  model_load_attempted=True, git=provenance,
                  command=sys.argv,
                  recorded_at=dt.datetime.now(dt.timezone.utc).isoformat())
    if args.probe_result is not None:
        probe = json.loads(args.probe_result.read_text(encoding="utf-8"))
        if probe.get("probe") != "P2" or set(probe["conditions"]) != {"audio", "video", "av"}:
            raise ValueError("Incomplete P2 condition result")
        run_meta = json.loads((args.probe_result.parent / "run_meta.json").read_text(encoding="utf-8"))
        if not run_meta.get("ended_at") or len(set(run_meta["first_token_ids"])) != 7:
            raise ValueError("P2 probe run did not finish or token IDs collided")
        output.update(status="completed", reason="ModelScope snapshot verified; NF4 thinker scored ET-val A/V/AV.",
                      scoring_attempted=True, probe_result_path=str(args.probe_result),
                      probe_result_sha256=sha256(args.probe_result),
                      probe_run_meta_sha256=sha256(args.probe_result.parent / "run_meta.json"),
                      label_first_token_ids=run_meta["first_token_ids"])
    else:
        if not args.failure_reason or not args.attempt_log or not args.attempt_log.is_file():
            raise ValueError("Failed P2 load requires a reason and attempt log")
        output.update(status="skipped_after_model_load_failure",
                      reason=args.failure_reason, scoring_attempted=False,
                      attempt_log_path=str(args.attempt_log),
                      attempt_log_sha256=sha256(args.attempt_log))
    write_json_atomic(args.output, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--migration-receipt", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--attempt-log", type=Path)
    parser.add_argument("--failure-reason")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
