"""Verify ModelScope snapshots against a frozen file-size plan and hash every file."""
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
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    receipts = {}
    for name, model in plan["models"].items():
        root = args.run_root / "models" / name
        files = []
        missing = []
        wrong_size = []
        for expected in model["planned_files"]:
            relative = Path(expected["Path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ModelScope path: {relative}")
            path = root / relative
            if not path.is_file():
                missing.append(str(relative))
                continue
            expected_bytes = int(expected["Size"])
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                wrong_size.append({"path": str(relative), "expected": expected_bytes,
                                   "actual": actual_bytes})
                continue
            files.append({"path": str(path), "bytes": actual_bytes,
                          "sha256": sha256(path)})
        status = "verified" if not missing and not wrong_size else "incomplete"
        receipts[name] = {"repo_id": model["repo_id"],
                          "source_url": model["source_url"],
                          "revision": model["revision"],
                          "revision_note": "ModelScope master is a mutable branch; local SHA-256 values pin these bytes.",
                          "status": status, "planned_bytes": model["planned_bytes"],
                          "verified_bytes": sum(item["bytes"] for item in files),
                          "files": files, "missing": missing, "wrong_size": wrong_size}
    result = {"models": receipts,
              "prior_verified_bytes": plan["prior_verified_bytes"],
              "modelscope_verified_bytes": sum(item["verified_bytes"] for item in receipts.values()),
              "planned_total_bytes": plan["planned_total_bytes"],
              "plan_path": str(args.plan), "plan_sha256": sha256(args.plan),
              "verified_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    prior_p2 = args.run_root / "p2_probe_status.json"
    failed_transfer = json.loads(prior_p2.read_text(encoding="utf-8")) if prior_p2.exists() else {}
    result["prior_failed_transfer_bytes"] = (
        int(failed_transfer.get("snapshot_failed_response_received_bytes", 0)) +
        int(failed_transfer.get("range_sample_bytes", 0)))
    result["all_public_model_bytes"] = (result["prior_verified_bytes"] +
                                        result["modelscope_verified_bytes"] +
                                        result["prior_failed_transfer_bytes"])
    if result["all_public_model_bytes"] > 100_000_000_000:
        raise ValueError("X02 100 GB public model download budget exceeded")
    write_json_atomic(args.output, result)
    write_json_atomic(args.output.with_name("modelscope_downloads_run_meta.json"),
                      {"git": provenance, "command": sys.argv,
                       "plan_sha256": result["plan_sha256"],
                       "receipt_sha256": sha256(args.output),
                       "ended_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    if any(item["status"] != "verified" for item in receipts.values()):
        raise RuntimeError("At least one ModelScope snapshot is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
