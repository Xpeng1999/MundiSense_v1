"""Read-only replay of the X05 expert/FMH leakage gate; never loads a model.

Run on the experiment server through stdin from a clean, committed checkout.
JSON is emitted to stdout, and exit status 2 means the protocol gate failed.
No predictions, labels, or per-utterance IDs are emitted.
"""
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    root = Path("/mnt/h/emotionAI/emotioncore_runs/X04a/heads/D1")
    fmh = {"06", "10", "11"}
    evidence = []
    for encoder in ("A3", "V2", "V1"):
        job = root / encoder
        path = job / "fold0/training_ids.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        overlap = [r for r in rows if r["corpus"] == "ET" and r["speaker_id"] in fmh]
        result_path = job / "results.json"
        result = json.loads(result_path.read_text())
        checkpoint = job / "final_head.pt"
        checkpoint_sha = digest(checkpoint)
        if checkpoint_sha != result["checkpoint_sha256"]:
            raise ValueError("Existing final expert checkpoint SHA changed")
        evidence.append({
            "encoder": encoder,
            "fold0_training_ids": {"path": str(path), "sha256": digest(path)},
            "fold0_training_count": len(rows),
            "fmh_training_overlap_count": len(overlap),
            "fmh_training_overlap_unique_ids": len({r["utt_id"] for r in overlap}),
            "fmh_training_overlap_by_speaker": dict(sorted(collections.Counter(
                r["speaker_id"] for r in overlap).items())),
            "saved_pt_checkpoints": [str(p.relative_to(job)) for p in sorted(job.rglob("*.pt"))],
            "fold0_files": sorted(p.name for p in (job / "fold0").iterdir()),
            "final_checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
            "results": {"path": str(result_path), "sha256": digest(result_path)},
            "original_run_meta": {"path": str(job / "run_meta.json"),
                                  "sha256": digest(job / "run_meta.json")},
        })
    failed = any(r["fmh_training_overlap_count"] for r in evidence)
    output = {
        "status": "STOP" if failed else "PASS",
        "gate": "expert_fold0_fmh_training_disjointness",
        "started_at": started,
        "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fmh_speakers": sorted(fmh),
        "expected_fmh_count": 3266,
        "server_root_available_bytes": shutil.disk_usage("/").free,
        "experts": evidence,
        "training_started": False,
        "inference_started": False,
        "method_selection_started": False,
        "q_mix_analysis_started": False,
        "gpu_hours": 0,
        "read_only_replay": True,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
