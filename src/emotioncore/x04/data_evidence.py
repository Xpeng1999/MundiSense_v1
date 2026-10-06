"""Aggregate new-corpus face/decode evidence; MER images stay on the server."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from emotioncore.x02 import face
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.prepare import require_h, storage_gate


def run(args):
    git = git_provenance(Path.cwd()); require_h(args.derived); require_h(args.output); storage_gate()
    args.output.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = args.derived/"manifests/all_features.jsonl"
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    groups = {}; mer_multiface = []
    for i, row in enumerate(rows):
        if i % 100 == 0: storage_gate()
        path = args.derived/"full/face/receipts"/(row["utt_id"]+".json")
        rec = json.loads(path.read_text())
        if rec["input_sha256"] != row["mp4"]["sha256"]: raise ValueError("Face receipt input changed")
        name = row["corpus"]+"_"+row["official_split"]
        groups.setdefault(name, []).append(rec)
        if row["corpus"] == "MER" and rec.get("multi_face_frames", 0) > 0 and rec.get("preview_path"):
            mer_multiface.append(rec)
    mer_multiface.sort(key=lambda r: r["id"])
    selected = random.Random(20261006).sample(mer_multiface, min(60, len(mer_multiface)))
    sheets = face.contact_sheets(selected, args.derived/"MER_contact_sheets")
    contact = {"seed": 20261006, "requested_count": 60, "sample_count": len(selected),
        "available_multiface_preview_count": len(mer_multiface), "MER_only": True,
        "sheets": sheets, "server_only": True}
    write_json_atomic(args.derived/"MER_contact_sheets/selection.json", contact)
    result = {"cohorts": {name: face.summarize(receipts) for name, receipts in groups.items()},
        "contact_sheets": contact, "manifest_path": str(manifest), "manifest_sha256": face.sha256(manifest),
        "decode_definition": "decoded frame/read failures from unchanged X02 face receipts; ffprobe registration is header-only"}
    write_json_atomic(args.output/"face_evidence.json", result)
    write_json_atomic(args.output/"run_meta.json", {"git": git, "command": sys.argv,
        "started_at": started, "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "face_evidence_sha256": face.sha256(args.output/"face_evidence.json")})


def main():
    p = argparse.ArgumentParser(); p.add_argument("--derived", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); run(p.parse_args())


if __name__ == "__main__": main()
