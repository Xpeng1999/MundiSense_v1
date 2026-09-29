"""Freeze Q's CH-SIMS valid-only A/V/AV target manifests."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x02.sims_prepare import validate_adapter
from emotioncore.x03.q_manifest import read_jsonl, sha256, write_jsonl


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    rows = read_jsonl(args.adapter)
    validate_adapter(rows)
    original = read_jsonl(args.valid_manifest)
    if len(original) != len(rows) or any(r["mode"] != "valid" for r in original):
        raise ValueError("Only frozen CH-SIMS valid may be inferred")
    by_id = {f"{r['video_id']}__{r['clip_id']}": r for r in original}
    if len(by_id) != 647 or set(by_id) != {r["utt_id"] for r in rows}:
        raise ValueError("SIMS-valid ID set changed")
    outputs = {}
    for form in ("A", "V", "AV"):
        records = []
        for row in sorted(rows, key=lambda r: r["utt_id"]):
            uid = row["utt_id"]
            receipt_path = args.media_root / "receipts" / f"{uid}.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["utt_id"] != uid or receipt["input_sha256"] != row["mp4"]["sha256"]:
                raise ValueError("SIMS-valid face video receipt mismatch")
            audio_missing = row["wav"]["sha256"] is None
            face_missing = receipt["status"] != "ok"
            if form == "A":
                effective = "uniform" if audio_missing else "A"
            elif form == "V":
                effective = "uniform" if face_missing else "V"
            else:
                effective = "uniform" if audio_missing else ("A" if face_missing else "AV")
            record = {"utt_id": uid, "speaker_id": "sims_valid", "split": "sims_valid",
                      "fold": None, "label": float(by_id[uid]["label"]),
                      "assigned_form": form, "effective_form": effective,
                      "face_missing": face_missing, "audio_missing": audio_missing,
                      "face_missing_fallback": form == "AV" and face_missing and not audio_missing,
                      "wav": str(args.audio_root / row["wav"]["path"]),
                      "wav_sha256": row["wav"]["sha256"]}
            if not face_missing and form in ("V", "AV"):
                video = Path(receipt["output"])
                if sha256(video) != receipt["output_sha256"]:
                    raise ValueError("SIMS-valid Q video SHA mismatch")
                record.update({"video": str(video), "video_sha256": receipt["output_sha256"]})
            records.append(record)
        name = f"final_sims_valid_{form}"
        path = args.output_dir / f"{name}.jsonl"
        outputs[name] = {"path": str(path), "sha256": write_jsonl(path, records),
                         "n": len(records), "uniform": sum(r["effective_form"] == "uniform" for r in records),
                         "face_missing_fallback": sum(r["face_missing_fallback"] for r in records)}
    meta = {"git": git, "command": "emotioncore.x03.q_sims_manifest",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "adapter_sha256": sha256(args.adapter),
            "valid_manifest_sha256": sha256(args.valid_manifest),
            "media_summary_sha256": sha256(args.media_root / "summary.json"),
            "split": "sims_valid", "outputs": outputs}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    write_json_atomic(args.output_dir / "summary.json", outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
