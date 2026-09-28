"""Server-only SenseVoiceSmall transcripts for the X02 text deployment condition."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


ENCODER = "SenseVoiceSmall"
PREPROCESS_VERSION = "1"
EXCLUDED_SPEAKERS = frozenset({"02", "13", "20", "21"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_main(model_dir: Path) -> None:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        model = AutoModel(model=str(model_dir), device="cuda:0", hub="hf",
                          disable_update=True)
    for line in sys.stdin:
        try:
            task = json.loads(line)
            started = time.monotonic()
            wav = Path(task["wav"])
            if sha256(wav) != task["input_sha256"]:
                raise ValueError("Input WAV sha256 differs from frozen manifest")
            with contextlib.redirect_stdout(sys.stderr):
                generated = model.generate(input=str(wav), cache={}, language="zh",
                                           use_itn=False, batch_size=1)
            raw = generated[0]["text"]
            transcript = rich_transcription_postprocess(raw).strip()
            if not isinstance(transcript, str):
                raise ValueError("SenseVoice returned non-text output")
            output = Path(task["output"])
            write_json_atomic(output, {"id": task["id"], "text": transcript})
            response = {"status": "ok", "transcript_path": str(output),
                        "transcript_sha256": sha256(output),
                        "characters": len(transcript), "empty_text": not bool(transcript),
                        "model_seconds": time.monotonic() - started}
        except Exception as exc:
            response = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        protocol.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol.flush()


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    provenance = git_provenance(root, args.allow_dirty)
    manifest_hash = sha256(args.manifest)
    model_hash = sha256(args.model_dir / "model.pt")
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if args.split == "cv" and row["speaker_id"] in EXCLUDED_SPEAKERS:
                continue
            if args.split == "test" and row["speaker_id"] not in EXCLUDED_SPEAKERS:
                continue
            rows.append(row)
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Repeated utterance IDs in input manifest")
    if args.split == "cv" and len(rows) != 16366 and args.limit is None:
        raise ValueError(f"Expected 16,366 CV utterances, got {len(rows)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    base = args.run_root / "asr" / ENCODER
    receipts_dir = base / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base / f"run_meta_{args.split}.json"
    meta = {"git": provenance, "command": sys.argv, "started_at":
            dt.datetime.now(dt.timezone.utc).isoformat(), "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash, "model": str(args.model_dir),
            "model_sha256": model_hash, "encoder": ENCODER,
            "preprocess_version": PREPROCESS_VERSION, "language": "zh",
            "use_itn": False, "vad": False, "split": args.split,
            "count": len(rows), "timeout_s": args.timeout,
            "transcripts_server_only": True}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "model_sha256", "encoder",
                    "preprocess_version", "split"):
            if old[key] != meta[key]:
                raise RuntimeError(f"Cannot resume ASR with changed {key}")
        if old["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume ASR from a different Git commit")
        meta["started_at"] = old["started_at"]
        meta["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(meta_path, meta)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    worker = PersistentJSONWorker([args.worker_python, "-u", "-m",
                                   "emotioncore.x02.asr_sensevoice", "--worker",
                                   "--model-dir", str(args.model_dir)],
                                  cwd=root, env=env, stderr_path=base / "worker_errors.log")
    status = {"ok": 0, "failed": 0}
    try:
        for index, row in enumerate(rows, 1):
            receipt_path = receipts_dir / f"{row['utt_id']}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (receipt["input_sha256"] != row["wav"]["sha256"] or
                        receipt["model_sha256"] != model_hash or
                        receipt["preprocess_version"] != PREPROCESS_VERSION):
                    raise RuntimeError(f"ASR receipt provenance changed: {row['utt_id']}")
            else:
                task = {"id": row["utt_id"],
                        "wav": str(args.media_root / row["wav"]["path"]),
                        "input_sha256": row["wav"]["sha256"],
                        "output": str(base / "transcripts" / f"{row['utt_id']}.json")}
                result = worker.request(task, args.timeout)
                receipt = {"id": row["utt_id"], "speaker_id": row["speaker_id"],
                           "official_split": row["official_split"],
                           "encoder": ENCODER, "model_sha256": model_hash,
                           "preprocess_version": PREPROCESS_VERSION,
                           "input_sha256": row["wav"]["sha256"], **result}
                write_json_atomic(receipt_path, receipt)
            status[receipt["status"]] += 1
            if index % 100 == 0 or index == len(rows):
                write_json_atomic(base / f"progress_{args.split}.json",
                                  {"done": index, "total": len(rows), "status": status})
    finally:
        worker.close()
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["status"] = status
    write_json_atomic(meta_path, meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--split", choices=("cv", "test"), default="cv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker_main(args.model_dir)
    else:
        for name in ("manifest", "media_root", "run_root"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required")
        run(args)


if __name__ == "__main__":
    main()
