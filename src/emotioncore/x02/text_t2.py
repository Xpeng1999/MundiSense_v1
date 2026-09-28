"""Extract BGE-large-zh-v1.5 sentence vectors from server-local annotations."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic


ENCODER = "T2_bge_large_zh_v1.5"
REVISION = "79e7739b6ab944e86d6171e44d24c997fc1e0116"
PREPROCESS_VERSION = "1"
EXCLUDED_SPEAKERS = frozenset({"02", "13", "20", "21"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_path(media_root: Path, wav_path: str) -> Path:
    wav = Path(wav_path)
    if wav.parts[:2] != ("Audio", "wav") or wav.suffix != ".wav":
        raise ValueError(f"Unexpected EmotionTalk WAV path: {wav_path}")
    return media_root / "Text" / "json" / Path(*wav.parts[2:]).with_suffix(".json")


def worker_main(model_dir: Path) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval().to("cuda:0")
    for line in sys.stdin:
        try:
            task = json.loads(line)
            started = time.monotonic()
            texts = []
            hashes = []
            for item in task["items"]:
                path = Path(item["text_json"])
                digest = sha256(path)
                source = json.loads(path.read_text(encoding="utf-8"))
                content = source["content"]
                if not isinstance(content, str):
                    raise ValueError("Text content is not a string")
                texts.append(content)
                hashes.append(digest)
            inputs = tokenizer(texts, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state[:, 0]
                vectors = F.normalize(hidden.float(), p=2, dim=1).cpu().numpy()
            results = []
            for item, vector, digest, content in zip(task["items"], vectors, hashes, texts):
                if vector.shape != (1024,) or not np.isfinite(vector).all():
                    raise ValueError("Unexpected or non-finite BGE vector")
                output = Path(item["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f".{output.stem}.{os.getpid()}.npy")
                np.save(temporary, vector.astype(np.float32))
                os.replace(temporary, output)
                results.append({"id": item["id"], "status": "ok",
                                "input_sha256": digest, "feature_path": str(output),
                                "feature_sha256": sha256(output), "shape": [1024],
                                "characters": len(content), "empty_text": not bool(content.strip())})
            response = {"status": "ok", "items": results,
                        "batch_seconds": time.monotonic() - started}
        except Exception as exc:
            response = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


def run(args: argparse.Namespace) -> None:
    root = Path.cwd()
    provenance = git_provenance(root, args.allow_dirty)
    manifest_hash = sha256(args.manifest)
    model_hash = sha256(args.model_dir / "pytorch_model.bin")
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if args.split == "cv" and row["speaker_id"] in EXCLUDED_SPEAKERS:
                continue
            if args.split == "test" and row["speaker_id"] not in EXCLUDED_SPEAKERS:
                continue
            rows.append(row)
    ids = [row["utt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated utterance IDs in input manifest")
    if args.split == "cv" and len(rows) != 16366 and args.limit is None:
        raise ValueError(f"Expected 16,366 CV utterances, got {len(rows)}")
    if args.limit is not None:
        rows = rows[:args.limit]
    base = args.run_root / "features" / "T2_o"
    receipts_dir = base / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    meta_path = base / f"run_meta_{args.split}.json"
    meta = {"git": provenance, "command": sys.argv, "started_at":
            dt.datetime.now(dt.timezone.utc).isoformat(), "manifest": str(args.manifest),
            "manifest_sha256": manifest_hash, "model": str(args.model_dir),
            "model_sha256": model_hash, "revision": REVISION,
            "encoder": ENCODER, "preprocess_version": PREPROCESS_VERSION,
            "split": args.split, "count": len(rows), "timeout_s": args.timeout,
            "batch_size": args.batch_size}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("manifest_sha256", "model_sha256", "revision", "encoder",
                    "preprocess_version", "split"):
            if old[key] != meta[key]:
                raise RuntimeError(f"Cannot resume T2 with changed {key}")
        if old["git"]["commit"] != meta["git"]["commit"]:
            raise RuntimeError("Cannot resume T2 from a different Git commit")
        meta["started_at"] = old["started_at"]
        meta["resumed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_atomic(meta_path, meta)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    worker = PersistentJSONWorker([args.worker_python, "-u", "-m",
                                   "emotioncore.x02.text_t2", "--worker",
                                   "--model-dir", str(args.model_dir)],
                                  cwd=root, env=env, stderr_path=base / "worker_errors.log")
    status = {"ok": 0, "failed": 0}
    try:
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            missing = [row for row in batch if not (receipts_dir / f"{row['utt_id']}.json").exists()]
            if missing:
                items = [{"id": row["utt_id"],
                          "text_json": str(text_path(args.media_root, row["wav"]["path"])),
                          "output": str(base / "data" / f"{row['utt_id']}.npy")}
                         for row in missing]
                response = worker.request({"items": items}, args.timeout)
                by_id = {item["id"]: item for item in response.get("items", [])}
                for row in missing:
                    result = by_id.get(row["utt_id"], {"status": "failed",
                                "error": response.get("error", "Missing result in BGE batch")})
                    receipt = {"id": row["utt_id"], "speaker_id": row["speaker_id"],
                               "official_split": row["official_split"],
                               "encoder": ENCODER, "revision": REVISION,
                               "model_sha256": model_hash,
                               "preprocess_version": PREPROCESS_VERSION, **result}
                    write_json_atomic(receipts_dir / f"{row['utt_id']}.json", receipt)
            for row in batch:
                receipt = json.loads((receipts_dir / f"{row['utt_id']}.json").read_text())
                if (receipt["model_sha256"] != model_hash or
                        receipt["preprocess_version"] != PREPROCESS_VERSION):
                    raise RuntimeError(f"T2 receipt provenance changed: {row['utt_id']}")
                status[receipt["status"]] += 1
            done = offset + len(batch)
            if done % 128 < args.batch_size or done == len(rows):
                write_json_atomic(base / f"progress_{args.split}.json",
                                  {"done": done, "total": len(rows), "status": status})
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
    parser.add_argument("--batch-size", type=int, default=16)
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
        if args.batch_size < 1 or args.timeout <= 0:
            parser.error("--batch-size and --timeout must be positive")
        run(args)


if __name__ == "__main__":
    main()
