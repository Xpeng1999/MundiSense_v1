"""Pack X02 cached target features and run the frozen E2 final head."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS
from emotioncore.x02.frames_pack import SPECS, read_feature
from emotioncore.x02.head_h1 import SEED, sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.e2 import CONDITIONS, E2Model, load_cv, predict, train_once


def target_rows(manifest: Path, split: str) -> list[dict]:
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if split == "et_test":
        rows = [row for row in rows if row["official_split"] == "test"]
        expected = 1929
    elif split == "sims_valid":
        if any(row["official_split"] != "sims_valid" for row in rows):
            raise ValueError("SIMS target manifest includes another partition")
        expected = 647
    else:
        raise ValueError("Only ET-test and SIMS-valid are target partitions")
    if len(rows) != expected or len({row["utt_id"] for row in rows}) != expected:
        raise ValueError("Target sample count or IDs differ from frozen protocol")
    return rows


def pack(args) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    rows = target_rows(args.manifest, args.split)
    face = json.loads(args.face_policy.read_text(encoding="utf-8"))
    missing_ids = set(face["face_missing_utt_ids"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for encoder, root in (("A3", args.a3_receipts), ("V2", args.v2_receipts)):
        max_frames, width = SPECS[encoder]
        shape = ((len(rows), 4, max_frames, width) if encoder == "A3"
                 else (len(rows), max_frames, width))
        target = args.output_dir / f"{encoder}_frames.npy"
        if target.exists():
            raise FileExistsError(f"Target pack already exists: {target}")
        temporary = target.with_name("." + target.name + ".pending.npy")
        frames = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
        frames[:] = 0
        lengths = np.zeros(len(rows), dtype=np.int16)
        aggregate = hashlib.sha256()
        status = {"ok": 0, "failed": 0, "no_face": 0, "face_missing": 0}
        try:
            for index, row in enumerate(rows):
                receipt_path = root / f"{row['utt_id']}.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                source_sha = row["wav"]["sha256"] if encoder == "A3" else row["mp4"]["sha256"]
                if receipt["id"] != row["utt_id"] or receipt["input_sha256"] != source_sha:
                    raise ValueError(f"{encoder} target receipt provenance mismatch: {row['utt_id']}")
                state = receipt["status"]
                if encoder == "V2" and row["utt_id"] in missing_ids:
                    state = "face_missing"
                if state not in status:
                    raise ValueError(f"Unsupported target feature status: {state}")
                status[state] += 1
                aggregate.update(f"{row['utt_id']}:{state}:{receipt.get('feature_sha256','')}\n".encode())
                if state == "ok":
                    feature = Path(receipt["feature_path"])
                    if sha256(feature) != receipt["feature_sha256"]:
                        raise ValueError(f"{encoder} target feature SHA mismatch: {row['utt_id']}")
                    source = read_feature(feature, encoder)
                    count = source.shape[-2]
                    if encoder == "A3" and (source.ndim != 3 or source.shape[0] != 4):
                        raise ValueError("A3 target layer count differs")
                    if count < 1 or source.shape[-1] != width or not np.isfinite(source).all():
                        raise ValueError("Malformed target frame feature")
                    selected = np.linspace(0, count - 1, min(count, max_frames)).round().astype(int)
                    frames[index, ..., :len(selected), :] = source[..., selected, :]
                    lengths[index] = len(selected)
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    frames.flush()
                    write_json_atomic(args.output_dir / f"{encoder}_progress.json",
                                      {"done": index + 1, "total": len(rows), "status": status})
            frames.flush()
            del frames
            temporary.replace(target)
        except BaseException:
            del frames
            temporary.unlink(missing_ok=True)
            raise
        lengths_path = args.output_dir / f"{encoder}_lengths.npy"
        np.save(lengths_path, lengths)
        provenance[encoder] = {"frames_path": str(target), "frames_sha256": sha256(target),
                               "lengths_path": str(lengths_path),
                               "lengths_sha256": sha256(lengths_path),
                               "source_receipts": str(root),
                               "source_receipt_aggregate_sha256": aggregate.hexdigest(),
                               "shape": list(shape), "status": status}
    meta = {"git": git, "command": sys.argv,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "split": args.split, "manifest_sha256": sha256(args.manifest),
            "face_policy_sha256": sha256(args.face_policy),
            "ids_sha256": hashlib.sha256("\n".join(row["utt_id"] for row in rows).encode()).hexdigest(),
            "count": len(rows), "packs": provenance}
    write_json_atomic(args.output_dir / "target_pack_meta.json", meta)


def load_target(args, rows):
    meta = json.loads((args.target_pack / "target_pack_meta.json").read_text(encoding="utf-8"))
    if (meta["split"] != args.split or meta["manifest_sha256"] != sha256(args.target_manifest)
            or meta["count"] != len(rows) or meta["face_policy_sha256"] != sha256(args.face_policy)):
        raise ValueError("Target pack provenance mismatch")
    arrays = {}
    for encoder in ("A3", "V2"):
        spec = meta["packs"][encoder]
        if sha256(Path(spec["frames_path"])) != spec["frames_sha256"] or sha256(Path(spec["lengths_path"])) != spec["lengths_sha256"]:
            raise ValueError(f"Target {encoder} pack SHA mismatch")
        arrays[encoder] = (np.load(spec["frames_path"], mmap_mode="r"), np.load(spec["lengths_path"]))
    with np.load(args.target_tokens) as archive:
        ids = [str(uid) for uid in archive["utt_ids"]]
        failed = archive["failed"].copy()
    if ids != [row["utt_id"] for row in rows]:
        raise ValueError("Target token IDs not aligned to manifest")
    cls_meta = json.loads(args.target_cls_meta.read_text(encoding="utf-8"))
    if (sha256(args.target_cls) != cls_meta["feature_sha256"] or
            sha256(args.target_tokens) != cls_meta["tokens_sha256"] or
            not cls_meta["checkpoint_policy"].startswith("X02 unfine-tuned T1 base")):
        raise ValueError("Target T1 CLS provenance mismatch")
    cls = np.load(args.target_cls, mmap_mode="r")
    if cls.shape != (len(rows), 1024):
        raise ValueError("Target T1 CLS shape mismatch")
    face = json.loads(args.face_policy.read_text(encoding="utf-8"))
    missing_ids = set(face["face_missing_utt_ids"])
    return {"audio": arrays["A3"][0], "audio_lengths": arrays["A3"][1],
            "visual": arrays["V2"][0], "visual_lengths": arrays["V2"][1],
            "text": cls, "text_available": failed == 0,
            "face_missing": np.asarray([row["utt_id"] in missing_ids for row in rows])}


def evaluate(args):
    import torch

    started = time.monotonic()
    git = git_provenance(Path.cwd(), args.allow_dirty)
    torch.set_num_threads(args.threads)
    cv = load_cv(args)
    y = np.asarray([LABELS.index(row["labels"]["M"]) for row in cv["rows"]], dtype=np.int64)
    results = json.loads(args.cv_results.read_text(encoding="utf-8"))
    config = results["final_config"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "final_model.pt"
    if not checkpoint.exists():
        trained = train_once(cv, y, np.arange(len(y)), None,
                             config["learning_rate"], config["dropout"],
                             fixed_epochs=config["best_epoch"], seed=SEED,
                             device=args.device, batch_size=args.batch_size)
        torch.save(trained["model"].state_dict(), checkpoint)
        model = trained["model"]
    else:
        model = E2Model.build(config["dropout"]).to(args.device)
        model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True))
    rows = target_rows(args.target_manifest, args.split)
    target = load_target(args, rows)
    results_by_condition = {}
    for condition in CONDITIONS:
        probabilities = predict(model, target, np.arange(len(rows)), condition,
                                args.device, args.batch_size)
        output = args.output_dir / f"{args.split}_{condition}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for row, probability, missing in zip(rows, probabilities, target["face_missing"]):
                item = {"utt_id": row["utt_id"], "speaker_id": row["speaker_id"],
                        "condition": condition, "probabilities": probability.astype(float).tolist(),
                        "status": "face_missing" if missing else "ok"}
                if args.split == "et_test":
                    item["truth_M"] = row["labels"]["M"]
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        results_by_condition[condition] = {"path": str(output), "sha256": sha256(output)}
    meta = {"git": git, "command": sys.argv,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_s": time.monotonic() - started,
            "cv_results_sha256": sha256(args.cv_results),
            "target_pack_meta_sha256": sha256(args.target_pack / "target_pack_meta.json"),
            "target_cls_meta_sha256": sha256(args.target_cls_meta),
            "final_config": config,
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "outputs": results_by_condition}
    write_json_atomic(args.output_dir / f"{args.split}_run_meta.json", meta)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack_parser = sub.add_parser("pack")
    pack_parser.add_argument("--manifest", type=Path, required=True)
    pack_parser.add_argument("--split", choices=("et_test", "sims_valid"), required=True)
    pack_parser.add_argument("--a3-receipts", type=Path, required=True)
    pack_parser.add_argument("--v2-receipts", type=Path, required=True)
    pack_parser.add_argument("--face-policy", type=Path, required=True)
    pack_parser.add_argument("--output-dir", type=Path, required=True)
    pack_parser.add_argument("--allow-dirty", action="store_true")
    eval_parser = sub.add_parser("evaluate")
    for flag in ("manifest", "folds", "a3-pack", "v2-pack", "t1-tokens",
                 "t1-cls", "t1-cls-meta", "face-policy", "target-manifest",
                 "target-pack", "target-tokens", "target-cls", "target-cls-meta",
                 "cv-results", "output-dir"):
        eval_parser.add_argument("--" + flag, type=Path, required=True)
    eval_parser.add_argument("--split", choices=("et_test", "sims_valid"), required=True)
    eval_parser.add_argument("--device", default="cpu")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--threads", type=int, default=8)
    eval_parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
