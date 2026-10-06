"""Frozen text candidate exclusion; no CH-SIMS labels or learned inference."""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
import shutil
import sys
import unicodedata
from pathlib import Path

from emotioncore.x02.face import sha256
from emotioncore.x02.runtime import git_provenance, run_with_hard_timeout, write_json_atomic
from emotioncore.x04.prepare import probe, require_h, storage_gate, write_rows


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if c.isalpha() or c.isdigit())


def bigrams(text: str) -> set:
    return set(zip(text, text[1:]))


def match(a: str, b: str, duration_a: float | None = None,
          duration_b: float | None = None) -> float | None:
    if not a or not b:
        return None
    if min(len(a), len(b)) < 5:
        return 1.0 if a == b and duration_a is not None and duration_b is not None and abs(duration_a-duration_b) <= .3 + 1e-9 else None
    if a == b:
        return 1.0
    ga, gb = bigrams(a), bigrams(b)
    score = len(ga & gb)/len(ga | gb)
    return score if score >= .7 else None


def candidates(source: list[dict], targets: list[dict], duration) -> list[dict]:
    inverted = collections.defaultdict(set)
    exact = collections.defaultdict(set)
    for i, row in enumerate(targets):
        if row["normalized"]:
            exact[row["normalized"]].add(i)
        if len(row["normalized"]) >= 5:
            for gram in bigrams(row["normalized"]):
                inverted[gram].add(i)
    output = []
    for a in source:
        x = a["normalized"]
        indices = set(exact.get(x, set()))
        if len(x) >= 5:
            for gram in bigrams(x):
                indices.update(inverted[gram])
        for i in sorted(indices):
            b = targets[i]
            da = duration(a) if min(len(x), len(b["normalized"])) < 5 else None
            db = duration(b) if min(len(x), len(b["normalized"])) < 5 else None
            score = match(x, b["normalized"], da, db)
            if score is not None:
                output.append({"source_id": a["utt_id"], "target_id": b["utt_id"],
                    "similarity": score, "short_exact": len(x) < 5})
    return output


def run(args):
    import pandas as pd
    storage_gate(); require_h(args.derived)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"git": git_provenance(Path.cwd()), "command": sys.argv,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sims_usecols": ["video_id", "clip_id", "text", "mode"],
        "sims_model_inference": 0, "sims_meta_sha256": sha256(args.sims_meta),
        "subtitle_sha256": sha256(args.mer / "subtitle_chieng.csv"),
        "fingerprint_method": "exact complete raw Chromaprint sequence equality"}
    write_json_atomic(args.output / "run_meta.json", meta)
    # pandas usecols prevents constructing or accessing any label columns.
    sims = pd.read_csv(args.sims_meta, usecols=meta["sims_usecols"], dtype=str, keep_default_na=False)
    subtitles = pd.read_csv(args.mer / "subtitle_chieng.csv", usecols=["name", "chinese"], dtype=str, keep_default_na=False)
    texts = dict(zip(subtitles.name, subtitles.chinese))
    def mer_rows(split):
        rows = [json.loads(line) for line in (args.derived / f"manifests/mer_{split}.jsonl").read_text().splitlines()]
        for row in rows:
            row["normalized"] = normalize(texts.get(row["native_id"], ""))
        return rows
    train, test = mer_rows("train"), mer_rows("test")
    by_id = {r["utt_id"]: r for r in train + test}
    sets = {"MER_test": test}
    for split in ("valid", "test"):
        sets[f"SIMS_{split}"] = [{"utt_id": f"sims_{r['video_id']}_{r['clip_id']}",
            "normalized": normalize(r["text"]), "video_path": str(args.sims_media / r["video_id"] / (r["clip_id"] + ".mp4"))}
            for r in sims[sims['mode'] == split].to_dict("records")]
        by_id.update({r["utt_id"]: r for r in sets[f"SIMS_{split}"]})
    duration_cache = {}
    def duration(row):
        if "wav" in row:
            return row["wav"]["duration_s"]
        if row["utt_id"] not in duration_cache:
            duration_cache[row["utt_id"]] = probe(Path(row["video_path"]))
        return duration_cache[row["utt_id"]]["duration_s"]
    reports = {}
    excluded = set()
    confirmed_ids = set()
    fingerprint_cache = {}
    fpcalc = shutil.which("fpcalc")
    def fingerprint(row):
        identifier = row["utt_id"]
        if identifier not in fingerprint_cache:
            path = Path(row.get("audio_path", row["video_path"]))
            rec = run_with_hard_timeout([fpcalc, "-json", "-raw", "-length", "0", str(path)], 120)
            try:
                data = json.loads(rec["stdout"])
                fingerprint_cache[identifier] = data["fingerprint"] if data.get("fingerprint") else None
            except (ValueError, KeyError):
                fingerprint_cache[identifier] = None
        return fingerprint_cache[identifier]
    for name, targets in sets.items():
        pairs = candidates(train, targets, duration)
        unique = {p["source_id"] for p in pairs}
        reports[name] = {"candidate_pairs": len(pairs), "candidate_mer_ids": len(unique),
            "excluded_mer_ids": len(unique), "confirmed_pairs": None, "confirmed_mer_ids": None,
            "pairs": pairs}
        if name == "SIMS_test" and len(unique)/len(train) > .05:
            write_json_atomic(args.output / "dedup_stop.json", {"reason": "CH-SIMS test text candidates exceed 5%", "n": len(unique), "denominator": len(train)})
            raise RuntimeError("STOP CH-SIMS test candidate threshold")
        excluded.update(unique)
        if fpcalc:
            confirmed = set(); count = 0
            for pair in pairs:
                a, b = fingerprint(by_id[pair["source_id"]]), fingerprint(by_id[pair["target_id"]])
                pair["audio_confirmed"] = a is not None and b is not None and a == b
                pair["audio_available"] = a is not None and b is not None
                if pair["audio_confirmed"]:
                    confirmed.add(pair["source_id"]); count += 1
            reports[name].update(confirmed_pairs=count, confirmed_mer_ids=len(confirmed))
            confirmed_ids.update(confirmed)
    # Never write subtitles/normalized strings into outputs.
    for row in train + test:
        row.pop("normalized")
    output = {"directions": reports, "mer_train_before": len(train), "mer_train_after": len(train)-len(excluded),
        "excluded_ids": sorted(excluded), "audio_confirmed_ids": sorted(confirmed_ids),
        "fingerprint_available": bool(fpcalc), "fpcalc_path": fpcalc,
        "short_duration_probe_failures": sum(r["status"] != "ok" for r in duration_cache.values()),
        "missing_subtitle_count": sum(not texts.get(r["native_id"], "") for r in train)}
    write_rows(args.derived / "manifests/mer_train_dedup.jsonl", [r for r in train if r["utt_id"] not in excluded])
    write_rows(args.derived / "manifests/mer_train_audioonly.jsonl", [r for r in train if r["utt_id"] not in confirmed_ids])
    write_json_atomic(args.output / "dedup_report.json", output)
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), report_sha256=sha256(args.output / "dedup_report.json"))
    write_json_atomic(args.output / "run_meta.json", meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mer", type=Path, default=Path("/mnt/h/emotion-data/MER2026"))
    p.add_argument("--sims-meta", type=Path, default=Path("/home/administrator/emotion-data/emotionAI_dataset/CH-SIMSV2/meta.csv"))
    p.add_argument("--sims-media", type=Path, required=True)
    p.add_argument("--derived", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
