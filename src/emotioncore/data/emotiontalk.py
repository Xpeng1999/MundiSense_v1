from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import subprocess
from pathlib import Path

from emotioncore.utils.common import load_paths, sha256_file, write_json_atomic

MODALITIES = {"M": "Multimodal", "A": "Audio", "V": "Video", "T": "Text"}
LABELS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise")
NORMALIZE = {"happy": "joy", "sad": "sadness", "angry": "anger", "disgusted": "disgust",
             "fearful": "fear", "surprised": "surprise", "neutral": "neutral"}
OFFICIAL = {
    "train": [5377, 2986, 919, 2950, 672, 1142, 1367],
    "val": [675, 360, 111, 409, 125, 95, 133],
    "test": [801, 246, 123, 339, 125, 134, 161],
}


def normalize(value: str) -> str:
    lowered = value.strip().lower()
    result = NORMALIZE.get(lowered, lowered)
    if result not in LABELS:
        raise ValueError(f"Unexpected label: {value}")
    return result


def split_for_group(group: str) -> str:
    if group in {"G00001", "G00012"}:
        return "val"
    if group in {"G00003", "G00015"}:
        return "test"
    return "train"


def fleiss_kappa(rows: list[list[str]]) -> float | None:
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(row) != n for row in rows):
        return None
    totals = collections.Counter(label for row in rows for label in row)
    p_e = sum((totals[label] / (len(rows) * n)) ** 2 for label in LABELS)
    if math.isclose(p_e, 1.0):
        return None
    p_bar = sum(sum(c * (c - 1) for c in collections.Counter(row).values()) / (n * (n - 1))
                for row in rows) / len(rows)
    return (p_bar - p_e) / (1.0 - p_e)


def probe(path: Path) -> dict:
    if not path.is_file():
        return {"error": "missing"}
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                             "-of", "json", str(path)], capture_output=True, text=True)
    if result.returncode:
        return {"sha256": sha256_file(path), "error": result.stderr.strip()[:300] or "ffprobe failed"}
    info = json.loads(result.stdout)
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    rate = video.get("avg_frame_rate") or "0/1"
    numerator, denominator = (rate.split("/") + ["1"])[:2]
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    return {"sha256": sha256_file(path), "duration_s": float(info.get("format", {}).get("duration") or 0),
            "fps": fps, "width": video.get("width"), "height": video.get("height"),
            "sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            "channels": audio.get("channels")}


def build_row(json_path: Path, root: Path) -> dict:
    utt_id = json_path.stem
    group, dialogue, speaker, _ = utt_id.split("_")
    labels = {}
    votes = {}
    m_continuous = None
    for short, folder in MODALITIES.items():
        path = root / folder / "json" / json_path.relative_to(root / "Multimodal" / "json")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if str(raw["speaker_id"]).zfill(2) != speaker:
            raise ValueError(f"Speaker mismatch in {path}")
        labels[short] = normalize(raw["emotion_result"])
        votes[short] = {annotator: {"emotion": normalize(v["emotion"]),
                                     "Confidence_degree": v.get("Confidence_degree")}
                        for annotator, v in raw["data"].items()}
        if short == "M":
            m_continuous = raw.get("Continuous label_result")
    rel = json_path.relative_to(root / "Multimodal" / "json")
    mp4 = Path("Multimodal/mp4") / rel.with_suffix(".mp4")
    wav = Path("Audio/wav") / rel.with_suffix(".wav")
    return {"utt_id": utt_id, "group": group, "dialogue": dialogue, "speaker_id": speaker,
            "official_split": split_for_group(group), "labels": labels, "votes": votes,
            "m_continuous_label_result": m_continuous,
            "mp4": {"path": mp4.as_posix()}, "wav": {"path": wav.as_posix()}}


def audit(rows: list[dict]) -> dict:
    counts = {split: collections.Counter() for split in OFFICIAL}
    speakers: dict[str, dict] = {}
    agreement = {mod: collections.Counter() for mod in ("A", "V", "T")}
    by_class = {label: {mod: collections.Counter() for mod in ("A", "V", "T")} for label in LABELS}
    pairwise = {left: {right: collections.Counter() for right in MODALITIES} for left in MODALITIES}
    pairwise_by_m_class = {label: {left: {right: collections.Counter() for right in MODALITIES}
                                   for left in MODALITIES} for label in LABELS}
    majority = {mod: collections.Counter() for mod in MODALITIES}
    kappa_rows = {mod: [] for mod in MODALITIES}
    for row in rows:
        truth = row["labels"]["M"]
        counts[row["official_split"]][truth] += 1
        entry = speakers.setdefault(row["speaker_id"], {"groups": set(), "splits": set(), "count": 0})
        entry["groups"].add(row["group"])
        entry["splits"].add(row["official_split"])
        entry["count"] += 1
        for mod in ("A", "V", "T"):
            same = row["labels"][mod] == truth
            agreement[mod]["total"] += 1
            agreement[mod]["same"] += same
            by_class[truth][mod]["total"] += 1
            by_class[truth][mod]["same"] += same
        for left in MODALITIES:
            for right in MODALITIES:
                same = row["labels"][left] == row["labels"][right]
                pairwise[left][right]["total"] += 1
                pairwise[left][right]["same"] += same
                pairwise_by_m_class[truth][left][right]["total"] += 1
                pairwise_by_m_class[truth][left][right]["same"] += same
        for mod in MODALITIES:
            chosen = [v["emotion"] for v in row["votes"][mod].values()]
            kappa_rows[mod].append(chosen)
            majority[mod][max(collections.Counter(chosen).values())] += 1
    observed = {split: {label: counts[split][label] for label in LABELS} for split in OFFICIAL}
    differences = [{"split": split, "label": label, "observed": counts[split][label], "official": expected}
                   for split in OFFICIAL for label, expected in zip(LABELS, OFFICIAL[split])
                   if counts[split][label] != expected]
    return {"rows": len(rows), "counts": observed, "official_differences": differences,
            "speakers": {key: {"groups": sorted(v["groups"]), "splits": sorted(v["splits"]),
                               "count": v["count"]} for key, v in sorted(speakers.items())},
            "m_vs_other": {mod: {"same": c["same"], "total": c["total"], "rate": c["same"] / c["total"]}
                           for mod, c in agreement.items()},
            "m_vs_other_by_class": {label: {mod: {"same": c["same"], "total": c["total"],
                                                  "rate": c["same"] / c["total"]}
                                           for mod, c in mods.items()} for label, mods in by_class.items()},
            "pairwise_agreement": {left: {right: {"same": c["same"], "total": c["total"],
                                                   "rate": c["same"] / c["total"]}
                                         for right, c in rights.items()} for left, rights in pairwise.items()},
            "pairwise_by_m_class": {label: {left: {right: {"same": c["same"], "total": c["total"],
                                                              "rate": c["same"] / c["total"]}
                                                    for right, c in rights.items()}
                                             for left, rights in lefts.items()}
                                    for label, lefts in pairwise_by_m_class.items()},
            "fleiss_kappa": {mod: fleiss_kappa(v) for mod, v in kappa_rows.items()},
            "majority_strength": {mod: dict(sorted(c.items())) for mod, c in majority.items()}}


def build(config: Path, output: Path, workers: int) -> None:
    paths = load_paths(config)
    root = paths["processed_root"] / "emotiontalk"
    json_paths = sorted((root / "Multimodal/json").rglob("*.json"))
    rows = [build_row(path, root) for path in json_paths]
    if len(rows) != len({row["utt_id"] for row in rows}):
        raise ValueError("Duplicate EmotionTalk utterance ID")
    if len(rows) != 19250:
        raise ValueError(f"Expected 19250 EmotionTalk rows; found {len(rows)}")
    media_paths = sorted({row[k]["path"] for row in rows for k in ("mp4", "wav")})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        metadata = dict(zip(media_paths, pool.map(lambda rel: probe(root / rel), media_paths)))
    for row in rows:
        for key in ("mp4", "wav"):
            row[key].update(metadata[row[key]["path"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json_atomic(output.with_name("et_audit.json"), audit(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.config, args.output, args.workers)


if __name__ == "__main__":
    main()
