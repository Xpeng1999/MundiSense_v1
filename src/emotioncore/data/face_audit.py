from __future__ import annotations

import argparse
import collections
import math
import random
from pathlib import Path

from emotioncore.utils.common import load_paths, read_jsonl, write_json_atomic


def sample_clip(video: Path, duration: float, detector) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {"faces": {"0": 0, "1": 0, "2+": 0}, "decode_failures": 1}
    counts = collections.Counter()
    decode_failures = 0
    try:
        for frame_no in range(max(1, math.ceil(duration * 2))):
            capture.set(cv2.CAP_PROP_POS_MSEC, 500.0 * frame_no)
            ok, frame = capture.read()
            if not ok:
                decode_failures += 1
                continue
            height, width = frame.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(frame)
            count = 0 if faces is None else len(faces)
            counts["2+" if count >= 2 else str(count)] += 1
    finally:
        capture.release()
    return {"faces": {key: counts[key] for key in ("0", "1", "2+")},
            "decode_failures": decode_failures}


def summarize(records: list[dict]) -> dict:
    totals = collections.Counter()
    by_speaker = collections.defaultdict(collections.Counter)
    for row in records:
        totals.update(row["faces"])
        if row.get("speaker_id"):
            by_speaker[row["speaker_id"]].update(row["faces"])
    def show(counts):
        n = sum(counts.values())
        return {"frames": n, "counts": {key: counts[key] for key in ("0", "1", "2+")},
                "fractions": {key: counts[key] / n if n else None for key in ("0", "1", "2+")}}
    return {"clips": len(records), "frames": show(totals),
            "by_speaker": {speaker: show(counts) for speaker, counts in sorted(by_speaker.items())},
            "decode_failures": sum(row["decode_failures"] for row in records),
            "per_clip": records}


def run(config: Path, results: Path, output: Path, seed: int = 20260928) -> None:
    import cv2

    paths = load_paths(config)
    detector = cv2.FaceDetectorYN.create(str(paths["yunet_model"]), "", (320, 320),
                                         0.6, 0.3, 50)
    et = list(read_jsonl(results / "et_utterances.jsonl"))
    sims = list(read_jsonl(results / "chsims_v2_clips.jsonl"))
    rng = random.Random(seed)
    et_sample = [row for row in et if row["official_split"] in ("test", "val")]
    et_sample += rng.sample([row for row in et if row["official_split"] == "train"], 2000)
    sims_sample = [row for row in sims if row["mode"] == "valid"]
    sims_sample += rng.sample([row for row in sims if row["mode"] == "train"], 1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = {"emotiontalk": [], "chsims_v2": []}
    for index, row in enumerate(et_sample):
        path = paths["processed_root"] / "emotiontalk" / row["mp4"]["path"]
        measured = sample_clip(path, row["mp4"].get("duration_s") or 0, detector)
        records["emotiontalk"].append({"id": row["utt_id"], "split": row["official_split"],
                                       "speaker_id": row["speaker_id"], **measured})
        if (index + 1) % 100 == 0:
            write_json_atomic(output.with_name("face_progress.json"),
                              {"emotiontalk_done": index + 1, "emotiontalk_total": len(et_sample),
                               "chsims_done": 0, "chsims_total": len(sims_sample)})
    for index, row in enumerate(sims_sample):
        path = paths["derived_root"] / "chsims_v2/raw" / row["media"]["path"]
        measured = sample_clip(path, row["media"].get("duration_s") or 0, detector)
        records["chsims_v2"].append({"id": f"{row['video_id']}__{row['clip_id']}",
                                    "split": row["mode"], **measured})
        if (index + 1) % 100 == 0:
            write_json_atomic(output.with_name("face_progress.json"),
                              {"emotiontalk_done": len(et_sample), "emotiontalk_total": len(et_sample),
                               "chsims_done": index + 1, "chsims_total": len(sims_sample)})
    write_json_atomic(output, {"seed": seed, "sampling_fps": 2, "yunet_score_threshold": 0.6,
                               "emotiontalk": summarize(records["emotiontalk"]),
                               "chsims_v2": summarize(records["chsims_v2"])})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.results, args.output)


if __name__ == "__main__":
    main()
