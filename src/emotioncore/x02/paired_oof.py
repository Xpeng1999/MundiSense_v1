"""Speaker-clustered paired interval for two frozen OOF probability files."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification, percentile
from emotioncore.x02.head_h1 import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def scored(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        if probabilities.shape != (7,) or not np.isfinite(probabilities).all() or \
           not np.isclose(probabilities.sum(), 1, atol=1e-5):
            raise ValueError("Invalid OOF probabilities")
        result.append({"id": row["utt_id"], "speaker_id": row["speaker_id"],
                       "truth": row["truth_M"],
                       "pred": LABELS[int(probabilities.argmax())] if row["status"] == "ok" else None})
    return result


def cluster_counts(rows: list[dict], speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    confusion = np.zeros((len(speakers), 7, 7), dtype=np.int64)
    supports = np.zeros((len(speakers), 7), dtype=np.int64)
    index = {speaker: k for k, speaker in enumerate(speakers)}
    for row in rows:
        s = index[row["speaker_id"]]
        truth = LABELS.index(row["truth"])
        supports[s, truth] += 1
        if row["pred"] is not None:
            confusion[s, truth, LABELS.index(row["pred"])] += 1
    return confusion, supports


def metrics(confusion: np.ndarray, supports: np.ndarray) -> dict:
    tp = np.diag(confusion)
    predicted = confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(7), where=predicted != 0)
    recall = np.divide(tp, supports, out=np.zeros(7), where=supports != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(7), where=(precision + recall) != 0)
    return {"macro_f1": float(f1.mean()), "uar": float(recall.mean()),
            "wa": float(tp.sum() / supports.sum())}


def compare(left: list[dict], right: list[dict], repetitions: int = 2000,
            seed: int = 20260928) -> dict:
    left, right = scored(left), scored(right)
    if [row["id"] for row in left] != [row["id"] for row in right] or \
       len({row["id"] for row in left}) != len(left):
        raise ValueError("Paired OOF files must have the same unique ordered IDs")
    if [(row["speaker_id"], row["truth"]) for row in left] != \
       [(row["speaker_id"], row["truth"]) for row in right]:
        raise ValueError("Paired OOF files disagree on speaker or truth")
    speakers = sorted({row["speaker_id"] for row in left})
    if len(speakers) != 15:
        raise ValueError("Expected 15 frozen CV speakers")
    left_c, left_s = cluster_counts(left, speakers)
    right_c, right_s = cluster_counts(right, speakers)
    if not np.array_equal(left_s, right_s):
        raise ValueError("Paired OOF support differs")
    baseline = {"left": classification(left), "right": classification(right)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(speakers), size=(repetitions, len(speakers)))
    deltas = {key: [] for key in ("macro_f1", "uar", "wa")}
    for draw in draws:
        a = metrics(left_c[draw].sum(0), left_s[draw].sum(0))
        b = metrics(right_c[draw].sum(0), right_s[draw].sum(0))
        for key in deltas:
            deltas[key].append(a[key] - b[key])
    return {"n": len(left), "speakers": speakers, "left": baseline["left"],
            "right": baseline["right"], "left_minus_right": {
                key: {"difference": baseline["left"][key] - baseline["right"][key],
                      "lower_95": percentile(values, 0.025),
                      "upper_95": percentile(values, 0.975),
                      "replicates": repetitions, "seed": seed, "unit": "speaker"}
                for key, values in deltas.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    provenance = git_provenance(Path.cwd(), args.allow_dirty)
    result = compare(read(args.left), read(args.right))
    result.update({"left_name": args.left_name, "right_name": args.right_name,
                   "left_path": str(args.left), "right_path": str(args.right),
                   "left_sha256": sha256(args.left), "right_sha256": sha256(args.right),
                   "git": provenance, "command": sys.argv,
                   "ended_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    write_json_atomic(args.output, result)


if __name__ == "__main__":
    main()
