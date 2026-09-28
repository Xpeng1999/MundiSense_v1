"""Create the deterministic speaker-disjoint X02 outer and inner folds."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path


EXCLUDED_SPEAKERS = ("02", "13", "20", "21")
SEED = 20260928
CANDIDATES = 200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_speakers(manifest: Path) -> tuple[Counter, Counter]:
    counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            speaker = row["speaker_id"]
            counts[speaker] += 1
            if row["official_split"] == "test":
                test_counts[speaker] += 1
    return counts, test_counts


def balanced_groups(counts: Counter, seed: int = SEED,
                    candidates: int = CANDIDATES) -> list[tuple[str, ...]]:
    speakers = tuple(sorted(s for s in counts if s not in EXCLUDED_SPEAKERS))
    if len(speakers) != 15:
        raise ValueError(f"Expected 15 CV speakers, got {len(speakers)}")
    rng = random.Random(seed)
    target = sum(counts[s] for s in speakers) / 5
    best_score = None
    best_groups = None
    for _ in range(candidates):
        shuffled = list(speakers)
        rng.shuffle(shuffled)
        groups = tuple(sorted(tuple(sorted(shuffled[i:i + 3])) for i in range(0, 15, 3)))
        sizes = [sum(counts[s] for s in group) for group in groups]
        score = (max(sizes) - min(sizes),
                 sum((size - target) ** 2 for size in sizes), groups)
        if best_score is None or score < best_score:
            best_score, best_groups = score, groups
    assert best_groups is not None
    return list(best_groups)


def inner_validation(remaining: tuple[str, ...], counts: Counter,
                     seed: int, fold_index: int) -> tuple[str, ...]:
    target = sum(counts[s] for s in remaining) / 4
    options = list(itertools.combinations(remaining, 3))
    def rank(group: tuple[str, ...]):
        keyed = f"{seed}:{fold_index}:{','.join(group)}".encode()
        return (abs(sum(counts[s] for s in group) - target),
                hashlib.sha256(keyed).hexdigest())
    return min(options, key=rank)


def build(manifest: Path, seed: int = SEED) -> dict:
    counts, test_counts = count_speakers(manifest)
    if set(test_counts) != set(EXCLUDED_SPEAKERS):
        raise ValueError(f"Unexpected ET-test speakers: {sorted(test_counts)}")
    groups = balanced_groups(counts, seed)
    pool = tuple(sorted(s for s in counts if s not in EXCLUDED_SPEAKERS))
    folds = []
    for index, outer in enumerate(groups):
        remaining = tuple(s for s in pool if s not in outer)
        inner = inner_validation(remaining, counts, seed, index)
        train = tuple(s for s in remaining if s not in inner)
        folds.append({
            "fold": index,
            "outer_holdout_speakers": list(outer),
            "outer_holdout_count": sum(counts[s] for s in outer),
            "inner_validation_speakers": list(inner),
            "inner_validation_count": sum(counts[s] for s in inner),
            "inner_train_speakers": list(train),
            "inner_train_count": sum(counts[s] for s in train),
        })
    return {
        "version": "1", "seed": seed,
        "algorithm": "200000 seeded speaker shuffles; minimize outer count range then squared deviation; inner triple closest to one quarter of remaining utterances",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "excluded_speakers": list(EXCLUDED_SPEAKERS),
        "et_test_count": sum(test_counts.values()),
        "cv_pool_speakers": list(pool),
        "cv_pool_count": sum(counts[s] for s in pool),
        "speaker_counts": dict(sorted(counts.items())),
        "folds": folds,
    }


def validate_oof(expected_ids: set[str], records: list[dict]) -> None:
    """Require one and only one prediction for every CV utterance."""
    ids = [row["utt_id"] for row in records]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != expected_ids:
        raise ValueError("OOF predictions must cover every CV utterance exactly once")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha-output", type=Path, required=True)
    args = parser.parse_args()
    content = json.dumps(build(args.manifest), ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != content:
        raise SystemExit("Frozen folds file differs; create a versioned protocol revision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    digest = sha256(args.output)
    args.sha_output.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"folds_sha256": digest,
                      "cv_pool_count": json.loads(content)["cv_pool_count"]}, indent=2))


if __name__ == "__main__":
    main()
