from __future__ import annotations

import collections
import math
import random
from typing import Callable

LABELS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise")


def classification(rows: list[dict], truth_key: str = "truth", pred_key: str = "pred") -> dict:
    confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    invalid = collections.Counter()
    support = collections.Counter()
    correct = 0
    valid = 0
    for row in rows:
        truth, pred = row[truth_key], row.get(pred_key)
        if truth not in LABELS:
            raise ValueError(f"Unknown truth label: {truth}")
        support[truth] += 1
        if pred in LABELS:
            confusion[truth][pred] += 1
            valid += 1
            correct += pred == truth
        else:
            invalid[truth] += 1
    per_class = {}
    for label in LABELS:
        tp = confusion[label][label]
        predicted = sum(confusion[truth][label] for truth in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support[label], "precision": precision,
                            "recall": recall, "f1": f1}
    n = len(rows)
    return {"n": n, "valid": valid, "failures": n - valid,
            "coverage": valid / n if n else None,
            "macro_f1": sum(x["f1"] for x in per_class.values()) / len(LABELS),
            "uar": sum(x["recall"] for x in per_class.values()) / len(LABELS),
            "wa": correct / n if n else None,
            "neutral_false_positive_rate":
                sum(confusion["neutral"][label] for label in LABELS if label != "neutral") /
                support["neutral"] if support["neutral"] else None,
            "per_class": per_class, "confusion": confusion,
            "invalid_by_true": {label: invalid[label] for label in LABELS}}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower) if upper != lower else ordered[lower]


def bootstrap(rows: list[dict], metric: Callable[[list[dict]], float],
              repetitions: int = 2000, seed: int = 20260928,
              group_key: str | None = None) -> dict:
    rng = random.Random(seed)
    if group_key:
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row[group_key]].append(row)
        keys = sorted(groups)
        samples = ([item for key in rng.choices(keys, k=len(keys)) for item in groups[key]]
                   for _ in range(repetitions))
    else:
        samples = (rng.choices(rows, k=len(rows)) for _ in range(repetitions))
    values = [metric(sample) for sample in samples]
    return {"replicates": repetitions, "seed": seed, "unit": group_key or "utterance",
            "lower_95": percentile(values, 0.025), "upper_95": percentile(values, 0.975)}


def paired_bootstrap(left: list[dict], right: list[dict], metric: str = "macro_f1",
                     repetitions: int = 2000, seed: int = 20260928) -> dict:
    if [row["id"] for row in left] != [row["id"] for row in right]:
        raise ValueError("Paired comparison requires identical ordered sample IDs")
    rng = random.Random(seed)
    n = len(left)
    values = []
    for _ in range(repetitions):
        indices = [rng.randrange(n) for _ in range(n)]
        a = classification([left[i] for i in indices])[metric]
        b = classification([right[i] for i in indices])[metric]
        values.append(a - b)
    return {"n": n, "metric": metric,
            "difference": classification(left)[metric] - classification(right)[metric],
            "left_minus_right_95": [percentile(values, 0.025), percentile(values, 0.975)],
            "replicates": repetitions, "seed": seed}


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    rank = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for k in range(i, j):
            rank[ordered[k]] = (i + j + 1) / 2
        i = j
    return rank


def sims_metrics(rows: list[dict]) -> dict:
    paired = [(row["s"], row["label"]) for row in rows if row.get("s") is not None]
    x, y = [p[0] for p in paired], [p[1] for p in paired]
    nonzero = [row for row in rows if row["label"] != 0]
    correct = sum(row.get("s") is not None and (row["s"] > 0) == (row["label"] > 0)
                  and row["s"] != 0 for row in nonzero)
    return {"n": len(rows), "valid": len(paired), "coverage": len(paired) / len(rows) if rows else None,
            "pearson": pearson(x, y), "spearman": pearson(ranks(x), ranks(y)),
            "acc_2_nonzero": correct / len(nonzero) if nonzero else None,
            "nonzero_denominator": len(nonzero), "nonzero_correct": correct,
            "correlation_denominator": len(paired)}
