"""Frozen-fold probability fusion, calibration, and neutral operating point."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from emotioncore.eval.metrics import LABELS, classification, sims_metrics
from emotioncore.x02.asr_cer import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA


COMBINATIONS = {
    "E1_A3": ("A3_h2",),
    "E1_A3_V2": ("A3_h2", "V2_h2"),
    "E1_A3_V2_V1": ("A3_h2", "V2_h2", "V1_h1"),
    "E1_A3_V2_T1r": ("A3_h2", "V2_h2", "T1_r"),
    "E1_A3_A2_V2_V1_T1r": ("A3_h2", "A2_h2", "V2_h2", "V1_h1", "T1_r"),
}


def softmax(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


def nll(logits: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))


def fit_fusion(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit nonnegative log-probability weights and centered class biases."""
    modes = logp.shape[1]

    def objective(theta: np.ndarray) -> float:
        bias = theta[modes:] - np.mean(theta[modes:])
        logits = np.einsum("nmc,m->nc", logp, theta[:modes]) + bias
        return nll(logits, y) + 1e-6 * float(np.sum(theta ** 2))

    initial = np.r_[np.full(modes, 1 / modes), np.zeros(7)]
    result = minimize(objective, initial, method="L-BFGS-B",
                      bounds=[(0, None)] * modes + [(None, None)] * 7,
                      options={"maxiter": 400, "ftol": 1e-10})
    if not result.success and np.linalg.norm(result.jac, np.inf) > 1e-4:
        raise RuntimeError(f"Fusion optimization failed: {result.message}")
    return result.x


def fusion_logits(logp: np.ndarray, theta: np.ndarray) -> np.ndarray:
    modes = logp.shape[1]
    return np.einsum("nmc,m->nc", logp, theta[:modes]) + theta[modes:] - np.mean(theta[modes:])


def fit_calibration(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit log-temperature and seven centered class biases on fitting rows only."""
    def objective(theta: np.ndarray) -> float:
        temperature = np.exp(theta[0])
        bias = theta[1:] - np.mean(theta[1:])
        return nll(logits / temperature + bias, y) + 1e-6 * float(np.sum(theta ** 2))

    result = minimize(objective, np.zeros(8), method="L-BFGS-B",
                      bounds=[(-4, 4)] + [(None, None)] * 7,
                      options={"maxiter": 400, "ftol": 1e-10})
    if not result.success and np.linalg.norm(result.jac, np.inf) > 1e-4:
        raise RuntimeError(f"Calibration optimization failed: {result.message}")
    return result.x


def calibrated_logits(logits: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return logits / np.exp(theta[0]) + theta[1:] - np.mean(theta[1:])


def fit_neutral_bias(logits: np.ndarray, y: np.ndarray, target: float = 0.15) -> float:
    """Find a fixed neutral boost giving the closest empirical neutral miss rate."""
    neutral = logits[y == 0]
    if not len(neutral):
        raise ValueError("Fitting rows contain no neutral examples")
    margin = np.max(neutral[:, 1:], axis=1) - neutral[:, 0]
    candidates = np.r_[np.nextafter(margin.min(), -np.inf),
                       np.unique(margin), np.nextafter(margin.max(), np.inf)]
    # Ties are resolved toward the smaller miss rate, then the smaller bias.
    ordered = np.sort(margin)
    miss = (len(margin) - np.searchsorted(ordered, candidates, side="right")) / len(margin)
    best = min(range(len(candidates)), key=lambda i: (abs(miss[i] - target), miss[i], candidates[i]))
    return float(candidates[best])


def ece15(probabilities: np.ndarray, y: np.ndarray) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = np.argmax(probabilities, axis=1) == y
    bins = np.minimum((confidence * 15).astype(int), 14)
    return float(sum(np.mean(bins == k) *
                     abs(float(np.mean(correct[bins == k])) - float(np.mean(confidence[bins == k])))
                     for k in range(15) if np.any(bins == k)))


def metrics(probabilities: np.ndarray, y: np.ndarray, speakers: list[str],
            ids: list[str], neutral_bias: float = 0.0) -> dict:
    scores = np.log(np.clip(probabilities, 1e-12, 1))
    scores[:, 0] += neutral_bias
    pred = np.argmax(scores, axis=1)
    rows = [{"id": uid, "truth": LABELS[int(truth)], "pred": LABELS[int(guess)],
             "speaker_id": speaker}
            for uid, truth, guess, speaker in zip(ids, y, pred, speakers)]
    result = classification(rows)
    result["nonneutral_macro_recall"] = float(np.mean(
        [result["per_class"][label]["recall"] for label in LABELS[1:]]))
    result["ece15"] = ece15(probabilities, y)
    return result


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def index_rows(rows: list[dict], id_key: str) -> dict[str, dict]:
    indexed = {row[id_key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return indexed


def apply_face_missing(probabilities: np.ndarray, ids: list[str],
                       components: tuple[str, ...], face_missing: set[str]) -> np.ndarray:
    result = probabilities.copy()
    missing = np.array([uid in face_missing for uid in ids])
    for index, component in enumerate(components):
        if component.startswith("V"):
            result[missing, index, :] = 1 / 7
    return result


def load_sources(index_path: Path, components: tuple[str, ...], split: str,
                 face_missing: set[str] | None = None) -> tuple[list[str], np.ndarray, list[dict], dict]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    maps = []
    provenance = {}
    for component in components:
        item = index["predictions"]["T1"]["r"] if component == "T1_r" else index["predictions"][component]
        if split not in item:
            raise FileNotFoundError(f"{component} has no frozen {split} predictions")
        spec = item[split]
        path = Path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"X02 artifact SHA mismatch: {path}")
        rows = read_rows(path)
        maps.append(index_rows(rows, "id" if split == "sims_valid" else "utt_id"))
        provenance[component] = spec
    ids = list(maps[0])
    if any(set(source) != set(ids) for source in maps[1:]):
        raise ValueError("Modal prediction ID coverage differs")
    records = [maps[0][uid] for uid in ids]
    if split != "sims_valid":
        for source in maps[1:]:
            if any((source[uid]["speaker_id"], source[uid]["truth_M"], source[uid].get("fold")) !=
                   (maps[0][uid]["speaker_id"], maps[0][uid]["truth_M"], maps[0][uid].get("fold"))
                   for uid in ids):
                raise ValueError("Modal predictions disagree on frozen labels or folds")
    else:
        for source in maps[1:]:
            if any(source[uid]["label"] != maps[0][uid]["label"] for uid in ids):
                raise ValueError("SIMS labels disagree between sources")
    probabilities = np.stack([[source[uid]["probabilities"] for uid in ids] for source in maps], axis=1)
    if face_missing:
        probabilities = apply_face_missing(probabilities, ids, components, face_missing)
    if (probabilities.shape != (len(ids), len(components), 7) or
            not np.isfinite(probabilities).all() or
            not np.allclose(probabilities.sum(axis=2), 1, atol=1e-5)):
        raise ValueError("Invalid cached probability tensor")
    return ids, np.log(np.clip(probabilities, 1e-12, 1)), records, provenance


def crossfit(logp: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if sorted(set(folds)) != list(range(5)):
        raise ValueError("Expected five frozen folds")
    op_a = np.empty((len(y), 7))
    op_n = np.empty((len(y), 7))
    parameters = []
    for fold in range(5):
        fit = folds != fold
        held = folds == fold
        if not fit.any() or not held.any():
            raise ValueError("Empty fitting or held-out fold")
        fusion = fit_fusion(logp[fit], y[fit])
        calibration = fit_calibration(fusion_logits(logp[fit], fusion), y[fit])
        fit_scores = calibrated_logits(fusion_logits(logp[fit], fusion), calibration)
        bias = fit_neutral_bias(fit_scores, y[fit])
        held_scores = calibrated_logits(fusion_logits(logp[held], fusion), calibration)
        op_a[held] = softmax(held_scores)
        held_scores[:, 0] += bias
        op_n[held] = softmax(held_scores)
        parameters.append({"fold": fold, "fit_count": int(fit.sum()), "held_count": int(held.sum()),
                           "fusion": fusion.tolist(), "calibration": calibration.tolist(),
                           "neutral_bias": bias})
    return op_a, op_n, parameters


def apply_final(logp_oof: np.ndarray, y: np.ndarray, logp_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    fusion = fit_fusion(logp_oof, y)
    calibration = fit_calibration(fusion_logits(logp_oof, fusion), y)
    fit_scores = calibrated_logits(fusion_logits(logp_oof, fusion), calibration)
    bias = fit_neutral_bias(fit_scores, y)
    scores = calibrated_logits(fusion_logits(logp_target, fusion), calibration)
    op_a = softmax(scores)
    scores[:, 0] += bias
    return op_a, softmax(scores), {"fusion": fusion.tolist(), "calibration": calibration.tolist(),
                                  "neutral_bias": bias}


def write_predictions(path: Path, ids: list[str], records: list[dict],
                      op_a: np.ndarray, op_n: np.ndarray, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for uid, row, a, n in zip(ids, records, op_a, op_n):
            item = {"id": uid, "split": split, "op_a": a.tolist(), "op_n": n.tolist()}
            for key in ("speaker_id", "fold", "truth_M", "label"):
                if key in row:
                    item[key] = row[key]
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.face_policy) != FACE_SHA:
        raise ValueError("Frozen face-missing policy SHA mismatch")
    face_missing = set(json.loads(args.face_policy.read_text(encoding="utf-8"))["face_missing_utt_ids"])
    folds_path = Path("configs/folds_v1.json")
    expected_fold_sha = "85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce"
    if sha256(folds_path) != expected_fold_sha:
        raise ValueError("Frozen fold SHA mismatch")
    manifest = Path("experiments/X01_benchmark/results/et_utterances.jsonl")
    frozen_rows = read_rows(manifest)
    cv_ids = {r["utt_id"] for r in frozen_rows
              if r["speaker_id"] in json.loads(folds_path.read_text(encoding="utf-8"))["cv_pool_speakers"]}
    test_ids_expected = {r["utt_id"] for r in frozen_rows if r["official_split"] == "test"}
    if len(cv_ids) != 16_366 or len(test_ids_expected) != 1_929:
        raise ValueError("Frozen ET manifest coverage mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {"git": git, "command": sys.argv,
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "fold_sha256": expected_fold_sha, "artifact_index_sha256": sha256(args.artifact_index),
                "face_policy_sha256": FACE_SHA}
    write_json_atomic(args.output_dir / "run_meta.json", run_meta)
    summary = {}
    for name, components in COMBINATIONS.items():
        ids, logp, records, sources = load_sources(args.artifact_index, components, "oof", face_missing)
        if set(ids) != cv_ids or len(ids) != 16_366:
            raise ValueError("Incomplete CV OOF coverage")
        y = np.array([LABELS.index(row["truth_M"]) for row in records])
        folds = np.array([row["fold"] for row in records])
        speakers = [row["speaker_id"] for row in records]
        frozen = json.loads(folds_path.read_text(encoding="utf-8"))
        expected = {s: f["fold"] for f in frozen["folds"] for s in f["outer_holdout_speakers"]}
        if any(expected[s] != int(f) for s, f in zip(speakers, folds)):
            raise ValueError("OOF fold/speaker leakage")
        op_a, op_n, params = crossfit(logp, y, folds)
        target_dir = args.output_dir / name
        write_predictions(target_dir / "oof.jsonl", ids, records, op_a, op_n, "oof")
        cv = {"op_a": metrics(op_a, y, speakers, ids),
              "op_n": metrics(op_n, y, speakers, ids)}
        write_json_atomic(target_dir / "parameters.json", {"components": components, "folds": params})
        test_ids, test_logp, test_records, test_sources = load_sources(args.artifact_index, components, "et_test", face_missing)
        if set(test_ids) != test_ids_expected or len(test_ids) != 1_929:
            raise ValueError("Incomplete ET-test coverage")
        test_a, test_n, final = apply_final(logp, y, test_logp)
        write_predictions(target_dir / "et_test.jsonl", test_ids, test_records, test_a, test_n, "et_test")
        test_y = np.array([LABELS.index(row["truth_M"]) for row in test_records])
        test = {"op_a": metrics(test_a, test_y, [r["speaker_id"] for r in test_records], test_ids),
                "op_n": metrics(test_n, test_y, [r["speaker_id"] for r in test_records], test_ids)}
        sims = None
        sims_sources = None
        try:
            sims_ids, sims_logp, sims_records, sims_sources = load_sources(args.artifact_index, components, "sims_valid", face_missing)
        except FileNotFoundError:
            pass
        else:
            if len(sims_ids) != 647:
                raise ValueError("Incomplete SIMS-valid coverage")
            sims_a, sims_n, _ = apply_final(logp, y, sims_logp)
            write_predictions(target_dir / "sims_valid.jsonl", sims_ids, sims_records, sims_a, sims_n, "sims_valid")
            sims = sims_metrics([{"id": uid, "label": row["label"],
                                  "s": float(p[1] - p[2:6].sum())}
                                 for uid, row, p in zip(sims_ids, sims_records, sims_n)])
        write_json_atomic(target_dir / "parameters_final.json", final)
        result = {"components": components, "cv": cv, "et_test": test, "sims_valid": sims,
                  "sources": {"oof": sources, "et_test": test_sources, "sims_valid": sims_sources},
                  "artifacts": {split: {"path": str(target_dir / f"{split}.jsonl"),
                                       "sha256": sha256(target_dir / f"{split}.jsonl")}
                                for split in ("oof", "et_test", "sims_valid")
                                if (target_dir / f"{split}.jsonl").is_file()}}
        write_json_atomic(target_dir / "metrics.json", result)
        summary[name] = {"cv_op_n_macro_f1": cv["op_n"]["macro_f1"],
                         "cv_op_n_neutral_miss_rate": cv["op_n"]["neutral_false_positive_rate"],
                         "sims_valid_available": sims is not None}
    best = max(summary, key=lambda name: summary[name]["cv_op_n_macro_f1"])
    write_json_atomic(args.output_dir / "summary.json", {"systems": summary, "best_e1": best})
    run_meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    run_meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", run_meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
