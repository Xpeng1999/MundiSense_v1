"""X04a report-only evidence, using X03's unchanged crossfit estimators."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, classification, sims_metrics
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03 import fusion, q_calibrate
from emotioncore.x03.complete_analysis import statistics
from emotioncore.x03.h_fusion import q_rows
from emotioncore.x03.q_manifest import read_jsonl, sha256
from emotioncore.x04.heads import CORPUS_CLASSES, FOLDS_SHA, LABELS8
from emotioncore.x04.prepare import require_h, storage_gate, write_rows

SEED = 20261006
REPETITIONS = 2000
SYSTEMS = {"A3": ("A3",), "V2": ("V2",), "V1": ("V1",),
           "H_ref": ("Q_A", "Q_AV", "A3", "V2", "V1"),
           "H3": ("Q_AV", "A3", "V2"), "E3": ("A3", "V2", "V1")}


def renormalize(p, classes):
    p = np.asarray(p, dtype=float)[:, classes]
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(1) <= 0).any():
        raise ValueError("Invalid probabilities in report input")
    return p / p.sum(1, keepdims=True)


def extend_mer(raw8, parameters, *, single=False, unsupported_worried=False):
    """Apply ET-fitted scalar weights/T; centered ET biases plus zero worried bias.

    Conditional seven-class results equal X03 application exactly. The eighth
    class has no ET calibration evidence; it never fits a target-set label.
    """
    logp = np.log(np.clip(raw8, 1e-12, 1))
    if single:
        scores = logp[:, 0]
    else:
        theta = np.asarray(parameters["fusion"])
        modes = raw8.shape[1]
        bias = theta[modes:] - theta[modes:].mean()
        scores = np.einsum("nmc,m->nc", logp, theta[:modes]) + np.r_[bias, 0.]
    calibration = np.asarray(parameters["calibration"])
    bias = calibration[1:] - calibration[1:].mean()
    scores = scores / np.exp(calibration[0]) + np.r_[bias, 0.]
    a = fusion.softmax(scores)
    scores[:, 0] += parameters["neutral_bias"]
    n = fusion.softmax(scores)
    if unsupported_worried:
        for p in (a, n):
            p[:, 7] = 0
            p /= p.sum(1, keepdims=True)
    return a, n


def native_metrics(probabilities8, records, invalid, *, shared=False):
    classes = tuple(k for k in CORPUS_CLASSES["MER"] if not shared or k != 7)
    include = np.array([not shared or r["truth_M"] != "worried" for r in records])
    p = renormalize(probabilities8[include], classes)
    truth = np.array([classes.index(LABELS8.index(r["truth_M"]))
                      for r, keep in zip(records, include) if keep])
    prediction = p.argmax(1)
    bad = invalid[include]
    matrix = np.zeros((len(classes), len(classes)+1), dtype=int)
    np.add.at(matrix, (truth, np.where(bad, len(classes), prediction)), 1)
    support = matrix.sum(1)
    predicted = matrix[:, :-1].sum(0)
    tp = np.diag(matrix[:, :-1])
    precision = np.divide(tp, predicted, out=np.zeros(len(classes)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support > 0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(len(classes)), where=precision+recall > 0)
    return {"n": len(truth), "classes": [LABELS8[i] for i in classes],
            "valid": int((~bad).sum()), "failures": int(bad.sum()),
            "macro_f1": float(f1.mean()), "uar": float(recall.mean()),
            "wa": float(tp.sum()/len(truth)), "confusion_with_invalid_column": matrix.tolist(),
            "per_class": {LABELS8[k]: {"support": int(support[i]),
                "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                for i, k in enumerate(classes)},
            "shared_definition": "remove worried truth and renormalize the five shared native classes" if shared else None}


def scored(probabilities, records, invalid):
    return [{"id": r["utt_id"], "speaker_id": r["speaker_id"], "truth": r["truth_M"],
             "pred": None if bad else LABELS[int(p.argmax())], "probabilities": p.tolist()}
            for r, p, bad in zip(records, probabilities, invalid, strict=True)]


def et_metrics(p, records, invalid):
    result = classification(scored(p, records, invalid))
    y = np.array([LABELS.index(r["truth_M"]) for r in records])
    result["ece15"] = q_calibrate.ece15_with_invalid(p, y, invalid)
    result["nonneutral_macro_recall"] = float(np.mean([result["per_class"][k]["recall"] for k in LABELS[1:]]))
    return result


def paired_speakers(left, right, repetitions=REPETITIONS, seed=SEED):
    if [(r["id"], r["truth"], r["speaker_id"]) for r in left] != [(r["id"], r["truth"], r["speaker_id"]) for r in right]:
        raise ValueError("Paired comparison differs in IDs, labels or speakers")
    speakers = sorted({r["speaker_id"] for r in left})
    location = {s: i for i, s in enumerate(speakers)}
    matrices = []
    for rows in (left, right):
        c = np.zeros((len(speakers), 7, 8), dtype=float)
        for r in rows:
            c[location[r["speaker_id"]], LABELS.index(r["truth"]),
              LABELS.index(r["pred"]) if r["pred"] in LABELS else 7] += 1
        matrices.append(c)
    rng = random.Random(seed)
    weights = np.zeros((repetitions, len(speakers)))
    for rep in range(repetitions):
        for s in rng.choices(speakers, k=len(speakers)):
            weights[rep, location[s]] += 1
    series = [statistics(np.einsum("rg,gij->rij", weights, c),
                         np.zeros((repetitions, 15, 3)))[0]["macro_f1"] for c in matrices]
    return {"n": len(left), "unit": "speaker_id", "speaker_count": len(speakers),
            "seed": seed, "replicates": repetitions,
            "difference": classification(left)["macro_f1"] - classification(right)["macro_f1"],
            "left_minus_right_95": np.percentile(series[0]-series[1], [2.5, 97.5]).tolist()}


def checked(path, digest, artifacts):
    if sha256(path) != digest:
        raise ValueError(f"Artifact changed: {path}")
    artifacts[str(path)] = {"path": str(path), "sha256": digest}
    return read_jsonl(path)


def source(args, arm, component, split, artifacts):
    if component.startswith("Q_"):
        form = component[2:]
        if split != "mer_test":
            rows, specs = q_rows(args.r3/"Q", form, split)
            artifacts.update({s["path"]: s for s in specs})
            return rows
        specs = json.loads((args.derived/"Q_predictions/artifacts.json").read_text())
        spec = specs[form]
        return checked(Path(spec["path"]), spec["sha256"], artifacts)
    job = args.derived/f"heads/{arm}/{component}"
    if split == "oof":
        info = json.loads((job/"results.json").read_text())
        return checked(job/"oof.jsonl", info["oof_sha256"], artifacts)
    spec = json.loads((job/"targets/artifacts.json").read_text())[split]
    return checked(Path(spec["path"]), spec["sha256"], artifacts)


def stack(args, arm, components, split, artifacts, face_missing):
    maps = []
    for component in components:
        rows = source(args, arm, component, split, artifacts)
        mapping = {r["utt_id"]: r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate prediction IDs")
        maps.append(mapping)
    ids = sorted(maps[0])
    if len(ids) != {"oof": 16366, "et_test": 1929, "sims_valid": 647, "mer_test": 572}[split] or any(set(m) != set(ids) for m in maps):
        raise ValueError("STOP incomplete target/OOF coverage")
    records = [maps[0][uid] for uid in ids]
    raw, invalid = [], []
    for component, mapping in zip(components, maps, strict=True):
        ordered = [mapping[uid] for uid in ids]
        for a, b in zip(records, ordered, strict=True):
            keys = ("label",) if split == "sims_valid" else ("truth_M",)
            if split == "oof": keys += ("speaker_id", "fold")
            if any(a[k] != b[k] for k in keys):
                raise ValueError("Component label/speaker/fold mismatch")
        p = np.array([r.get("probabilities8", r["probabilities"]+[0.]) for r in ordered], dtype=float)
        p = renormalize(p, tuple(range(8)))
        bad = np.array([r.get("effective_form") == "uniform" or r.get("status") in ("failed", "missing_media", "no_face") for r in ordered])
        if component.startswith("V"):
            bad |= np.array([uid in face_missing for uid in ids])
            p[bad] = 1/8
        if arm == "D0" or component.startswith("Q_"):
            p[:, 7] = 0
            p /= p.sum(1, keepdims=True)
        raw.append(p); invalid.append(bad)
    # Standalone failed modalities are errors. Fusion retains valid experts.
    bad = np.logical_and.reduce(invalid)
    if any(c.startswith("Q_") for c in components):
        # Match X03 H's missing-media handling (face fallback is still valid).
        bad |= np.array([r.get("status") == "missing_media" for r in records])
    return records, np.stack(raw, axis=1), bad


def run(args):
    git = git_provenance(Path.cwd()); require_h(args.output); storage_gate()
    folds_path = Path("configs/folds_v1.json")
    if sha256(folds_path) != FOLDS_SHA: raise ValueError("Frozen folds changed")
    frozen = json.loads(folds_path.read_text())
    et = read_jsonl(Path("experiments/X01_benchmark/results/et_utterances.jsonl"))
    cv = {r["utt_id"]: r for r in et if r["speaker_id"] in frozen["cv_pool_speakers"]}
    test = {r["utt_id"] for r in et if r["official_split"] == "test"}
    expected = {s: f["fold"] for f in frozen["folds"] for s in f["outer_holdout_speakers"]}
    face_path = Path("configs/x03_face_missing_policy.json")
    face_missing = set(json.loads(face_path.read_text())["face_missing_utt_ids"])
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "Q_training": False, "target_labels_used_for_fitting": False, "folds_sha256": FOLDS_SHA,
            "face_policy_sha256": sha256(face_path), "bootstrap_seed": SEED, "bootstrap_repetitions": REPETITIONS}
    write_json_atomic(args.output/"run_meta.json", meta)
    artifacts = {}; table = {}; paired_rows = {}; reproduction = {}; selections = {}
    planned = [(arm, name, components) for arm in ("D0", "D1", "D2", "D3") for name, components in SYSTEMS.items()]
    planned += [("Q", "Q_A", ("Q_A",)), ("Q", "Q_AV", ("Q_AV",))]
    if (args.derived/"heads/D1_audioonly/A3/results.json").exists():
        planned += [("D1_audioonly", "A3", ("A3",))]
    for arm, name, components in planned:
        storage_gate()
        records, raw, bad = stack(args, arm, components, "oof", artifacts, face_missing)
        if {r["utt_id"] for r in records} != set(cv) or any(r["speaker_id"] != cv[r["utt_id"]]["speaker_id"] or r["truth_M"] != cv[r["utt_id"]]["labels"]["M"] or r["fold"] != expected[r["speaker_id"]] for r in records):
            raise ValueError("STOP CV speaker/label/coverage leakage")
        y = np.array([LABELS.index(r["truth_M"]) for r in records]); folds = np.array([r["fold"] for r in records])
        p7 = raw[..., :7]/raw[..., :7].sum(2, keepdims=True)
        logp = np.log(np.clip(p7, 1e-12, 1))
        single = len(components) == 1
        if single:
            a, n, params = q_calibrate.crossfit_single(logp[:, 0], y, folds, bad)
            calibration, boost = q_calibrate.fit_single(logp[:, 0], y, bad)
            final = {"calibration": calibration.tolist(), "neutral_bias": boost}
        else:
            a, n, params = fusion.crossfit(logp, y, folds)
            _, _, final = fusion.apply_final(logp, y, logp[:1])
            a[bad] = n[bad] = 1/7
        key = arm+"/"+name; output = args.output/key
        output.mkdir(parents=True, exist_ok=True)
        item = {"arm": arm, "system": name, "components": components,
                "cv": {"op_a": et_metrics(a, records, bad), "op_n": et_metrics(n, records, bad)},
                "parameters_oof": params, "parameters_final": final, "artifacts": {}}
        paired_rows[key] = scored(n, records, bad)
        def save(split, rows, pa, pn, invalid):
            path = output/(split+".jsonl")
            write_rows(path, [{**r, "op_a": aa.tolist(), "op_n": nn.tolist(), "forced_error": bool(b)}
                             for r, aa, nn, b in zip(rows, pa, pn, invalid, strict=True)])
            item["artifacts"][split] = {"path": str(path), "sha256": sha256(path)}
        save("oof", records, a, n, bad)
        for split in ("et_test", "sims_valid", "mer_test"):
            rows, target, invalid = stack(args, arm, components, split, artifacts, face_missing)
            if split == "et_test" and {r["utt_id"] for r in rows} != test: raise ValueError("ET-test membership changed")
            if split == "mer_test":
                pa, pn = extend_mer(target, final, single=single, unsupported_worried=arm in ("D0", "Q"))
                classes = CORPUS_CLASSES["MER"]
                if invalid.any():
                    for p in (pa, pn):
                        p[invalid] = 0
                        p[np.ix_(invalid, classes)] = 1/len(classes)
                item[split] = {point: {"native6": native_metrics(p, rows, invalid),
                    "shared5": native_metrics(p, rows, invalid, shared=True)} for point, p in (("op_a", pa), ("op_n", pn))}
                if arm in ("D0", "Q"):
                    worried = [r for r in rows if r["truth_M"] == "worried"]
                    item["worried_to_ET_op_n"] = dict(collections.Counter(LABELS[int(p[:7].argmax())]
                        for r, p, bad_row in zip(rows, pn, invalid) if r["truth_M"] == "worried" and not bad_row))
                    item["worried_support"] = len(worried)
                    item["worried_failed"] = sum(r["truth_M"] == "worried" and b for r, b in zip(rows, invalid))
            else:
                p = target[..., :7]/target[..., :7].sum(2, keepdims=True)
                lp = np.log(np.clip(p, 1e-12, 1))
                if single:
                    pa, pn = q_calibrate.apply_single(lp[:, 0], invalid, np.array(final["calibration"]), final["neutral_bias"])
                else:
                    logits = fusion.calibrated_logits(fusion.fusion_logits(lp, np.array(final["fusion"])), np.array(final["calibration"]))
                    pa = fusion.softmax(logits); logits[:, 0] += final["neutral_bias"]; pn = fusion.softmax(logits)
                    pa[invalid] = pn[invalid] = 1/7
                item[split] = et_metrics(pn, rows, invalid) if split == "et_test" else sims_metrics([
                    {"id": r["utt_id"], "label": r["label"], "s": None if b else float(p[1]-p[2:6].sum())}
                    for r, p, b in zip(rows, pn, invalid)])
            save(split, rows, pa, pn, invalid)
        write_json_atomic(output/"metrics.json", item)
        table[key] = item
        if name in ("A3", "V2", "V1"):
            job = args.derived/f"heads/{arm}/{name}"
            selections[key] = json.loads((job/"results.json").read_text())
            if arm == "D0":
                old_path = args.r2/f"heads/{name}_h2_cv/oof.jsonl"
                old = {r["utt_id"]: r for r in read_jsonl(old_path)}
                fresh = source(args, arm, name, "oof", artifacts)
                fresh.sort(key=lambda r: r["utt_id"])
                if set(old) != set(cv): raise ValueError("X02 reference coverage changed")
                p_old = np.array([old[r["utt_id"]]["probabilities"] for r in fresh])
                p_new = np.array([r["probabilities"] for r in fresh])
                raw_bad = np.array([r["status"] != "ok" for r in fresh])
                old_bad = np.array([old[r["utt_id"]]["status"] != "ok" for r in fresh])
                reproduction[name] = {"argmax_agreement": float((p_old.argmax(1) == p_new.argmax(1)).mean()),
                    "new_raw": et_metrics(p_new, fresh, raw_bad), "old_raw": et_metrics(p_old, fresh, old_bad),
                    "reference_path": str(old_path), "reference_sha256": sha256(old_path)}
                reproduction[name]["macro_f1_difference"] = reproduction[name]["new_raw"]["macro_f1"]-reproduction[name]["old_raw"]["macro_f1"]
        write_json_atomic(args.output/"progress.json", {"completed": key, "done": len(table), "total": len(planned)})
    comparisons = [("D1/"+s, "D0/"+s) for s in SYSTEMS] + [("D3/H_ref", "D1/H_ref")]
    if "D1_audioonly/A3" in paired_rows: comparisons += [("D1_audioonly/A3", "D1/A3")]
    deltas = {a+" minus "+b: paired_speakers(paired_rows[a], paired_rows[b]) for a, b in comparisons}
    for filename, value in (("main_table.json", table), ("paired_differences.json", deltas),
                            ("d0_reproducibility.json", reproduction), ("selected_configs.json", selections),
                            ("source_artifact_index.json", artifacts)):
        write_json_atomic(args.output/filename, value)
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                outputs={p.name: sha256(p) for p in args.output.glob("*.json") if p.name != "run_meta.json"})
    write_json_atomic(args.output/"run_meta.json", meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--derived", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--r2", type=Path, default=Path("/home/administrator/emotion-data/emotioncore_runs/X02"))
    p.add_argument("--r3", type=Path, default=Path("/home/administrator/emotion-data/emotioncore_runs/X03"))
    run(p.parse_args())


if __name__ == "__main__": main()
