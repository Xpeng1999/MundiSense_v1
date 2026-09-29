"""Crossfit Q-A/Q-AV probability fusion and the five preregistered H systems."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from emotioncore.eval.metrics import LABELS, sims_metrics
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x03.budget_v2 import FACE_SHA
from emotioncore.x03.fusion import apply_final, crossfit, load_sources, write_predictions
from emotioncore.x03.q_calibrate import metrics
from emotioncore.x03.q_manifest import FOLD_SHA, read_jsonl, sha256


COMBINATIONS = {
    "Q_fusion": ("Q_A", "Q_AV"),
    "H_QAV_V2": ("Q_AV", "V2_h2"),
    "H_QAV_V2_V1": ("Q_AV", "V2_h2", "V1_h1"),
    "H_QAV_A3_V2": ("Q_AV", "A3_h2", "V2_h2"),
    "H_QA_V2": ("Q_A", "V2_h2"),
    "H_QA_QAV_A3_V2_V1": ("Q_A", "Q_AV", "A3_h2", "V2_h2", "V1_h1"),
}


def q_rows(root: Path, form: str, split: str) -> tuple[list[dict], list[dict]]:
    if split == "oof":
        paths = [root / f"predict_fold{i}" / f"{form}.jsonl" for i in range(5)]
    else:
        paths = [root / f"predict_final_{split}" / f"{form}.jsonl"]
    rows = []
    sources = []
    for path in paths:
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        digest = sha256(path)
        if summary[form]["sha256"] != digest:
            raise ValueError(f"Q raw prediction SHA mismatch: {path}")
        rows.extend(read_jsonl(path))
        sources.append({"path": str(path), "sha256": digest})
    if len(rows) != len({r["utt_id"] for r in rows}):
        raise ValueError(f"Repeated Q-{form} {split} IDs")
    rows.sort(key=lambda r: r["utt_id"])
    return rows, sources


def stack_inputs(components: tuple[str, ...], split: str, q_root: Path,
                 index: Path, face_missing: set[str]) -> tuple[list[str], np.ndarray, list[dict], dict]:
    q_forms = sorted({component.removeprefix("Q_") for component in components
                      if component.startswith("Q_")})
    q_data = {form: q_rows(q_root, form, split) for form in q_forms}
    if not q_data:
        raise ValueError("H candidate has no Q component")
    first_form = components[0].removeprefix("Q_")
    base, _ = q_data[first_form]
    ids = [r["utt_id"] for r in base]
    if split == "oof" and len(ids) != 16366 or split == "et_test" and len(ids) != 1929 or split == "sims_valid" and len(ids) != 647:
        raise ValueError("Q target coverage differs from frozen split")
    q_maps = {form: {r["utt_id"]: r for r in rows} for form, (rows, _) in q_data.items()}
    if any(set(mapping) != set(ids) for mapping in q_maps.values()):
        raise ValueError("Q forms have unequal ID coverage")
    experts = tuple(c for c in components if not c.startswith("Q_"))
    expert_maps = {}
    expert_sources = {}
    if experts:
        expert_ids, expert_logp, expert_rows, expert_sources = load_sources(
            index, experts, split, face_missing)
        if set(expert_ids) != set(ids):
            raise ValueError("Q and X02 expert IDs differ")
        location = {uid: i for i, uid in enumerate(expert_ids)}
        for column, expert in enumerate(experts):
            expert_maps[expert] = expert_logp[[location[uid] for uid in ids], column]
        for row, expert_row in zip(base, [expert_rows[location[uid]] for uid in ids], strict=True):
            if split == "sims_valid":
                if row["label"] != expert_row["label"]:
                    raise ValueError("SIMS-valid label differs between Q and X02")
            elif (row["speaker_id"], row["truth_M"], row.get("fold")) != \
                 (expert_row["speaker_id"], expert_row["truth_M"], expert_row.get("fold")):
                raise ValueError("Q and X02 frozen labels/folds differ")
    columns = []
    for component in components:
        if component.startswith("Q_"):
            form = component.removeprefix("Q_")
            columns.append(np.log(np.clip(np.array([q_maps[form][uid]["probabilities"] for uid in ids]),
                                          1e-12, 1)))
        else:
            columns.append(expert_maps[component])
    logp = np.stack(columns, axis=1)
    if logp.shape != (len(ids), len(components), 7) or not np.isfinite(logp).all():
        raise ValueError("Invalid Q/expert probability stack")
    return ids, logp, base, {"q": {form: spec for form, (_, spec) in q_data.items()},
                             "experts": expert_sources}


def run(args: argparse.Namespace) -> None:
    git = git_provenance(Path.cwd(), args.allow_dirty)
    if sha256(args.folds) != FOLD_SHA or sha256(args.face_policy) != FACE_SHA:
        raise ValueError("Frozen X03 fold/face policy changed")
    frozen = json.loads(args.folds.read_text(encoding="utf-8"))
    face_missing = set(json.loads(args.face_policy.read_text(encoding="utf-8"))["face_missing_utt_ids"])
    expected_fold = {s: f["fold"] for f in frozen["folds"] for s in f["outer_holdout_speakers"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "folds_sha256": FOLD_SHA, "face_policy_sha256": FACE_SHA,
            "artifact_index_sha256": sha256(args.artifact_index), "q_root": str(args.q_root)}
    write_json_atomic(args.output_dir / "run_meta.json", meta)
    summary = {}
    for name, components in COMBINATIONS.items():
        ids, logp, records, sources = stack_inputs(components, "oof", args.q_root,
                                                   args.artifact_index, face_missing)
        if any(r["fold"] != expected_fold[r["speaker_id"]] for r in records):
            raise ValueError("Q raw OOF fold differs from frozen speakers")
        y = np.array([LABELS.index(r["truth_M"]) for r in records])
        fold = np.array([r["fold"] for r in records])
        a, n, params = crossfit(logp, y, fold)
        output = args.output_dir / name
        output.mkdir(parents=True, exist_ok=True)
        write_predictions(output / "oof.jsonl", ids, records, a, n, "oof")
        result = {"components": components,
                  "cv": {"op_a": metrics(a, y, np.zeros(len(y), dtype=bool), records),
                         "op_n": metrics(n, y, np.zeros(len(y), dtype=bool), records)},
                  "parameters_oof": params, "sources": {"oof": sources}}
        paths = {"oof": {"path": str(output / "oof.jsonl"),
                         "sha256": sha256(output / "oof.jsonl")}}
        for split in ("et_test", "sims_valid"):
            target_ids, target_logp, target_records, target_sources = stack_inputs(
                components, split, args.q_root, args.artifact_index, face_missing)
            target_a, target_n, final_params = apply_final(logp, y, target_logp)
            invalid = np.array([r["status"] == "missing_media" for r in target_records])
            target_a[invalid] = 1 / 7
            target_n[invalid] = 1 / 7
            path = output / f"{split}.jsonl"
            write_predictions(path, target_ids, target_records, target_a, target_n, split)
            paths[split] = {"path": str(path), "sha256": sha256(path)}
            result["sources"][split] = target_sources
            result["parameters_final"] = final_params
            if split == "et_test":
                truth = np.array([LABELS.index(r["truth_M"]) for r in target_records])
                result[split] = {"op_a": metrics(target_a, truth, invalid, target_records),
                                 "op_n": metrics(target_n, truth, invalid, target_records)}
            else:
                result[split] = sims_metrics([
                    {"id": uid, "label": row["label"],
                     "s": None if bad else float(p[1] - p[2:6].sum())}
                    for uid, row, p, bad in zip(target_ids, target_records, target_n,
                                                invalid, strict=True)])
        result["artifacts"] = paths
        write_json_atomic(output / "metrics.json", result)
        summary[name] = {"cv_op_n_macro_f1": result["cv"]["op_n"]["macro_f1"],
                         "et_test_op_n_macro_f1": result["et_test"]["op_n"]["macro_f1"],
                         "sims_valid_pearson": result["sims_valid"]["pearson"]}
    summary["best_h"] = max((name for name in COMBINATIONS if name.startswith("H_")),
                             key=lambda name: summary[name]["cv_op_n_macro_f1"])
    write_json_atomic(args.output_dir / "summary.json", summary)
    meta["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta["summary_sha256"] = sha256(args.output_dir / "summary.json")
    write_json_atomic(args.output_dir / "run_meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--face-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
