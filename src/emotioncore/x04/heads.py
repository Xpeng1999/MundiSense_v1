"""Eight-output X02 h2 heads with corpus masks and frozen nested speaker CV."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x02 import head_h1, head_h2
from emotioncore.x02.folds import validate_oof
from emotioncore.x02.frames_pack import SPECS, read_feature
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.prepare import require_h, storage_gate, write_rows

LABELS8 = (*LABELS, "worried")
CORPUS_CLASSES = {"ET": (0,1,2,3,4,5,6), "MER": (0,1,2,3,6,7), "CREMA": (0,1,2,3,4,5)}
ALPHAS = (.25, .5, 1.)
FOLDS_SHA = "85564e8a22e6aa76775061de6d2e322ba80d10c1fe77d11edef4863d8b5791ce"


def build_model(dimension, layered, device):
    import torch
    model = head_h2.build_model(dimension, layered, device)
    old = model.classifier[-1]
    # Preserve the original seven-class initialization and RNG sequence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_h1.SEED + 8000)
        replacement = torch.nn.Linear(256, 8)
    with torch.no_grad():
        replacement.weight[:7].copy_(old.weight.cpu())
        replacement.bias[:7].copy_(old.bias.cpu())
    model.classifier[-1] = replacement.to(device)
    return model


def masks(corpora, device):
    import torch
    return torch.tensor([[i in CORPUS_CLASSES[c] for i in range(8)] for c in corpora], dtype=torch.bool, device=device)


def masked_ce(logits, labels, mask):
    import torch
    if not mask.gather(1, labels[:, None]).all():
        raise ValueError("Training truth is absent from corpus mask")
    return torch.nn.functional.cross_entropy(logits.masked_fill(~mask, float("-inf")), labels, reduction="none")


def validate_training(rows, indices, held, inner, excluded, *, stage):
    forbidden = set(held) | (set(inner) if stage == "selection" else set())
    identifiers = []
    for index in indices:
        row = rows[int(index)]; identifiers.append(row["utt_id"])
        if row["corpus"] not in CORPUS_CLASSES:
            raise ValueError("STOP unsupported corpus (including CH-SIMS) in training")
        if row["corpus"] == "ET":
            if row["speaker_id"] in forbidden or row["official_split"] == "test":
                raise ValueError("STOP ET speaker/split leakage")
        elif row["official_split"] != "train" or row["utt_id"] in excluded:
            raise ValueError("STOP MER-test/dedup exclusion leakage")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("STOP duplicated training ID")


def pack(rows, receipts_root, encoder, output):
    require_h(output); storage_gate()
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output / "pack_meta.json"
    identities = [r["utt_id"] for r in rows]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["ids"] != identities or meta["encoder"] != encoder:
            raise ValueError("Cached frame pack identity changed")
        for key in ("frame", "lengths"):
            if head_h1.sha256(Path(meta[key + "_file"])) != meta[key + "_sha256"]:
                raise ValueError("Cached frame pack SHA changed")
        return np.load(meta["frame_file"], mmap_mode="r"), np.load(meta["lengths_file"])
    count, dimension = SPECS[encoder]
    shape = (len(rows), 4, count, dimension) if encoder == "A3" else (len(rows), count, dimension)
    frame_path = output / "frames.npy"
    frames = np.lib.format.open_memmap(frame_path, mode="w+", dtype=np.float16, shape=shape)
    lengths = np.zeros(len(rows), dtype=np.int16)
    for i, row in enumerate(rows):
        receipt = json.loads((receipts_root / (row["utt_id"] + ".json")).read_text())
        source = row["wav" if encoder == "A3" else "mp4"]
        if receipt["id"] != row["utt_id"] or receipt["input_sha256"] != source["sha256"]:
            raise ValueError("Frame pack input provenance mismatch")
        if receipt["status"] != "ok":
            continue
        path = Path(receipt["feature_path"])
        if head_h1.sha256(path) != receipt["feature_sha256"]:
            raise ValueError("Frame feature SHA mismatch")
        feature = read_feature(path, encoder)
        if not np.isfinite(feature).all() or feature.shape[-1] != dimension:
            raise ValueError("Invalid frame feature")
        subset = np.linspace(0, feature.shape[-2]-1, min(count, feature.shape[-2])).round().astype(int)
        frames[i, ..., :len(subset), :] = feature[..., subset, :]
        lengths[i] = len(subset)
    frames.flush(); del frames
    length_path = output / "lengths.npy"; np.save(length_path, lengths)
    write_json_atomic(meta_path, {"encoder": encoder, "ids": identities,
        "frame_file": str(frame_path), "frame_sha256": head_h1.sha256(frame_path),
        "lengths_file": str(length_path), "lengths_sha256": head_h1.sha256(length_path),
        "shape": shape, "subsample": "X02 evenly spaced incl first and last"})
    return np.load(frame_path, mmap_mode="r"), lengths


class Frames:
    """Logical concatenation avoids duplicating the 17-GB frozen A3 CV pack."""
    def __init__(self, arrays, lengths):
        self.arrays = arrays; self.offsets = np.cumsum([0] + [len(a) for a in arrays])
        self.lengths = np.concatenate(lengths)
        self.layered = arrays[0].ndim == 4; self.dimension = arrays[0].shape[-1]

    def batch(self, indices, device):
        import torch
        shape = (len(indices), *self.arrays[0].shape[1:])
        x = np.empty(shape, dtype=np.float32)
        for j, a in enumerate(self.arrays):
            valid = (indices >= self.offsets[j]) & (indices < self.offsets[j+1])
            if valid.any():
                x[valid] = a[indices[valid] - self.offsets[j]]
        return torch.from_numpy(x).to(device), torch.from_numpy(self.lengths[indices].astype(np.int64)).to(device)


def predict(model, data, indices, device, corpus="ET", unsupported_worried=False):
    import torch
    result = np.full((len(indices), 8), 0., dtype=np.float32)
    classes = CORPUS_CLASSES[corpus]
    if unsupported_worried and corpus == "MER":
        classes = tuple(i for i in classes if i != 7)
    result[:, classes] = 1/len(classes)
    valid = data.lengths[indices] > 0
    model.eval()
    with torch.inference_mode():
        for positions in head_h2.batches(np.flatnonzero(valid), 128):
            x, n = data.batch(indices[positions], device)
            probs = model(x, n)[:, classes].softmax(1).cpu().numpy()
            result[np.ix_(positions, classes)] = probs
    return result


def train(data, rows, train_indices, val_indices, config, seed, device, fixed_epochs=None):
    import torch
    valid_train = train_indices[data.lengths[train_indices] > 0]
    if not len(valid_train):
        raise ValueError("No usable training features")
    y = np.asarray([LABELS8.index(r["label"]) for r in rows])
    sample_weights = np.zeros(len(rows), dtype=np.float32)
    alpha_weights = np.ones(len(rows), dtype=np.float32)
    counts = {}
    for corpus, classes in CORPUS_CLASSES.items():
        ids = np.asarray([i for i in valid_train if rows[i]["corpus"] == corpus], dtype=int)
        if not len(ids): continue
        counts[corpus] = np.bincount(y[ids], minlength=8).tolist()
        if any(counts[corpus][k] == 0 for k in classes):
            raise ValueError(f"Missing expected class in {corpus} training fold")
        for index in ids:
            sample_weights[index] = len(ids) / (len(classes) * counts[corpus][y[index]])
            if corpus != "ET": alpha_weights[index] = config["alpha"]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = build_model(data.dimension, data.layered, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best = None; best_state = None; best_epoch = 0; patience = 0; epochs = 0
    history = []
    for epoch in range(1, (fixed_epochs if fixed_epochs is not None else head_h2.MAX_EPOCHS)+1):
        storage_gate()
        model.train()
        for batch in head_h2.batches(valid_train, 128, seed + epoch):
            x, n = data.batch(batch, device)
            labels = torch.from_numpy(y[batch]).to(device)
            weight = torch.from_numpy(sample_weights[batch]).to(device)
            alpha = torch.from_numpy(alpha_weights[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            # D0 uses X02's weighted-mean CE exactly; alpha is applied afterward.
            loss = (masked_ce(model(x, n), labels, masks([rows[i]["corpus"] for i in batch], device)) * weight * alpha).sum() / weight.sum()
            loss.backward(); optimizer.step()
        epochs = epoch
        if val_indices is None: continue
        probabilities = predict(model, data, val_indices, device)
        score = head_h1.scores(y[val_indices], probabilities.argmax(1), data.lengths[val_indices] > 0)
        history.append({"epoch": epoch, **score})
        rank = (score["macro_f1"], score["uar"])
        if best is None or rank > best:
            best = rank; best_epoch = epoch; patience = 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= head_h2.PATIENCE: break
    if best_state is not None: model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch if val_indices is not None else fixed_epochs,
        "validation": {"macro_f1": best[0], "uar": best[1]} if best else None,
        "epochs_executed": epochs, "history": history, "class_counts": counts}


def choose(candidates):
    return max(candidates, key=lambda r:(r["validation"]["macro_f1"], r["validation"]["uar"],
        -r["learning_rate"], -r["weight_decay"], -r["alpha"]))


def run(args):
    import torch
    git = git_provenance(Path.cwd()); require_h(args.output); storage_gate()
    if args.device != "cpu":
        raise ValueError("X04a head launcher reserves GPU budget by using CPU only")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    folds_path = Path("configs/folds_v1.json")
    if head_h1.sha256(folds_path) != FOLDS_SHA: raise ValueError("Frozen folds changed")
    folds = json.loads(folds_path.read_text())
    manifest = Path("experiments/X01_benchmark/results/et_utterances.jsonl")
    cv = head_h1.load_rows(manifest, folds)
    for row in cv: row.update(corpus="ET", label=row["labels"]["M"])
    extra_rows = [json.loads(s) for s in (args.derived / "manifests/all_features.jsonl").read_text().splitlines()]
    budget_path = args.derived / "budget_preflight.json"
    budget = json.loads(budget_path.read_text())
    if not budget["passed"] or head_h1.sha256(args.derived / "manifests/all_features.jsonl") != budget["all_features_manifest_sha256"] or head_h1.sha256(args.dedup) != budget["dedup_report_sha256"]:
        raise ValueError("Data or dedup changed after X04a budget/freeze gate")
    excluded = set(json.loads(args.dedup.read_text())["excluded_ids"])
    if args.sensitivity: excluded = set(json.loads(args.dedup.read_text())["audio_confirmed_ids"])
    mer = [i for i,r in enumerate(extra_rows) if r["corpus"] == "MER" and r["official_split"] == "train" and r["utt_id"] not in excluded]
    crema = [i for i,r in enumerate(extra_rows) if r["corpus"] == "CREMA"]
    rows = cv + extra_rows
    extra = ([] if args.arm == "D0" else mer) + (crema if args.arm == "D3" else [])
    extra_ids = np.asarray([len(cv)+i for i in extra], dtype=int)
    et_pack = json.loads((args.r2 / f"packs/{args.encoder}/pack_meta.json").read_text())
    if et_pack["folds_sha256"] != FOLDS_SHA or et_pack["manifest_sha256"] != head_h1.sha256(manifest):
        raise ValueError("Frozen ET feature pack provenance differs")
    for k in ("frame", "lengths"):
        if head_h1.sha256(Path(et_pack[k + "_file"])) != et_pack[k + "_sha256"]: raise ValueError("ET pack SHA changed")
    ef = np.load(et_pack["frame_file"], mmap_mode="r"); el = np.load(et_pack["lengths_file"])
    nf, nl = pack(extra_rows, args.derived / f"full/features/{args.encoder}/receipts", args.encoder, args.derived / f"packs/{args.encoder}")
    data = Frames([ef, nf], [el,nl]); speakers = np.asarray([r["speaker_id"] for r in cv])
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"git": git, "command": sys.argv, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "encoder": args.encoder, "arm": args.arm, "device": args.device,
        "heads_only": True, "Q_training": False, "folds_sha256": FOLDS_SHA,
        "seed": head_h1.SEED, "max_epochs": head_h2.MAX_EPOCHS, "patience": head_h2.PATIENCE,
        "batch_size":128,"dropout":head_h2.DROPOUT,"alphas":ALPHAS if args.arm != "D0" else [1.],
        "extra_train_count":len(extra_ids),"excluded_ids_sha256":head_h1.sha256(args.dedup),
        "weight_normalization":"per-corpus sample-average=1, batch denominator sum base weights, then alpha numerator"}
    write_json_atomic(args.output / "run_meta.json", meta)
    configs = [dict(learning_rate=lr, weight_decay=wd, alpha=a) for lr in head_h1.LEARNING_RATES for wd in head_h1.WEIGHT_DECAYS for a in (ALPHAS if args.arm != "D0" else [1.])]
    started = time.monotonic(); completed = []; oof=[]; config_history = {i:[] for i in range(len(configs))}
    for fold in folds["folds"]:
        outer = np.flatnonzero(np.isin(speakers,fold["outer_holdout_speakers"]))
        inner = np.flatnonzero(np.isin(speakers,fold["inner_validation_speakers"]))
        et_train = np.flatnonzero(np.isin(speakers,fold["inner_train_speakers"])) if args.arm != "D2" else np.array([],dtype=int)
        train_indices = np.concatenate([et_train,extra_ids])
        validate_training(rows,train_indices,fold["outer_holdout_speakers"],fold["inner_validation_speakers"],excluded,stage="selection")
        candidates=[]
        for i, config in enumerate(configs):
            rec_path=args.output/f"fold{fold['fold']}/grid{i}.json"
            if rec_path.exists():
                candidate=json.loads(rec_path.read_text())
            else:
                model, result=train(data,rows,train_indices,inner,config,head_h1.SEED+fold["fold"],args.device)
                candidate={**config,**result}; del model
                write_json_atomic(rec_path,candidate)
            candidates.append(candidate); config_history[i].append(candidate)
            write_json_atomic(args.output/"progress.json",{"fold":fold["fold"],"grid_done":i+1,"grid_total":len(configs),"phase":"selection","elapsed_s":time.monotonic()-started})
        chosen=choose(candidates)
        et_outer_train=np.flatnonzero(~np.isin(speakers,fold["outer_holdout_speakers"])) if args.arm != "D2" else np.array([],dtype=int)
        outer_train=np.concatenate([et_outer_train,extra_ids])
        validate_training(rows,outer_train,fold["outer_holdout_speakers"],[],excluded,stage="refit")
        model, _=train(data,rows,outer_train,None,chosen,head_h1.SEED+fold["fold"],args.device,fixed_epochs=chosen["best_epoch"])
        probs=predict(model,data,outer,args.device)
        for index,p in zip(outer,probs):
            r=rows[index];oof.append({"utt_id":r["utt_id"],"speaker_id":r["speaker_id"],"fold":fold["fold"],"truth_M":r["label"],"probabilities":p[:7].astype(float).tolist(),"probabilities8":p.astype(float).tolist(),"status":"ok" if data.lengths[index] else "failed"})
        completed.append({"fold":fold["fold"],"selected":chosen,"outer_count":len(outer),"train_count":len(outer_train)})
        write_rows(args.output/f"fold{fold['fold']}/training_ids.jsonl",[{"utt_id":rows[i]["utt_id"],"corpus":rows[i]["corpus"],"speaker_id":rows[i]["speaker_id"]} for i in outer_train])
        write_rows(args.output/"oof_partial.jsonl",oof);del model
        storage_gate()
    validate_oof({r["utt_id"] for r in cv},oof); oof.sort(key=lambda r:r["utt_id"])
    write_rows(args.output/"oof.jsonl",oof)
    final_grid=[]
    for i, config in enumerate(configs):
        history=config_history[i]
        final_grid.append({**config,"validation":{"macro_f1":statistics.mean(r["validation"]["macro_f1"] for r in history),"uar":statistics.mean(r["validation"]["uar"] for r in history)},"best_epoch":int(statistics.median_low(r["best_epoch"] for r in history))})
    final=choose(final_grid)
    final_ids=np.concatenate([np.arange(len(cv)) if args.arm != "D2" else np.array([],dtype=int),extra_ids])
    validate_training(rows,final_ids,folds["excluded_speakers"],[],excluded,stage="final")
    model,_=train(data,rows,final_ids,None,final,head_h1.SEED,args.device,fixed_epochs=final["best_epoch"])
    torch.save({"state_dict":model.state_dict(),"encoder":args.encoder,"arm":args.arm,"head":"h2_masked8","config":final},args.output/"final_head.pt")
    write_json_atomic(args.output/"results.json",{"folds":completed,"final_grid":final_grid,"final_config":final,"oof_sha256":head_h1.sha256(args.output/"oof.jsonl"),"checkpoint_sha256":head_h1.sha256(args.output/"final_head.pt"),"elapsed_s":time.monotonic()-started})
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_s=time.monotonic()-started)
    write_json_atomic(args.output/"run_meta.json",meta)


def main():
    p=argparse.ArgumentParser();p.add_argument("--arm",choices=["D0","D1","D2","D3"],required=True)
    p.add_argument("--encoder",choices=["A3","V2","V1"],required=True)
    p.add_argument("--derived",type=Path,required=True);p.add_argument("--dedup",type=Path,required=True)
    p.add_argument("--r2",type=Path,default=Path("/home/administrator/emotion-data/emotioncore_runs/X02"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");p.add_argument("--threads",type=int,default=8)
    p.add_argument("--sensitivity",action="store_true");args=p.parse_args()
    if args.sensitivity and (args.arm!="D1" or args.encoder!="A3"):p.error("Only D1 A3 sensitivity is authorized")
    run(args)


if __name__=="__main__":main()
