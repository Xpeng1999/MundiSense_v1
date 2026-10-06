"""Report-only final-head targets; ET and SIMS remain seven-class cohorts."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
from pathlib import Path
import numpy as np
from emotioncore.x02 import head_h1
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.heads import Frames, build_model, pack
from emotioncore.x04.prepare import require_h, storage_gate, write_rows


def run(args):
    import torch
    git=git_provenance(Path.cwd());require_h(args.output);storage_gate()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    checkpoint=torch.load(args.job/"final_head.pt",map_location="cpu",weights_only=False)
    if checkpoint["encoder"]!=args.encoder or checkpoint["arm"]!=args.arm:raise ValueError("Head identity mismatch")
    args.output.mkdir(parents=True,exist_ok=True)
    meta={"git":git,"command":sys.argv,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"device":"cpu",
          "checkpoint_sha256":head_h1.sha256(args.job/"final_head.pt"),"target_labels_used_for_selection":False}
    write_json_atomic(args.output/"run_meta.json",meta)
    all_et=[json.loads(s) for s in Path("experiments/X01_benchmark/results/et_utterances.jsonl").read_text().splitlines()]
    mer=[json.loads(s) for s in (args.derived/"manifests/mer_test.jsonl").read_text().splitlines()]
    sims=[json.loads(s) for s in (args.r2/"sims_valid/sims_valid_adapter.jsonl").read_text().splitlines()]
    cohorts={"et_test":([r for r in all_et if r["official_split"]=="test"],args.r2/f"features/{args.encoder}/receipts"),
             "sims_valid":(sims,args.r2/f"sims_valid/features/{args.encoder}/receipts"),
             "mer_test":(mer,args.derived/f"full/features/{args.encoder}/receipts")}
    artifacts={}
    for split,(rows,receipts) in cohorts.items():
        expected={"et_test":1929,"sims_valid":647,"mer_test":572}[split]
        if len(rows)!=expected:raise ValueError("Frozen target count differs")
        frames,lengths=pack(rows,receipts,args.encoder,args.derived/f"target_packs/{split}/{args.encoder}")
        data=Frames([frames],[lengths]);model=build_model(data.dimension,data.layered,"cpu")
        model.load_state_dict(checkpoint["state_dict"]);model.eval()
        raw=np.full((len(rows),8),1/8,dtype=np.float32)
        with torch.inference_mode():
            for indices in batches(np.flatnonzero(lengths>0)):
                x,n=data.batch(indices,"cpu");raw[indices]=model(x,n).softmax(1).cpu().numpy()
        if args.arm=="D0":
            raw[:,7]=0;raw/=raw.sum(1,keepdims=True)
        et_probs=raw[:,:7]/raw[:,:7].sum(1,keepdims=True)
        output=[]
        for i,row in enumerate(rows):
            rec={"utt_id":row["utt_id"],"speaker_id":row["speaker_id"],"status":"ok" if lengths[i] else "failed",
                 "feature_available":bool(lengths[i]),"probabilities":et_probs[i].astype(float).tolist(),
                 "probabilities8":raw[i].astype(float).tolist()}
            if split=="et_test":rec["truth_M"]=row["labels"]["M"]
            if split=="sims_valid":rec.update(id=row["utt_id"],label=row["label"],audio_missing=bool(row.get("audio_missing",False)))
            if split=="mer_test":rec.update(truth_M=row["label"],native_label=row["native_label"],corpus="MER")
            output.append(rec)
        path=args.output/f"{split}.jsonl";write_rows(path,output)
        artifacts[split]={"path":str(path),"sha256":head_h1.sha256(path),"count":len(rows),"feature_available":int((lengths>0).sum())}
        del model,frames,data
    write_json_atomic(args.output/"artifacts.json",artifacts)
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),artifacts=artifacts)
    write_json_atomic(args.output/"run_meta.json",meta)


def batches(indices):
    for offset in range(0,len(indices),128):yield indices[offset:offset+128]


def main():
    p=argparse.ArgumentParser();p.add_argument("--encoder",choices=["A3","V2","V1"],required=True)
    p.add_argument("--arm",choices=["D0","D1","D2","D3"],required=True)
    p.add_argument("--job",type=Path,required=True);p.add_argument("--derived",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--r2",type=Path,default=Path("/home/administrator/emotion-data/emotioncore_runs/X02"))
    run(p.parse_args())


if __name__=="__main__":main()
