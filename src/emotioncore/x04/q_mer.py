"""Frozen X03 Q: MER-test A/AV inference and CPU-only X04b budget inputs."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x02.face import sha256
from emotioncore.x02.runtime import PersistentJSONWorker, git_provenance, write_json_atomic
from emotioncore.x03 import media, q_cache
from emotioncore.x03.q_smoke import MODEL, assert_modality_receipt, model_and_optimizer
from emotioncore.x03.q_train import load_input
from emotioncore.x04.features import YUNET
from emotioncore.x04.prepare import require_h, storage_gate, write_rows

ADAPTER = Path("/home/administrator/emotion-data/emotioncore_runs/X03/Q/final/best_adapter_eval8")
ADAPTER_SHA = "427ebc148d447ba03dc747429323b440b10cbb8b4b8876247f48f71573d2d5a4"


def read_rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def prepare(args):
    git=git_provenance(Path.cwd());require_h(args.derived);storage_gate()
    root=args.derived/"Q_media";root.mkdir(parents=True,exist_ok=True)
    meta={"git":git,"command":sys.argv,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "Q_training":False,"learned_inference":False,"size":media.SIZE,"frames":media.FRAME_COUNT,
        "min_face_px_at_448":media.MIN_FACE_PX_AT_448,"min_confidence":media.MIN_CONFIDENCE,
        "frozen_worker_sha256":sha256(Path(media.__file__)),"yunet_sha256":sha256(YUNET)}
    write_json_atomic(root/"run_meta.json",meta)
    rows=read_rows(args.derived/"manifests/mer_train_dedup.jsonl")+read_rows(args.derived/"manifests/mer_test.jsonl")
    if any(r["corpus"]!="MER" for r in rows):raise ValueError("Q media only MER")
    local=threading.local();workers=[];lock=threading.Lock()
    def one(row):
        storage_gate();uid=row["utt_id"];path=root/"receipts"/(uid+".json")
        face=args.derived/"full/face/receipts"/(uid+".json")
        if path.exists():
            old=json.loads(path.read_text())
            if old["input_sha256"]!=row["mp4"]["sha256"] or old["face_receipt_sha256"]!=sha256(face):raise ValueError("Q media input changed")
            return old
        if not hasattr(local,"worker"):
            local.worker=PersistentJSONWorker([sys.executable,"-u","-m","emotioncore.x03.media","--worker","--detector",str(YUNET)],cwd=Path.cwd(),env=os.environ.copy(),stderr_path=root/"worker_errors.log")
            with lock:workers.append(local.worker)
        result=local.worker.request({"utt_id":uid,"face_receipt":str(face),"input_sha256":row["mp4"]["sha256"],"duration_s":row["mp4"]["duration_s"],"output":str(root/"videos"/(uid+".mp4"))},120)
        rec={"utt_id":uid,"split":row["official_split"],"input_sha256":row["mp4"]["sha256"],"face_receipt_sha256":sha256(face),**result}
        write_json_atomic(path,rec);return rec
    try:
        receipts=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for i,rec in enumerate(pool.map(one,rows),1):
                receipts.append(rec)
                if i%100==0:write_json_atomic(root/"progress.json",{"done":i,"total":len(rows)})
    finally:
        for worker in workers:worker.close()
    byid={r["utt_id"]:r for r in receipts}; outputs={}
    for split in ("train","test"):
        selected=[r for r in rows if r["official_split"]==split]
        for form in (("A","V","AV") if split=="train" else ("A","AV")):
            output=[]
            for row in selected:
                video=byid[row["utt_id"]];missing=video["status"]!="ok"
                effective="A" if missing and form in ("V","AV") else form
                output.append({"utt_id":row["utt_id"],"speaker_id":"unknown","split":"mer_"+split,
                    "assigned_form":form,"effective_form":effective,"face_missing":missing,"face_missing_fallback":missing and form=="AV",
                    "wav":row["wav"]["path"],"wav_sha256":row["wav"]["sha256"],
                    "video":video.get("output") if effective in ("V","AV") else None,
                    "video_sha256":video.get("output_sha256") if effective in ("V","AV") else None,
                    "truth_M":row["label"]})
            manifest=root/"manifests"/f"{split}_{form}.jsonl";write_rows(manifest,output)
            outputs[f"{split}_{form}"]={"path":str(manifest),"sha256":sha256(manifest),"n":len(output)}
    summary={"outputs":outputs,"by_split":{s:{"n":sum(r["split"]==s for r in receipts),"missing":sum(r["split"]==s and r["status"]!="ok" for r in receipts)} for s in ("train","test")}}
    write_json_atomic(root/"summary.json",summary);meta["ended_at"]=dt.datetime.now(dt.timezone.utc).isoformat();write_json_atomic(root/"run_meta.json",meta)


def cache(args):
    git_provenance(Path.cwd());storage_gate();require_h(args.derived)
    root=args.derived/"Q_media"
    summary=json.loads((root/"summary.json").read_text())
    for name,spec in summary["outputs"].items():
        path=Path(spec["path"])
        if sha256(path)!=spec["sha256"]:raise ValueError("Q MER input manifest changed")
        q_cache.run(argparse.Namespace(manifest=path,cache_root=args.derived/"Q_inputs",output_dir=args.derived/"Q_cache"/name,workers=4,timeout=120,allow_dirty=False))


def predict(args):
    import torch
    from transformers import Qwen2_5OmniProcessor
    git=git_provenance(Path.cwd());storage_gate();require_h(args.output)
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"):raise RuntimeError("Q requires default allocator")
    if sha256(ADAPTER/"adapter_model.safetensors")!=ADAPTER_SHA:raise ValueError("Frozen adapter changed")
    root=args.derived/"Q_media";summary=json.loads((root/"summary.json").read_text())
    manifests={}
    for form in ("A","AV"):
        spec=summary["outputs"]["test_"+form];path=Path(spec["path"])
        if sha256(path)!=spec["sha256"]:raise ValueError("MER-test manifest changed")
        rows=read_rows(path)
        if len(rows)!=572 or len({r["utt_id"] for r in rows})!=572 or any(r["split"]!="mer_test" for r in rows):raise ValueError("Only 572 MER-test rows permitted for Q inference")
        manifests[form]=rows
    args.output.mkdir(parents=True,exist_ok=True)
    meta={"git":git,"command":sys.argv,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "adapter_sha256":ADAPTER_SHA,"base_index_sha256":sha256(MODEL/"model.safetensors.index.json"),"Q_training":False,"cuda_allocator":"default","forms":["A","AV"]}
    write_json_atomic(args.output/"run_meta.json",meta)
    started=time.monotonic();torch.set_num_threads(4)
    processor=Qwen2_5OmniProcessor.from_pretrained(str(MODEL),local_files_only=True)
    token_ids=[processor.tokenizer(label,add_special_tokens=False).input_ids[0] for label in LABELS]
    if len(set(token_ids))!=7:raise ValueError("Frozen Q first-token labels collide")
    thinker,optimizer,info=model_and_optimizer();del optimizer
    thinker.load_adapter(str(ADAPTER),adapter_name="best");thinker.set_adapter("best");thinker.eval()
    for parameter in thinker.parameters():parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats();artifacts={}
    with torch.inference_mode():
        for form,rows in manifests.items():
            output=[]
            for i,row in enumerate(rows):
                if i%50==0:storage_gate()
                values=load_input(args.derived/"Q_inputs",row)
                assert_modality_receipt(row["effective_form"],set(values))
                result=thinker(**values,use_cache=False,use_audio_in_video=False)
                p=result.logits[0,-1].float()[token_ids].softmax(0).cpu().tolist()
                output.append({"utt_id":row["utt_id"],"speaker_id":"unknown","truth_M":row["truth_M"],"probabilities":p,
                    "probabilities8":p+[0.],"face_missing":row["face_missing"],"face_missing_fallback":row["face_missing_fallback"],
                    "status":"face_missing_fallback" if row["face_missing_fallback"] else "ok","form":form,"effective_form":row["effective_form"]})
                del values,result
                if i%50==0:write_json_atomic(args.output/"progress.json",{"form":form,"done":i+1,"total":len(rows)})
            path=args.output/(form+".jsonl");write_rows(path,output);artifacts[form]={"path":str(path),"sha256":sha256(path),"n":len(output)}
    meta.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),gpu_wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_checkpointing=info["gradient_checkpointing"],artifacts=artifacts)
    write_json_atomic(args.output/"run_meta.json",meta);write_json_atomic(args.output/"artifacts.json",artifacts)


def main():
    p=argparse.ArgumentParser();p.add_argument("--action",choices=["media","cache","predict"],required=True)
    p.add_argument("--derived",type=Path,required=True);p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.action=="predict" and args.output is None:p.error("--output required for predict")
    {"media":prepare,"cache":cache,"predict":predict}[args.action](args)


if __name__=="__main__":main()
