"""Authorized X04a execution, with budget/storage gates and isolated children."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from emotioncore.x02.face import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.heads import pack
from emotioncore.x04.prepare import require_h, storage_gate


def run(args):
    git=git_provenance(Path.cwd());require_h(args.derived);storage_gate()
    recorded=Path("experiments/X04a_data/results/budget_preflight.json")
    if not recorded.exists() or sha256(recorded)!=sha256(args.budget):
        raise ValueError("Budget receipt must be committed with REPORT before full execution")
    budget=json.loads(args.budget.read_text())
    if not budget["passed"] or budget["total_projection_gpu_hours"]>20:raise ValueError("X04a budget gate closed")
    source=args.derived/"manifests/all_features.jsonl"
    dedup=args.preflight/"dedup/dedup_report.json"
    # Older preflight worktrees lack identity fields; freeze identities separately,
    # without editing the original budget or its measured numbers.
    frozen={**budget,"source_budget_sha256":sha256(args.budget),
        "all_features_manifest_sha256":sha256(source),"dedup_report_sha256":sha256(dedup)}
    write_json_atomic(args.derived/"budget_preflight.json",frozen)
    args.output.mkdir(parents=True,exist_ok=True)
    progress={"git":git,"command":sys.argv,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),
        "budget_sha256":sha256(args.budget),"report_sha256":sha256(Path("experiments/X04a_data/REPORT.md")),"steps":[],"status":"running"}
    write_json_atomic(args.output/"pipeline_progress.json",progress)
    started_pipeline=time.monotonic()
    charged=(budget["measured_gpu_hours"]+args.prior_gpu_hours)*3600
    progress["prior_gpu_hours"]=args.prior_gpu_hours
    children=[];cpu_jobs=[]
    def spawn(name,module,options,*,cpu=False):
        env=os.environ.copy();env["PYTHONDONTWRITEBYTECODE"]="1"
        if cpu:env["CUDA_VISIBLE_DEVICES"]=""
        else:
            env.pop("PYTORCH_CUDA_ALLOC_CONF",None);env.pop("PYTORCH_ALLOC_CONF",None)
        command=[sys.executable,"-u","-m",module,*map(str,options)]
        stream=(args.output/(name+".log")).open("ab",buffering=0)
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                               env=env,cwd=Path.cwd(),start_new_session=True)
        stream.close();record={"name":name,"pid":child.pid,"command":command,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"cpu_only":cpu}
        children.append(child);progress["steps"].append(record)
        write_json_atomic(args.output/"pipeline_progress.json",progress)
        return child,record,time.monotonic()
    def wait(item,*,gpu=False):
        nonlocal charged
        child,record,started=item
        while child.poll() is None:
            storage_gate()
            for other in children:
                if other.poll() not in (None, 0):
                    raise RuntimeError(f"STOP concurrent X04a child failed: pid={other.pid} exit={other.returncode}")
            if gpu and charged+time.monotonic()-started>20*3600-120:
                raise RuntimeError("STOP actual GPU time approaches 20h cap")
            time.sleep(5)
        elapsed=time.monotonic()-started
        record.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_s=elapsed,returncode=child.returncode)
        if gpu:charged+=elapsed
        progress["charged_gpu_hours"]=charged/3600
        write_json_atomic(args.output/"pipeline_progress.json",progress)
        if child.returncode:raise RuntimeError(f"X04a child failed: {record['name']} exit={child.returncode}")
    def execute(name,module,options,*,cpu=False,gpu=False):wait(spawn(name,module,options,cpu=cpu),gpu=gpu)
    def heads(encoder):
        if encoder=="A3":raise ValueError("A3 must be serialized on GPU")
        import json as _json
        rows=[_json.loads(s) for s in source.read_text().splitlines()]
        pack(rows,args.derived/f"full/features/{encoder}/receipts",encoder,args.derived/f"packs/{encoder}")
        for arm in ("D0","D1","D2","D3"):
            output=args.derived/f"heads/{arm}/{encoder}"
            options=["--arm",arm,"--encoder",encoder,"--derived",args.derived,"--dedup",dedup,
                     "--output",output,"--device","cpu","--threads",8 if encoder=="A3" else 4]
            cpu_jobs.append((arm,encoder,output,spawn(f"heads_{arm}_{encoder}","emotioncore.x04.heads",options,cpu=True)))
    def a3_options(arm,*,first=False,sensitivity=False):
        output=args.derived/f"heads/{arm}/A3"
        options=["--arm","D1" if sensitivity else arm,"--encoder","A3","--derived",args.derived,
                 "--dedup",dedup,"--output",output,"--device","cuda:0","--threads",8]
        if first:options.append("--stop-after-first-grid")
        if sensitivity:options.append("--sensitivity")
        return options
    try:
        face=spawn("full_face","emotioncore.x04.features",["--stage","face","--manifest",source,"--run-root",args.derived/"full"],cpu=True)
        execute("full_A3","emotioncore.x04.features",["--stage","A3","--manifest",source,"--run-root",args.derived/"full"],gpu=True)
        wait(face)
        execute("data_evidence","emotioncore.x04.data_evidence",["--derived",args.derived,"--output",args.derived/"data_evidence"],cpu=True)
        for encoder in ("V2","V1"):
            execute("full_"+encoder,"emotioncore.x04.features",["--stage",encoder,"--manifest",source,"--run-root",args.derived/"full"],gpu=True)
            heads(encoder)
        rows=[json.loads(s) for s in source.read_text().splitlines()]
        pack(rows,args.derived/"full/features/A3/receipts","A3",args.derived/"packs/A3")
        execute("heads_D0_A3_first_grid","emotioncore.x04.heads",a3_options("D0",first=True),gpu=True)
        execute("first_grid_budget","emotioncore.x04.head_budget",[
            "--derived",args.derived,"--dedup",dedup,
            "--first-grid",args.derived/"heads/D0/A3/first_grid_complete.json",
            "--budget",args.budget,"--charged-gpu-hours",charged/3600,
            "--wall-elapsed-hours",(time.monotonic()-started_pipeline)/3600,
            "--cpu-remaining-hours",0,"--cpu-elapsed-hours",min((time.monotonic()-job[3][2])/3600 for job in cpu_jobs),
            "--output",args.output/"first_grid_budget"],cpu=True)
        progress["first_grid_budget_sha256"]=sha256(args.output/"first_grid_budget/wall_time_estimate.json")
        # The budget child writes REPORT section12 before the next GPU head.
        for arm in ("D0","D1","D2","D3"):
            execute(f"heads_{arm}_A3","emotioncore.x04.heads",a3_options(arm),gpu=True)
            execute(f"targets_{arm}_A3","emotioncore.x04.targets",["--arm",arm,"--encoder","A3",
                "--job",args.derived/f"heads/{arm}/A3","--derived",args.derived,
                "--output",args.derived/f"heads/{arm}/A3/targets"],cpu=True)
        if json.loads(dedup.read_text())["fingerprint_available"]:
            execute("heads_D1_audioonly_A3","emotioncore.x04.heads",a3_options("D1_audioonly",sensitivity=True),gpu=True)
            execute("targets_D1_audioonly_A3","emotioncore.x04.targets",["--arm","D1","--encoder","A3",
                "--job",args.derived/"heads/D1_audioonly/A3","--derived",args.derived,
                "--output",args.derived/"heads/D1_audioonly/A3/targets"],cpu=True)
        execute("q_media","emotioncore.x04.q_mer",["--action","media","--derived",args.derived],cpu=True)
        execute("q_cache","emotioncore.x04.q_mer",["--action","cache","--derived",args.derived],cpu=True)
        execute("x04b_budget_only","emotioncore.x04.budget",["--derived",args.derived,"--output",args.derived/"budget_estimate"],cpu=True)
        execute("q_mer_predict","emotioncore.x04.q_mer",["--action","predict","--derived",args.derived,"--output",args.derived/"Q_predictions"],gpu=True)
        for arm,encoder,output,item in cpu_jobs:
            wait(item)
            # Sensitivity also reports frozen targets; it is not a new recipe.
            execute(f"targets_{arm}_{encoder}","emotioncore.x04.targets",["--arm","D1" if arm=="D1_audioonly" else arm,
                "--encoder",encoder,"--job",output,"--derived",args.derived,"--output",output/"targets"],cpu=True)
        execute("report_analysis","emotioncore.x04.analysis",["--derived",args.derived,"--output",args.derived/"analysis"],cpu=True)
        progress.update(status="analysis_complete",ended_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json_atomic(args.output/"pipeline_progress.json",progress)
    except Exception as exc:
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        progress.update(status="stopped",error=f"{type(exc).__name__}: {exc}",at=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json_atomic(args.output/"pipeline_stop.json",progress);raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--derived",type=Path,required=True);p.add_argument("--budget",type=Path,required=True)
    p.add_argument("--preflight",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior-gpu-hours",type=float,required=True)
    run(p.parse_args())


if __name__=="__main__":main()
