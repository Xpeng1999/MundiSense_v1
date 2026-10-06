"""Sequential dedup and mandatory 200-MER GPU budget gate before full extraction."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from emotioncore.x02.face import sha256
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.prepare import storage_gate, write_rows


def run(args):
    git=git_provenance(Path.cwd());storage_gate()
    args.output.mkdir(parents=True,exist_ok=True)
    write_json_atomic(args.output/"run_meta.json",{"git":git,"command":sys.argv,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),"full_run_authorized_only_after_budget_pass":True})
    def execute(module, *opts):
        subprocess.run([sys.executable,"-u","-m",module,*map(str,opts)],check=True)
    execute("emotioncore.x04.dedup","--derived",args.derived,"--output",args.output/"dedup","--sims-media","/home/administrator/emotion-data/emotioncore_derived/chsims_v2/raw/Raw")
    registration=json.loads(args.registration.read_text())
    if any(c["audio_probe_status"].get("missing",0) or c["video_probe_status"].get("missing",0) for c in registration["cohorts"].values()):
        raise RuntimeError("Registered source files missing; no benchmark permitted")
    bench=Path(registration["benchmark200"]["path"])
    if sha256(bench)!=registration["benchmark200"]["sha256"]:raise ValueError("200-MER selection changed")
    rows=[json.loads(s) for s in bench.read_text().splitlines()]
    if len(rows)!=200:raise ValueError("Budget benchmark must contain exactly 200 MER samples")
    all_rows=[]
    for name in ("mer_train","mer_test","crema_train"):
        all_rows.extend(json.loads(s) for s in (args.derived/f"manifests/{name}.jsonl").read_text().splitlines())
    write_rows(args.derived/"manifests/all_features.jsonl",all_rows)
    timings={}
    benchmark_root = args.benchmark_root or args.derived/"benchmark200"
    for stage in ("face","A3","V2","V1"):
        storage_gate()
        execute("emotioncore.x04.features","--stage",stage,"--manifest",bench,"--run-root",benchmark_root)
        path=benchmark_root/("face/timing.json" if stage=="face" else f"features/{stage}/run_meta.json")
        timings[stage]=json.loads(path.read_text())
    if timings["A3"]["status"]!={"ok":200}:
        raise RuntimeError("A3 benchmark contains failed outputs; repair before estimating budget")
    q_path=Path("/home/administrator/emotion-data/emotioncore_runs/X03/Q/predict_final_et_test/run_meta.json")
    q=json.loads(q_path.read_text());q_seconds=(dt.datetime.fromisoformat(q["ended_at"])-dt.datetime.fromisoformat(q["started_at"])).total_seconds()
    measured_gpu=sum(timings[e]["gpu_wall_seconds"] for e in ("A3","V2","V1"))
    extraction_projection=measured_gpu*len(all_rows)/200
    q_projection=q_seconds*572/1929  # conservatively retains X03's V condition too.
    load_reserve=600.
    projection=(extraction_projection+q_projection)*1.2+load_reserve+measured_gpu
    budget={"benchmark_n":200,"benchmark_manifest_sha256":sha256(bench),"total_feature_samples":len(all_rows),
        "all_features_manifest_sha256":sha256(args.derived/"manifests/all_features.jsonl"),
        "dedup_report_sha256":sha256(args.output/"dedup/dedup_report.json"),
        "timings":{e:{"gpu_seconds":timings[e].get("gpu_wall_seconds",0),"cpu_seconds":timings[e].get("cpu_wall_seconds",0),"status":timings[e].get("status")} for e in timings},
        "head_device":"cpu","head_gpu_hours":0,"measured_gpu_hours":measured_gpu/3600,
        "extraction_projection_hours":extraction_projection/3600,"Q_mer_test_projection_hours":q_projection/3600,
        "Q_projection_source":str(q_path),"Q_projection_source_sha256":sha256(q_path),
        "reserve_multiplier":1.2,"load_reserve_hours":load_reserve/3600,
        "total_projection_gpu_hours":projection/3600,"limit_gpu_hours":20,"passed":projection<=20*3600,
        "CPU_face_projection_hours":timings["face"]["cpu_wall_seconds"]*len(all_rows)/200/3600,
        "fingerprint_sensitivity_training_device":"cpu","CH_SIMS_model_inference":0}
    write_json_atomic(args.output/"budget_preflight.json",budget)
    if not budget["passed"]:
        write_json_atomic(args.output/"STOP.json",{"reason":"Projected GPU usage exceeds 20h","budget":budget})
        raise RuntimeError("STOP GPU budget preflight")
    write_json_atomic(args.output/"PASS.json",{"budget_sha256":sha256(args.output/"budget_preflight.json"),"at":dt.datetime.now(dt.timezone.utc).isoformat()})


def main():
    p=argparse.ArgumentParser();p.add_argument("--derived",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--registration",type=Path,required=True)
    p.add_argument("--benchmark-root",type=Path)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        write_json_atomic(args.output/"failure.json",{"type":type(exc).__name__,"reason":str(exc),"at":dt.datetime.now(dt.timezone.utc).isoformat()});raise


if __name__=="__main__":main()
