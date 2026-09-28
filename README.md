# emotioncore

X01 registers Chinese emotion data, freezes Benchmark v1, audits leakage, and re-evaluates the existing Qwen2.5-Omni-7B base and checkpoint-5265 adapter. It does not train a model or run inference on the CH-SIMS v2 test lock-box.

## Server paths and environment

Edit only `configs/paths.server.yaml` for a different host. All absolute paths, including the read-only legacy project, source datasets, processed media, model, adapter, derived-data root, and X01 run root, are there. `configs/x01.yaml` freezes media limits and evaluation parameters.

On the X01 server, reproduce the independent Python 3.11 environment from the audited legacy package tree without writing to the legacy directory:

```bash
python3.11 -m venv .venv
legacy_root=$(awk '$1 == "legacy_root:" {print $2}' configs/paths.server.yaml)
rsync -a --exclude='__pycache__/' "$legacy_root/.venv/lib/python3.11/site-packages/" .venv/lib/python3.11/site-packages/
python3 - <<'PY'
from pathlib import Path
import shutil
for metadata in Path('.venv/lib/python3.11/site-packages').glob('setuptools-*.dist-info'):
    if metadata.name != 'setuptools-78.1.0.dist-info':
        shutil.rmtree(metadata)
PY
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m emotioncore.utils.environment \
  --paths configs/paths.server.yaml --verify-current \
  --output experiments/X01_benchmark/results/new_venv_verification.json
```

The lock was copied from the legacy `preflight_v2/requirements.actual.lock.txt`. `new_venv_verification.json` records zero version mismatches across all 296 locked packages, the separate interpreter and package path, and successful CUDA detection on the RTX 5090. The copied package set contains a pre-existing `gradio`/`starlette` constraint conflict, so dependency resolving `pip install -r requirements.lock.txt` is not the verified setup path. No package was installed into the old environment. The running X01 D2 evaluation was started consistently with the legacy interpreter before the new environment was fully provisioned; a fresh run can use the new `.venv`.

## Reproduce X01

From the repository root, one command runs extraction (if needed), both manifest builders, benchmark split generation, four-level leak checks, D3, all D2 conditions with resume, and metric analysis:

```bash
PYTHONPATH=src .venv/bin/python -m emotioncore.eval.run_all --paths configs/paths.server.yaml --config configs/x01.yaml
```

Add `--with-face-audit` to run the CPU YuNet audit after D2. The face audit can also run independently with `PYTHONPATH=src .venv/bin/python -m emotioncore.data.face_audit --config configs/paths.server.yaml --results experiments/X01_benchmark/results --output experiments/X01_benchmark/results/face_audit.json`.

The D2 runner refuses to start unless D3 passed all 40 comparisons and ET-test/ET-val are clean at all four leakage levels. Each of the 16 model × dataset × condition runs resumes by sample ID, writes each prediction atomically under the server X01 run root, and stores its own `run_meta.json`. The repository contains only transcript-free manifests and aggregate results; per-sample predictions remain on the server.

The face audit also resumes from atomic per-clip receipts under the server X01 run root; its repository JSON contains aggregate frame and speaker counts. The CH-SIMS media audit fully decodes present audio/video streams to a null sink, including test lock-box media, without running model inference on test.

## Focused commands

```bash
PYTHONPATH=src .venv/bin/python -m emotioncore.tools.leakage_check --train TRAIN_AV.jsonl TRAIN_AUDIO.jsonl --eval EVAL.jsonl --output leakage.json
PYTHONPATH=src .venv/bin/python -m pytest -q
```

See `docs/BENCHMARK_v1.md` for frozen cohorts, labels, and metrics; data cards for audit detail; and `experiments/X01_benchmark/REPORT.md` for results and deviations.
