# emotioncore

X01 registers Chinese emotion data, freezes Benchmark v1, audits leakage, and re-evaluates the existing Qwen2.5-Omni-7B base and checkpoint-5265 adapter. It does not train a model or run inference on the CH-SIMS v2 test lock-box.

## Server paths and environment

Edit only `configs/paths.server.yaml` for a different host. All absolute paths, including the read-only legacy project, source datasets, processed media, model, adapter, derived-data root, and X01 run root, are there. `configs/x01.yaml` freezes media limits and evaluation parameters.

Use Python 3.11 in a new repository-local environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

The lock was copied from the legacy `preflight_v2/requirements.actual.lock.txt` and must be compared against the read-only legacy environment before GPU use. On the X01 server, the new `.venv` was created for audit and tests. Until the full CUDA stack is installed there, the documented temporary inference interpreter is `/home/administrator/projects/emotion/.venv/bin/python`; no package is installed into that old environment.

## Reproduce X01

From the repository root, one command runs extraction (if needed), both manifest builders, benchmark split generation, four-level leak checks, D3, all D2 conditions with resume, and metric analysis:

```bash
PYTHONPATH=src .venv/bin/python -m emotioncore.eval.run_all --paths configs/paths.server.yaml --config configs/x01.yaml
```

Use the fully provisioned `.venv` for this command. If its CUDA dependencies are not yet available on the server, the same command can temporarily use the read-only legacy interpreter after changing only the interpreter path in the shell command. Add `--with-face-audit` to run the CPU YuNet audit after D2. The face audit can also run independently with `python -m emotioncore.data.face_audit --config configs/paths.server.yaml --results experiments/X01_benchmark/results --output experiments/X01_benchmark/results/face_audit.json`.

The D2 runner refuses to start unless D3 passed all 40 comparisons and ET-test/ET-val are clean at all four leakage levels. Each of the 16 model × dataset × condition runs resumes by sample ID, writes each prediction atomically under the server X01 run root, and stores its own `run_meta.json`. The repository contains only transcript-free manifests and aggregate results; per-sample predictions remain on the server.

## Focused commands

```bash
PYTHONPATH=src .venv/bin/python -m emotioncore.tools.leakage_check --train TRAIN_AV.jsonl TRAIN_AUDIO.jsonl --eval EVAL.jsonl --output leakage.json
PYTHONPATH=src .venv/bin/python -m pytest -q
```

See `docs/BENCHMARK_v1.md` for frozen cohorts, labels, and metrics; data cards for audit detail; and `experiments/X01_benchmark/REPORT.md` for results and deviations.
