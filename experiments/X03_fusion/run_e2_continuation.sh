#!/usr/bin/env bash
# Continue the frozen E2 CPU run after its five outer folds finish.
set -euo pipefail

R2=/home/administrator/emotion-data/emotioncore_runs/X02
R3=/home/administrator/emotion-data/emotioncore_runs/X03
PYTHON="$R2/venv/bin/python"
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "$R3/code"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "E2 continuation requires a clean server worktree" >&2
  exit 2
fi
echo "E2 continuation git HEAD: $(git rev-parse HEAD)"
echo "Waiting for the five frozen E2 CV folds"
until [[ -s "$R3/E2/cv/results.json" ]]; do
  if ! kill -0 951064 2>/dev/null; then
    echo "E2 CV process exited before writing results.json" >&2
    exit 3
  fi
  sleep 60
done
echo "E2 CV completed at $(date -Is)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Server worktree changed while E2 CV ran" >&2
  exit 4
fi

COMMON=(
  --manifest experiments/X01_benchmark/results/et_utterances.jsonl
  --folds configs/folds_v1.json
  --a3-pack "$R2/packs/A3"
  --v2-pack "$R2/packs/V2"
  --t1-tokens "$R2/tokens/T1_r_cv.npz"
  --t1-cls "$R3/E2/cls_cv/T1_r_cv_cls.npy"
  --t1-cls-meta "$R3/E2/cls_cv/cls_run_meta.json"
  --face-policy configs/x03_face_missing_policy.json
  --cv-results "$R3/E2/cv/results.json"
  --output-dir "$R3/E2/final"
  --device cpu --batch-size 32 --threads 8
)

echo "Training the frozen full-CV E2 head and evaluating ET-test"
timeout 7200 "$PYTHON" -u -m emotioncore.x03.e2_target evaluate \
  "${COMMON[@]}" \
  --target-manifest experiments/X01_benchmark/results/et_utterances.jsonl \
  --target-pack "$R3/E2/target_et" \
  --target-tokens "$R2/tokens/T1_r_test.npz" \
  --target-cls "$R3/E2/cls_test/T1_r_test_cls.npy" \
  --target-cls-meta "$R3/E2/cls_test/cls_run_meta.json" \
  --split et_test
echo "E2 ET-test complete at $(date -Is)"

echo "Evaluating only the frozen SIMS-valid subset"
timeout 7200 "$PYTHON" -u -m emotioncore.x03.e2_target evaluate \
  "${COMMON[@]}" \
  --target-manifest "$R2/sims_valid/sims_valid_adapter.jsonl" \
  --target-pack "$R3/E2/target_sims_valid" \
  --target-tokens "$R3/T1r_sims_valid/tokens/T1_r_sims_valid.npz" \
  --target-cls "$R3/E2/cls_sims_valid/T1_r_sims_valid_cls.npy" \
  --target-cls-meta "$R3/E2/cls_sims_valid/cls_run_meta.json" \
  --split sims_valid
echo "E2 SIMS-valid complete at $(date -Is)"

echo "Cross-fitting calibration and scoring E2"
timeout 7200 "$PYTHON" -u -m emotioncore.x03.e2_metrics \
  --manifest experiments/X01_benchmark/results/et_utterances.jsonl \
  --folds configs/folds_v1.json \
  --cv-dir "$R3/E2/cv" \
  --target-dir "$R3/E2/final" \
  --sims-dir "$R3/E2/final" \
  --artifact-index experiments/X02_unimodal/results/artifact_index.json \
  --output-dir "$R3/E2/eval"
echo "E2 calibrated metrics complete at $(date -Is)"

echo "Measuring serialized steady-state ET-test latency"
timeout 7200 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PYTHON" -u -m emotioncore.x03.e2_latency \
  --manifest experiments/X01_benchmark/results/et_utterances.jsonl \
  --face-policy configs/x03_face_missing_policy.json \
  --target-pack "$R3/E2/target_et" \
  --t1-tokens "$R2/tokens/T1_r_test.npz" \
  --t1-cls "$R3/E2/cls_test/T1_r_test_cls.npy" \
  --t1-cls-meta "$R3/E2/cls_test/cls_run_meta.json" \
  --t1-model "$R2/models/T1" \
  --checkpoint "$R3/E2/final/final_model.pt" \
  --et-meta "$R3/E2/final/et_test_run_meta.json" \
  --face-receipts "$R2/face/receipts" \
  --a3-receipts "$R2/features/A3/receipts" \
  --v2-receipts "$R2/features/V2/receipts" \
  --asr-receipts "$R2/asr/SenseVoiceSmall/receipts" \
  --output-dir "$R3/E2/latency" \
  --threads 4
echo "E2 latency complete at $(date -Is)"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path('/home/administrator/emotion-data/emotioncore_runs/X03/E2')
for condition in ('AVT', 'A', 'V', 'AV'):
    path = root / 'eval' / condition / 'metrics.json'
    metrics = json.loads(path.read_text())
    print(condition, 'CV OP-N macro-F1:', metrics['cv']['op_n']['macro_f1'],
          'ET OP-N macro-F1:', metrics['et_test']['op_n']['macro_f1'],
          'SIMS Pearson:', metrics['sims_valid']['pearson'])
latency = json.loads((root / 'latency' / 'summary.json').read_text())
print('E2 serial ET latency:', latency['stage_statistics']['serial_total'])
PY
