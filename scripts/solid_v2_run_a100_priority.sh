#!/usr/bin/env bash
set -euo pipefail

cd /data/llada_eval
source scripts/env_llada.sh

ROOT="results/domain_shift/task_aware/solid_v2/a100_clean_priority"
MODEL="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"
TASKS="${TASKS:-winogrande,commonsenseqa,arc_challenge,hellaswag,boolq}"
LIMIT="${LIMIT:-50}"
SEED="${SEED:-23}"

mkdir -p "$ROOT/raw" "$ROOT/logs" "$ROOT/tables" "$ROOT/reports"

echo "[priority] waiting for torch and full model..."
while true; do
  if ./venv/bin/python -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    if ./venv/bin/python - <<'PY'
from pathlib import Path
model = Path("/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
expected = {
    "model-00001-of-00006.safetensors": 2109747184,
    "model-00002-of-00006.safetensors": 2948598128,
    "model-00003-of-00006.safetensors": 2986883656,
    "model-00004-of-00006.safetensors": 2952797224,
    "model-00005-of-00006.safetensors": 2919239128,
    "model-00006-of-00006.safetensors": 2113931792,
}
for name, target in expected.items():
    path = model / name
    if not path.exists() or path.stat().st_size < target:
        raise SystemExit(1)
PY
    then
      break
    fi
  fi
  date
  ./venv/bin/python - <<'PY' 2>/dev/null || du -sh "$MODEL" 2>/dev/null || true
from pathlib import Path
model = Path("/data/hf/models/GSAI-ML/LLaDA-8B-Instruct")
expected = {
    "model-00001-of-00006.safetensors": 2109747184,
    "model-00002-of-00006.safetensors": 2948598128,
    "model-00003-of-00006.safetensors": 2986883656,
    "model-00004-of-00006.safetensors": 2952797224,
    "model-00005-of-00006.safetensors": 2919239128,
    "model-00006-of-00006.safetensors": 2113931792,
}
current = 0
total = sum(expected.values())
for name, target in expected.items():
    path = model / name
    current += min(path.stat().st_size if path.exists() else 0, target)
print(f"{current / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB")
PY
  sleep 60
done

echo "[priority] installing non-torch dependencies..."
./venv/bin/pip install \
  transformers==4.57.6 datasets accelerate safetensors sentencepiece protobuf peft tqdm scipy matplotlib pandas huggingface_hub \
  2>&1 | tee "$ROOT/logs/pip_deps.log"

echo "[priority] smoke test..."
./venv/bin/python - <<'PY' 2>&1 | tee "$ROOT/logs/smoke_test.log"
import torch
from transformers import AutoModel, AutoTokenizer
model_path="/data/hf/models/GSAI-ML/LLaDA-8B-Instruct"
tok=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model=AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
print("SMOKE_OK", torch.__version__, torch.cuda.get_device_name(0), len(tok))
PY

echo "[priority] run our non-toy selective re-masking refinement controller first"
./venv/bin/python scripts/eval_llada_refinement_controller.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --budgets 8,16,24,32 \
  --risk-t16 0.24 \
  --risk-t24 0.38 \
  --risk-t32 0.56 \
  --multi-disagreement-policy ignore \
  --out "$ROOT/raw/llada8b_refinement_controller_limit${LIMIT}_seed${SEED}.json" \
  2>&1 | tee "$ROOT/logs/refinement_controller_limit${LIMIT}_seed${SEED}.log"

./venv/bin/python scripts/audit_eval_outputs.py \
  "$ROOT/raw/llada8b_refinement_controller_limit${LIMIT}_seed${SEED}.json" \
  --out "$ROOT/tables/refinement_output_audit_limit${LIMIT}_seed${SEED}.csv" \
  --max-invalid-rate 0.02 \
  2>&1 | tee "$ROOT/logs/refinement_output_audit_limit${LIMIT}_seed${SEED}.log" || true

echo "[priority] run fixed-step and sampler baselines after our method"
run_sampler() {
  local name="$1"
  local steps="$2"
  local schedule="$3"
  ./venv/bin/python scripts/eval_llada_sampler_variants.py \
    --model "$MODEL" \
    --tasks "$TASKS" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --steps "$steps" \
    --gen-length 32 \
    --block-length 32 \
    --schedule "$schedule" \
    --prompt-style final_label_typed \
    --out "$ROOT/raw/${name}_limit${LIMIT}_seed${SEED}.json" \
    2>&1 | tee "$ROOT/logs/${name}_limit${LIMIT}_seed${SEED}.log"
}

run_sampler "llada8b_fixed8_back_loaded" 8 back_loaded
run_sampler "llada8b_fixed16_back_loaded" 16 back_loaded
run_sampler "llada8b_jys_like_middle16" 16 middle_heavy
run_sampler "llada8b_fixed32_uniform" 32 uniform

./venv/bin/python scripts/eval_llada_prophet_early_commit.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --max-steps 32 \
  --min-steps 8 \
  --check-interval 4 \
  --patience 2 \
  --gen-length 32 \
  --block-length 32 \
  --schedule uniform \
  --out "$ROOT/raw/llada8b_prophet_early_commit_limit${LIMIT}_seed${SEED}.json" \
  2>&1 | tee "$ROOT/logs/prophet_early_commit_limit${LIMIT}_seed${SEED}.log"

./venv/bin/python scripts/audit_eval_outputs.py "$ROOT/raw" \
  --out "$ROOT/tables/output_audit_limit${LIMIT}_seed${SEED}.csv" \
  --max-invalid-rate 0.02 \
  2>&1 | tee "$ROOT/logs/output_audit_limit${LIMIT}_seed${SEED}.log" || true

./venv/bin/python scripts/analyze_clean_retest.py \
  --root "$ROOT" \
  2>&1 | tee "$ROOT/logs/analyze_priority_limit${LIMIT}_seed${SEED}.log"

echo "[priority] done"
