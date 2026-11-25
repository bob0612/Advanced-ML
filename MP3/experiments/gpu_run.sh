#!/usr/bin/env bash
# LoRA training + validation evaluation on a GPU server, after server_setup.sh.
# Usage (from the setup working directory): bash repo/MP3/experiments/gpu_run.sh <tag> [train_lora args...]
# Uses TRAIN for training and VALIDATION for evaluation only; never touches the test split.
set -euo pipefail
WORK=$(pwd)
TAG=${1:-r16}; shift || true
cd MP3_SQL_student_package
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/submission"
OUT="$WORK/runs/$TAG"; mkdir -p "$OUT"
[ -f "$WORK/runs/lora-train.jsonl" ] || python submission/tools/build_lora_data.py --config submission/config.json --split train --output "$WORK/runs/lora-train.jsonl"
[ -f "$OUT/adapter/adapter_model.safetensors" ] || python submission/tools/train_lora.py --base "$WORK/qwen3-0.6b" --data "$WORK/runs/lora-train.jsonl" --output "$OUT/adapter" "$@" | tee "$OUT/train.out"
# validation config: same as the submitted one plus an enabled llm section
python - "$OUT/config-llm.json" <<'PY'
import json, sys
c = json.load(open("submission/config.json"))
c["llm"] = {"enabled": True, "n": 4, "reserve": 3, "max_new_tokens": 256}
json.dump(c, open(sys.argv[1], "w"), indent=1)
PY
export MP3_BASE_MODEL="$WORK/qwen3-0.6b" MP3_ADAPTER_DIR="$OUT/adapter" MP3_LLM_DEVICE=cuda MP3_LLM_LOG="$OUT/llm-log-val.jsonl"
python -m mp3_sql.predict --split validation --method submission/submission.py --checkpoint "$OUT/config-llm.json" --workers 1 --output "$OUT/val.json"
python -m mp3_sql.score --split validation --predictions "$OUT/val.json" --output "$OUT/val-score.json"
python -c "import json;s=json.load(open('$OUT/val-score.json'));print({k:v for k,v in s.items() if not isinstance(v,(list,dict))})"
echo "RUN DONE $TAG"
