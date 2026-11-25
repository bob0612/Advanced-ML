#!/usr/bin/env bash
# Validation-only evaluation of one LLM setting with an already trained adapter.
# Usage: bash repo/MP3/experiments/eval_val.sh <name> <adapter-run-tag|none> '<llm json overrides>'
set -euo pipefail
WORK=$(pwd); NAME=$1; ADAPTER=$2; LLM=${3:-'{}'}
cd MP3_SQL_student_package; source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/submission"
OUT="$WORK/runs/eval-$NAME"; mkdir -p "$OUT"; rm -f "$OUT/llm-log-val.jsonl"
python - "$OUT/config.json" "$LLM" <<'PY'
import json, sys
c = json.load(open("submission/config.json"))
c["llm"] = {"enabled": True, "n": 4, "reserve": 3, "max_new_tokens": 256, **json.loads(sys.argv[2])}
json.dump(c, open(sys.argv[1], "w"), indent=1)
PY
export MP3_BASE_MODEL="$WORK/qwen3-0.6b" MP3_LLM_DEVICE=cuda MP3_LLM_LOG="$OUT/llm-log-val.jsonl"
if [ "$ADAPTER" = none ]; then export MP3_ADAPTER_DIR=""; else export MP3_ADAPTER_DIR="$WORK/runs/$ADAPTER/adapter"; fi
python -m mp3_sql.predict --split validation --method submission/submission.py --checkpoint "$OUT/config.json" --workers 1 --output "$OUT/val.json" 2>"$OUT/stderr.log"
python -m mp3_sql.score --split validation --predictions "$OUT/val.json" --output "$OUT/val-score.json" >/dev/null
python -c "import json;s=json.load(open('$OUT/val-score.json'));print('$NAME', s['correct'], s['n'], round(s['mean_tool_calls'],2))" | tee -a "$WORK/runs/summary.txt"
