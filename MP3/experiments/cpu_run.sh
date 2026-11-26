#!/usr/bin/env bash
# Graded-configuration run: CPU only, 4 threads pinned to 4 cores, --workers 1, offline model files.
# Usage: bash repo/MP3/experiments/cpu_run.sh <split> <config.json> <adapter_dir> <out_dir> <cores e.g. 0-3>
set -euo pipefail
WORK=$(pwd); SPLIT=$1; CFG=$2; ADAPTER=$3; OUT=$4; CORES=${5:-0-3}
mkdir -p "$OUT"; rm -f "$OUT/llm-log.jsonl"
cd MP3_SQL_student_package; source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/submission" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MP3_LLM_DEVICE=cpu
export MP3_BASE_MODEL="$WORK/qwen3-0.6b" MP3_ADAPTER_DIR="$ADAPTER" MP3_LLM_LOG="$OUT/llm-log.jsonl"
taskset -c "$CORES" python - "$SPLIT" "$CFG" "$OUT" <<'PY' 2> "$OUT/stderr.log"
import json, resource, runpy, sys, time
split, cfg, out = sys.argv[1:4]
sys.argv = ["predict", "--split", split, "--method", "submission/submission.py", "--checkpoint", cfg,
            "--workers", "1", "--output", f"{out}/{split}.json"]
t0, c0 = time.time(), time.process_time()
try:
    runpy.run_module("mp3_sql.predict", run_name="__main__")
finally:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    json.dump({"wall_seconds": round(time.time() - t0, 1), "cpu_seconds": round(time.process_time() - c0, 1),
               "peak_rss_mib": round(ru.ru_maxrss / 1024, 1), "threads": 4}, open(f"{out}/resources.json", "w"))
PY
python -m mp3_sql.score --split "$SPLIT" --predictions "$OUT/$SPLIT.json" --output "$OUT/$SPLIT-score.json" > /dev/null
python -c "import json;s=json.load(open('$OUT/$SPLIT-score.json'));r=json.load(open('$OUT/resources.json'));print('$SPLIT', s['correct'], s['n'], r)" | tee "$OUT/summary.txt"
