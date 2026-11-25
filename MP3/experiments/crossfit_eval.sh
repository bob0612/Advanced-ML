#!/usr/bin/env bash
# Re-run the agent on the 5 cross-fitting folds (TRAIN only) with llm overrides, folds in parallel.
# Usage (setup working dir): bash repo/MP3/experiments/crossfit_eval.sh <name> '<llm json overrides>'
set -uo pipefail
W=$(pwd); NAME=$1; LLM=$2; CF=$W/runs/cf
cd MP3_SQL_student_package && source .venv/bin/activate && export PYTHONPATH="$PWD:$PWD/submission"
for k in 0 1 2 3 4; do
  D=$CF/f$k; O=$D/$NAME; mkdir -p $O
  python - $D/config.json $O/config.json "$LLM" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["llm"] = {**c["llm"], **json.loads(sys.argv[3])}
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
  MP3_BASE_MODEL=$W/qwen3-0.6b MP3_ADAPTER_DIR=$D/adapter MP3_LLM_DEVICE=cuda MP3_LLM_LOG=$O/llm-log.jsonl OMP_NUM_THREADS=4 \
    python $W/repo/MP3/experiments/crossfit_run.py --split train --ids $CF/fold$k.json --checkpoint $O/config.json --output $O/run.jsonl 2> $O/run.err &
done
wait
python - $CF $NAME <<'PY' | tee -a $CF/summary.txt
import json, sys
L = [json.loads(l) for k in range(5) for l in open(f"{sys.argv[1]}/f{k}/{sys.argv[2]}/run.jsonl")]
print(sys.argv[2], sum(r["passed"] for r in L), len(L))
PY
