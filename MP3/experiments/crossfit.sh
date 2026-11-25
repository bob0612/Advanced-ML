#!/usr/bin/env bash
# Cross-fitting on TRAIN (never test): 5 db-grouped folds; per fold, fit the ranker and the LoRA adapter without
# the fold's tasks, then run the full agent (submitted llm settings) on the fold. Run from the setup working dir.
set -uo pipefail
W=$(pwd); CF=$W/runs/cf; mkdir -p $CF
cd MP3_SQL_student_package && source .venv/bin/activate && export PYTHONPATH="$PWD:$PWD/submission"
python - $CF <<'PY2'
import json, sys, numpy as np
from mp3_sql.common import load_bank
rows = load_bank("train")[0]
dbs = sorted({r["db_id"] for r in rows}); perm = np.random.RandomState(0).permutation(len(dbs))
fold = {dbs[i]: k % 5 for k, i in enumerate(perm)}
for k in range(5):
    json.dump([r["id"] for r in rows if fold[r["db_id"]] == k], open(f"{sys.argv[1]}/fold{k}.json", "w"))
PY2
for k in 0 1 2 3 4; do
  D=$CF/f$k; mkdir -p $D
  [ -f $D/config.json ] || { python submission/tools/train_ranker.py --exclude-ids $CF/fold$k.json --output $D/config.json > $D/ranker.out 2>&1 &&
    python - $D/config.json <<'PY2'
import json, sys
c = json.load(open(sys.argv[1])); s = json.load(open("submission/config.json"))
c["llm"] = s["llm"]; json.dump(c, open(sys.argv[1], "w"), indent=1)
PY2
  }
  [ -f $D/adapter/adapter_model.safetensors ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python submission/tools/train_lora.py --micro 1 \
     --base $W/qwen3-0.6b --data $W/runs/lora-train-v3.jsonl --epochs 2 --exclude-ids $CF/fold$k.json --output $D/adapter > $D/lora.out 2>&1
  MP3_BASE_MODEL=$W/qwen3-0.6b MP3_ADAPTER_DIR=$D/adapter MP3_LLM_DEVICE=cuda MP3_LLM_LOG=$D/llm-log.jsonl \
    python $W/repo/MP3/experiments/crossfit_run.py --split train --ids $CF/fold$k.json --checkpoint $D/config.json --output $D/run.jsonl 2> $D/run.err
  echo "fold $k done $(python -c "import json;L=[json.loads(l) for l in open('$D/run.jsonl')];print(sum(r['passed'] for r in L),len(L))")" | tee -a $CF/summary.txt
done
echo CF DONE | tee -a $CF/summary.txt
