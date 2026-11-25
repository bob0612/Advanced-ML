#!/usr/bin/env bash
# Validation-only sweep for prompt-variant LoRA data. Run from the setup working directory after server_setup.sh.
set -uo pipefail
W=$(pwd); E=repo/MP3/experiments/eval_val.sh; S='{"n":6,"temperature":1.0,"reserve":6}'
cd MP3_SQL_student_package && source .venv/bin/activate && export PYTHONPATH="$PWD:$PWD/submission"
mkdir -p $W/runs
[ -f $W/runs/lora-train.jsonl ] || python submission/tools/build_lora_data.py --config submission/config.json --split train --output $W/runs/lora-train.jsonl
[ -f $W/runs/lora-train-v3.jsonl ] || python submission/tools/build_lora_data.py --config submission/config.json --split train --variants 3 --output $W/runs/lora-train-v3.jsonl
train() { [ -f $W/runs/$1/adapter/adapter_model.safetensors ] || python submission/tools/train_lora.py --base $W/qwen3-0.6b --output $W/runs/$1/adapter "${@:2}" > $W/runs/$1.out 2>&1; }
cd $W
(cd MP3_SQL_student_package && train v3e2 --data $W/runs/lora-train-v3.jsonl --epochs 2); bash $E v3e2 v3e2 "$S"
(cd MP3_SQL_student_package && train v3e3 --data $W/runs/lora-train-v3.jsonl --epochs 3); bash $E v3e3 v3e3 "$S"
(cd MP3_SQL_student_package && train r16b --data $W/runs/lora-train.jsonl); bash $E r16b r16b "$S"
echo SWEEP DONE
