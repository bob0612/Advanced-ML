# DASE7506 MP3: feedback-driven SQL repair agent with a LoRA proposer

**Official test result: 187/240 = 77.92%** (`outputs/test-score.json`), in the graded live-LLM configuration: `Qwen/Qwen3-0.6B` + our LoRA adapter, CPU only, 4 threads, `--workers 1`. Validation: 101/120 = 84.17%.
Without a local model the same export runs as the CPU-only controller: 182/240 on test, 96/120 on validation.
Report: [`report.pdf`](report.pdf).

## Contents
| Path | Purpose |
|---|---|
| `submission.py` | entry point: `create_agent(checkpoint)` returns an agent with `act(observation)` |
| `mp3agent/agent.py` | the perceive-plan-act controller (check draft, read hints, check candidates, ask the LoRA model when nothing matches, submit) |
| `mp3agent/candidates.py`, `mp3agent/sqlops.py` | candidate pool and the sqlglot rewrites (align, complete, dropjoin, DISTINCT toggles, COUNT(*), argmax, intersect, capital) |
| `mp3agent/features.py` | candidate features (static and from check feedback) |
| `mp3agent/llm.py`, `mp3agent/prompting.py` | offline loading of the base model + adapter, generation (≤ 6 completions per task, ≤ 256 new tokens each) and logging; prompt construction and SQL extraction |
| `config.json` | learned prior and ranker weights (train split only) and the validation-selected `llm` settings |
| `tools/train_ranker.py` | regenerates the ranker part of `config.json` from the train split |
| `tools/build_lora_data.py`, `tools/train_lora.py` | LoRA training data (train split only) and LoRA training (only A/B matrices trainable) |
| `requirements.txt` | run-time dependency (same pin as the official package); the live model also needs the official `requirements-model.txt` |
| `requirements-train.txt` | extra dependencies for ranker retraining only |
| `report/` | report source (Markdown, figure, CSS); rendered to `report.pdf` with pandoc and Chromium |

The LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`) is submitted separately as `mp3-lora-adapter.zip` (37 MB).

## Reproduce (Linux, Python 3.12, offline, 4 CPU threads)
Extract this ZIP inside the official `MP3_SQL_student_package/` directory, so that `submission/` and `outputs/` sit next to `mp3_sql/`.
```bash
cd MP3_SQL_student_package
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-model.txt
python -m mp3_sql.check_data
# provision the approved base model once (pinned revision), then work offline
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='qwen3-0.6b')"
unzip ../mp3-lora-adapter.zip -d lora-adapter
export MP3_BASE_MODEL=$PWD/qwen3-0.6b MP3_ADAPTER_DIR=$PWD/lora-adapter MP3_LLM_DEVICE=cpu \
       OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 MP3_LLM_LOG=outputs/llm-log-test.jsonl
PYTHONPATH="$PWD/submission" taskset -c 0-3 python -m mp3_sql.predict --split test --workers 1 \
  --method submission/submission.py --checkpoint submission/config.json --output outputs/test.json
python -m mp3_sql.score --split test --predictions outputs/test.json --output outputs/test-score.json
```
Expected: `"correct": 187`, `"n": 240`. Measured: 11 min wall, peak RSS 5.0 GB (the model is called on 29 of 240 tasks). Sampling uses fixed per-task seeds, so the run repeats given the same library versions and thread count.

Without `MP3_BASE_MODEL` the agent prints one warning and runs the CPU-only controller (182/240, about 32 s, no model dependencies needed).

Optional retraining (train split only):
```bash
python -m pip install -r submission/requirements-train.txt
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/train_ranker.py --output submission/config.json
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/build_lora_data.py --config submission/config.json --split train --variants 3 --output lora-train.jsonl
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/train_lora.py --base qwen3-0.6b --data lora-train.jsonl --output lora-adapter --epochs 2  # GPU recommended
```

## Disclosure
The only live model is `Qwen/Qwen3-0.6B` (revision `c1899de2…`) with our LoRA adapter, trained on the train split only and loaded from local files (`local_files_only`, safetensors, no remote code). Every prompt and completion is logged with token counts and model/adapter identity (`outputs/llm-log-*.jsonl`). The `repair_hint` tool reads the course's fixed hint cache. LoRA training and the validation sweeps used rented GPUs (RunPod: RTX 4090, then A40). AI coding assistance (Claude, Anthropic) was used substantially to write the code and the report. Data: only the official Spider-derived package (CC BY-SA 4.0).
