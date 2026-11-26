#!/usr/bin/env bash
# One-time setup on a Linux GPU server (run from an empty working directory, e.g. ~/mp3).
set -euo pipefail
WORK=$(pwd)
[ -d repo ] || git clone -b mp3-llm https://github.com/bob0612/Advanced-ML repo
if [ ! -d MP3_SQL_student_package ]; then
  curl -fsSL -o pkg.zip "https://xudongwu-0.github.io/courses/dase7506/mp3/downloads/MP3_SQL_student_package.zip?v=406d69507ccf"
  echo "406d69507ccf955a53b35db1c1368bd7452565ebd928b47d0f6161fe35b99a61  pkg.zip" | sha256sum -c -
  unzip -q pkg.zip
fi
cd MP3_SQL_student_package
ln -sfn "$WORK/repo/MP3/submission" submission
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q -r requirements.txt -r requirements-model.txt -r submission/requirements-train.txt
python -m mp3_sql.check_data
MODEL_DIR="$WORK/qwen3-0.6b"
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  python - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca", local_dir=sys.argv[1],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"])
PY
fi
python -c "import torch;print('cuda', torch.cuda.is_available(), torch.__version__)"
echo "SETUP OK; model at $MODEL_DIR"
