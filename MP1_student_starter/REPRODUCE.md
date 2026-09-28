# Reproduce MP1 (student 3036797441)

The submitted predictor is **B0: the original GPT, seed 17, final raw weights**.
EMA was implemented and tested, but its validation BPB was worse in both predeclared
seeds. The negative result and paired ablation are part of the report.

Expected full-test CPU FP32 BPB: **2.1012651958438786**.
Required evaluation checkpoint: **`models/final.pt`**. Its SHA-256 is
`5bb600b4dc65e8a8fa0eaa6841955b99754bbb61b060fb626d0855940d26206a`.
The exact published weights, rather than a new training run, define the submission.

## 1. Obtain the immutable code and matching bundle

Use the commit-specific code URL in the course submission. Download that commit's
repository ZIP, or check out the exact commit with Git. The matching model bundle is
`MP1_student_starter/submission/model-bundle.zip` in the same commit; it is a regular
Git file, not a Git LFS pointer. The course checkpoint link points directly to that
file at the fixed commit.

From the repository root:

```bash
cd MP1_student_starter/code
```

All subsequent model commands run from **this `code/` directory**. No dataset or
pretrained model downloads are needed. After dependency installation, evaluation
works offline. `common.py` checks the included data and tokenizer hashes.

## 2. Install

Use Python **3.12** (measured: 3.12.14). Create a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

macOS, exact recorded dependency versions:

```bash
python -m pip install -r requirements.macos-py312.lock.txt
```

Linux/Windows CPU (select this instead of the macOS installation):

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

Windows activation is `.venv\Scripts\Activate.ps1`. The measured local environment
was macOS 27.0 arm64 on an Apple M5 (10 cores, 16 GiB RAM). Linux/Windows installation
commands follow the supplied starter; they were not independently benchmarked here.
Do not use BF16 for ranked evaluation. No CUDA or Apple MPS is needed.

## 3. Verify and evaluate without retraining

```bash
python -m unittest discover -s tests -v
python -m zipfile -e ../submission/model-bundle.zip models
python -c "import hashlib; from pathlib import Path; p=Path('models/final.pt'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='5bb600b4dc65e8a8fa0eaa6841955b99754bbb61b060fb626d0855940d26206a'; print('Checkpoint SHA-256 verified')"
python evaluate.py --checkpoint models/final.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

Expect **18 tests passing**, protocol `7506-mp1-wt2-v2`, `428405` targets,
`1292013` UTF-8 bytes, and BPB **2.1012651958438786**. Read the complete `bpb`
field in `reproduced-test.json`; do not substitute token perplexity or validation BPB.
The local clean-directory reproduction and repeated timing runs are retained in
`../submission/evidence/`. Cross-platform floating-point differences can be small;
report the actual reproduced value and environment rather than editing the JSON.

`models/MANIFEST.json` records every checkpoint hash. `baseline.pt` and `final.pt`
are byte-identical because validation selected the original baseline. Optional
`ema-seed17.pt`, `raw-seed23.pt`, and `ema-seed23.pt` let readers verify the negative
result without retraining. To compare them, use **validation**, with distinct output
names, for example:

```bash
python evaluate.py --checkpoint models/ema-seed17.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced-ema17-validation.json
python evaluate.py --checkpoint models/raw-seed23.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced-raw23-validation.json
python evaluate.py --checkpoint models/ema-seed23.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced-ema23-validation.json
```

## 4. Reproduce training and the paired ablation (optional)

All runs start from random initialization; none loads a parent checkpoint. A new,
empty run directory is required each time. Training is not required for scoring.

```bash
# Initial baseline: 1,200 x 32 x 256 = 9,830,400 processed targets
python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 300 --run-dir runs/reproduce-baseline-s17

# Paired raw/EMA export, original predeclared setting
python train.py --implementation student --device cpu --precision fp32 --threads 4 --seed 17 --steps 1200 --batch-size 32 --eval-every 300 --ema-decay 0.99 --ema-start 600 --run-dir runs/reproduce-ema-s17

# Predeclared second seed; same recipe and budget
python train.py --implementation student --device cpu --precision fp32 --threads 4 --seed 23 --steps 1200 --batch-size 32 --eval-every 300 --ema-decay 0.99 --ema-start 600 --run-dir runs/reproduce-ema-s23
```

`checkpoint.pt` is the raw final state. `checkpoint-ema.pt` is an independent
average initialized **after optimizer step 600**, followed by **600** averaging
updates at steps 601–1200. Both checkpoints inherit the full 9,830,400 training
targets. `metrics.json` preserves both validation scores and checkpoint hashes.
The EMA helper neither updates the optimizer nor feeds averaged weights back into
training. The default `--ema-decay 0` disables averaging.

| Checkpoint | Validation BPB |
|---|---:|
| Original baseline / paired raw seed 17 | 2.0710874333793914 |
| EMA seed 17 | 2.0750514282877415 |
| Raw seed 23 | 2.0760114040227444 |
| EMA seed 23 | 2.0797434995485724 |

Original source hashes and an archived source snapshot are under
`../submission/evidence/baseline-original/`. `train.py` now has optional EMA support;
its default raw trajectory was checked against the original baseline, including all
53 state tensors after full training. Serialized file bytes can differ after a new
training run; compare model states and scores, while using the published bundle's
fixed SHA-256 to verify downloaded assets.

## 5. Resource and cost evidence

`../submission/RESOURCE_RESULTS.json` includes three alternating baseline/final
full-test repetitions, scorer seconds, process wall time, peak RSS and byte counts.
The baseline and final predictor are identical, so the measured ratio principally
reflects timing variability. To repeat one measurement on macOS:

```bash
/usr/bin/time -l python evaluate.py --checkpoint models/baseline.pt --device cpu --precision fp32 --threads 4 --split test --output baseline-timing.json
/usr/bin/time -l python evaluate.py --checkpoint models/final.pt --device cpu --precision fp32 --threads 4 --split test --output final-timing.json
```

On Linux, `/usr/bin/time -v` reports maximum RSS in KiB, whereas macOS `time -l`
reports bytes. Compare the JSON `seconds` field for the CPU-time ratio; do not mix
it with whole-process wall time. The limits are 5x baseline, 4 GiB peak RAM and
64 MiB uncompressed inference assets. `ASSET_MANIFEST.csv` includes all supplied
split files and all five bundled checkpoints in its conservative total; the
compressed ZIP size is not the asset-budget measure. The installed interpreter
and dependency environment are listed separately.

`SEARCH_COSTS.json`, `code/EXPERIMENTS.csv`, and the report disclose both successful
and aborted runs. There were five completed corpus-training trajectories (three
full and two smoke) and one aborted full attempt. Its last logged step was 200;
its exact final step and elapsed training time are unknown, so total cost bounds
are reported. The paired raw/EMA exports are not counted as two training runs.

## 6. Report and provenance

See `../report/REPORT.pdf` (at most 10 pages) and `../report/REPORT.md`. Substantive
Codex assistance and starter/data reuse are disclosed in the repository root README.
No external training text, pretrained weights, test-based tuning, evaluation network
access, or cross-window state was used.

The PDF is already included; report generation is optional and separate from the
model environment. It uses `reportlab==5.0.1`. Run `python report/build_pdf.py`
from `MP1_student_starter/`, optionally adding `--font /path/to/CJK-font.ttf` to
embed a local CJK TrueType font. The delivered PDF embeds its Chinese font.
