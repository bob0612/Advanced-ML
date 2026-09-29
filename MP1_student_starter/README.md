# MP1: Causal suffix-copy language model

A 4-layer GPT (width 192, 4 heads, dropout 0.1) mixed with a stateless,
within-window suffix-copy distribution. The final neural weights average updates
5,400 and 6,000; validation selected copy weights `[0.1, 0.3, 0.5]`.
Training uses only the supplied training text and starts from random weights.

## Results

| Predictor | Validation BPB | Full-test BPB |
| --- | ---: | ---: |
| Original baseline | 2.07108743 | 2.10126520 |
| Wide GPT at 1,200 updates (equal training targets) | 1.96000081 | - |
| Selected neural weights, copy disabled | 1.71387940 | 1.74305338 |
| Final predictor | 1.65357322 | **1.66816836** |

CPU FP32, four threads: 1.444x baseline median scoring time, 1.620 GiB peak
process RSS, 10.116 MiB uncompressed inference assets. The complete score covers
428,405 targets and 1,292,013 raw UTF-8 bytes. Predictor choices were frozen
before testing; repeated final scores agree exactly.

[REPORT.pdf](REPORT.pdf) describes the method, controls, costs and limitations.
[code/results.json](code/results.json) retains measured scores, all copy-search
trials, training curves, search costs, checkpoint ancestry and frozen source hashes.

## Install and evaluate

Use Python 3.12. Recorded versions: Python 3.12.14, PyTorch 2.7.1, NumPy 2.5.3,
tokenizers 0.21.4. The original run used an Apple M5 MacBook Air with 16 GB RAM.
Extract the code and checkpoint archives into the same directory, then:

```bash
cd MP1_project/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py --checkpoint release/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

For the working folder, use `cd MP1_student_starter/code` instead. On Windows,
activate with `.venv\Scripts\Activate.ps1`. On Linux/Windows CPU, install
`torch==2.7.1` from `https://download.pytorch.org/whl/cpu` before requirements.txt.
No retraining or evaluation network access is required. Report the JSON `bpb`;
minor numerical differences can occur across hardware or library builds.

Final checkpoint SHA256:
`6de4fff40c1d5ea9b27d188a04cd063b49459fe19bba3e942f26159d9959ae6c`.

## Reproduce training

All commands below run in `code/`. Use new output directories. The recorded seed
is 17 and batch size is 32, with 256 targets per sequence. AdamW uses learning
rate 0.001, weight decay 0.1, 100-step warmup, cosine decay with a 10% floor and
gradient clipping at 1.0. No optimizer state is needed for final evaluation.

Original baseline (1,200 updates; 9,830,400 targets):

```bash
python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --run-dir runs/baseline
python select_copy.py --checkpoint runs/baseline/checkpoint.pt --output-dir runs/baseline-copy --space initial
```

Wide model (6,000 updates; 49,152,000 targets):

```bash
python train_resumable.py --run-dir runs/wide --steps 6000 --chunk-steps 600
for i in 1 2 3 4 5 6 7 8 9; do
  python train_resumable.py --run-dir runs/wide --steps 6000 --chunk-steps 600 --resume || exit 1
done
```

The shell loop is for bash/zsh. On PowerShell, run the same `--resume` command
nine times. Each chunk preserves optimizer, learning-rate schedule, sampler RNG
and dropout RNG. Validation runs every 600 updates; step-01200.pt supplies the
equal-target comparison. The trainer defaults to CPU FP32, four threads, seed 17,
batch 32 and configs/wide.json.

Two late snapshot averages were compared on validation; the two-snapshot mean
was selected:

```bash
python average_checkpoints.py runs/wide/step-04800.pt runs/wide/step-05400.pt runs/wide/step-06000.pt --output runs/average-last3/checkpoint.pt
python evaluate.py --checkpoint runs/average-last3/checkpoint.pt --split validation --output runs/average-last3/validation.json
python average_checkpoints.py runs/wide/step-05400.pt runs/wide/step-06000.pt --output runs/average-last2/checkpoint.pt
python evaluate.py --checkpoint runs/average-last2/checkpoint.pt --split validation --output runs/average-last2/validation.json
python select_copy.py --checkpoint runs/average-last2/checkpoint.pt --output-dir runs/final-copy --space extended
```

The two means scored 1.71568774 and 1.71387940 respectively. The extended search
selects `[0.1, 0.3, 0.5]`. All selection uses validation; the recorded test results
are for reproducing the frozen predictor, not further tuning.

## Reproduce controls and resource measurements

Evaluate the reproduced baseline with the same full-test command, replacing
`--checkpoint` with `runs/baseline/checkpoint.pt`. To disable copying without
changing the final neural weights, create the ablation checkpoint:

```bash
python - <<'PYCODE'
from pathlib import Path
import torch
checkpoint = torch.load('release/checkpoint.pt', map_location='cpu', weights_only=True)
checkpoint['config'] = dict(checkpoint['config'], copy_weights=[0.0, 0.0, 0.0])
Path('runs').mkdir(exist_ok=True)
torch.save(checkpoint, 'runs/ablation.pt')
PYCODE
python evaluate.py --checkpoint runs/ablation.pt --device cpu --precision fp32 --threads 4 --split test --output ablation-test.json
```

Stop training before timing. Alternate baseline/final evaluation three times with
new output names, then compare median JSON `seconds` values. On macOS, prefix
the evaluation command with `/usr/bin/time -l` to record peak process RSS in bytes;
on Linux use `/usr/bin/time -v` (maximum RSS in KiB). CPU `peak_allocated_gb=0`
is a GPU metric, not a CPU RAM measurement. The limit is 5x baseline scoring time,
4 GiB peak process RAM and 64 MiB uncompressed inference assets.

## Costs, assistance and attribution

The baseline processed 9,830,400 targets in 315.42 training seconds; the main
run processed 49,152,000 targets in 2,435.11 seconds. A discarded interrupted
pilot processed 5,734,400-6,553,600 completed targets with at least 374.69 recorded
training seconds; additional unlogged partial work is unknown. A functional
smoke run processed 512 targets. Copy selection evaluated 8 baseline and 12 final
validation settings; all recorded selection and evaluation costs appear in the
report and results.json. The discarded pilot contributes no final weights.

OpenAI Codex substantially assisted with requirements, method design, code and
tests, local experiments, source checking, code review and report/documentation
writing. The student is responsible for understanding the submitted implementation.
The supplied GPT, training framework, evaluator, tokenizer and benchmark are reused
with attribution. No pretrained weights or external training text were used.

Recent-history mixtures are motivated by [Merity et al., 2016](https://arxiv.org/abs/1609.07843)
and [Grave et al., 2016](https://arxiv.org/abs/1612.04426). This implementation uses
exact within-window suffix counts, not their neural pointer/cache architectures.

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and
Richard Socher. The text is by Wikipedia contributors. The
[upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html);
retain these notices when redistributing the data. The supplied wikitext-2-raw-v1
splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined
with newlines and encoded as UTF-8; the tokenizer is fitted only to training text.
Dataset hashes are in code/data/manifest.json. These notices do not assign a new
license to the surrounding classroom code.
