# MP1: Rotary language model with causal suffix copying

A from-scratch 6-layer, width-320 Transformer with RoPE, RMSNorm, SwiGLU,
dropout 0.2 and tied token/output embeddings. Stateless suffix copying uses only
observed successors in the current 256-token window. No external training text
or pretrained weights are used.

## Frozen results

| Predictor | Training targets | Validation BPB | Full-test BPB |
| --- | ---: | ---: | ---: |
| Original baseline | 9,830,400 | 2.07108743 | 2.10126520 |
| Rotary neural, update 1,200 | 9,830,400 | 1.74161605 | - |
| Rotary + copy, update 1,200 | 9,830,400 | 1.67271168 | - |
| Final neural weights, copy disabled | 68,812,800 | 1.55865998 | - |
| **Final predictor, update 8,400** | **68,812,800** | **1.53268101** | **1.5501346122** |

CPU FP32,4 threads, complete test:428,405 targets and 1,292,013 raw UTF-8 bytes.
Three final passes reproduce the same score. Median scoring time is
3.022 x baseline (limit 5 x); peak process RSS
1.638 GiB (limit 4 GiB); uncompressed checkpoint plus tokenizer
31.707 MiB (limit 64 MiB). Training stopped at
the user's submission cutoff. The earlier aspiration of BPB below 1.5 was not
achieved; the score above is the measured result.

[REPORT.pdf](REPORT.pdf) contains methods, equal-target controls, the copy ablation,
costs and limitations. [code/results.json](code/results.json) records the measured
results, trial summaries, ancestry and frozen hashes. Development experiments,
optimizer states and temporary audit tools are excluded from the submission.

## Install and evaluate without retraining

Use Python 3.12. Recorded versions: Python 3.12.14, PyTorch 2.7.1, NumPy 2.5.3,
tokenizers 0.21.4. Measurements used an Apple M5 MacBook Air with 16 GB RAM.

From a repository checkout:

```bash
cd MP1_student_starter/code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py --checkpoint release/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

Alternatively, extract `artifacts/mp1-code.tar.gz` and
`artifacts/mp1-checkpoint.tar.gz` into the same directory and use
`cd MP1_project/code`. On Windows activate with `.venv\Scripts\Activate.ps1`.
For Linux/Windows CPU, install `torch==2.7.1` from
`https://download.pytorch.org/whl/cpu` before requirements.txt if needed.
Evaluation is offline and needs no optimizer state or retraining. Submit JSON
`bpb`, not token perplexity or validation BPB. Hardware/library changes can cause
small numerical differences.

Checkpoint SHA256:
`e7657479f414c8224e20b7f1531b14d5d9ddc33542ccbf59bdc56cb6a5b03939`.

## Reproduce training

All following commands run in `code/`, using new output directories. The final
run used seed 17, batch 32, context 256, FP32 Apple MPS training, AdamW learning
rate 0.001, weight decay 0.1,100 warmup updates, cosine decay with a 10%floor, and
gradient clipping at 1.0. **Keep the 24,000-update cosine horizon, but stop after
8,400 updates.** Changing `--steps` to 8400 changes the trained model.

```bash
python train_resumable.py --config configs/scaled.json --run-dir runs/rotary320 --steps 24000 --chunk-steps 300 --eval-every 1200 --device mps
python - <<'PYCODE'
import subprocess, sys
for _ in range(27):
    subprocess.run([sys.executable, 'train_resumable.py', '--config', 'configs/scaled.json',
        '--run-dir', 'runs/rotary320', '--steps', '24000', '--chunk-steps', '300',
        '--eval-every', '1200', '--device', 'mps', '--resume'], check=True)
PYCODE
```

On a machine without Apple MPS, replace `--device mps` with `--device cpu` in
both commands. That reproduces the recipe, but cross-device retraining is not
guaranteed bitwise identical. Each 300-step boundary preserves optimizer,
sampler, CPU and MPS RNG state. CPU validation runs every 1,200 updates. Only
train and validation data are loaded by this trainer.

Prepare the selected predictor (no new optimization or averaging):

```bash
python - <<'PYCODE'
from pathlib import Path
import torch
x = torch.load('runs/rotary320/step-08400.pt', map_location='cpu', weights_only=True)
x['config'] = dict(x['config'], copy_weights=[0.1, 0.3, 0.5])
Path('runs/reproduced-final').mkdir(parents=True, exist_ok=True)
torch.save(x, 'runs/reproduced-final/checkpoint.pt')
PYCODE
python evaluate.py --checkpoint runs/reproduced-final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation --output reproduced-validation.json
```

The scalar copy weights were selected by validation in the earlier development
round and held fixed throughout this rotary run. The final model uses one
checkpoint; it is not a weight average or ensemble. `average_checkpoints.py`,
`configs/wide.json` and `select_copy.py` reproduce the earlier reported searches.

## Controls and resource measurement

Original baseline (1,200 updates,9,830,400 targets):

```bash
python train.py --implementation model --device cpu --precision fp32 --threads 4 --seed 17 --run-dir runs/baseline
```

The rotary `step-01200.pt` is its equal-target comparison. To reproduce the
paired mechanism ablation, set `copy_weights=[0.0,0.0,0.0]` in a copy of the final
checkpoint and evaluate validation with the identical frozen neural tensors.
This is already the configuration in `runs/rotary320/step-08400.pt`.

Stop training before timing. Alternate baseline and final complete-test
evaluations three times with new output names; divide median JSON `seconds`.
Use identical CPU FP324-thread settings. Prefix commands with `/usr/bin/time -l`
on macOS or `/usr/bin/time -v` on Linux for maximum RSS (bytes on macOS, KiB on
Linux). JSON `peak_allocated_gb=0` is a GPU statistic, not CPU RAM.
Inference assets are the complete checkpoint plus tokenizer; both are bundled.

## Costs, assistance and attribution

Final-model ancestry:68,812,800 training targets and 4528.26 s
training. Previous baseline/wide training and a rejected learned-pointer pilot
are additional search costs, detailed in the report/results. A discarded early
interrupted pilot has partially unrecorded cost, disclosed as unknown rather
than zero. Synthetic speed/correctness checks contribute no final weights.
An untrained MoS prototype was not selected or shipped.

OpenAI Codex substantially assisted with requirements, method design,
implementation, tests, local experiments, independent code review and report
writing. The student is responsible for understanding the implementation.
The supplied GPT, training framework, evaluator, tokenizer and benchmark are
reused with attribution. The supplied evaluator, model.py, common.py and data
remain unchanged. No course form/score issue or peer review was submitted by
the assistant.

Architectural references: [RoPE](https://arxiv.org/abs/2104.09864),
[RMSNorm](https://arxiv.org/abs/1910.07467),
[SwiGLU](https://arxiv.org/abs/2002.05202).
Recent-history mixtures are motivated by
[Merity et al.](https://arxiv.org/abs/1609.07843) and
[Grave et al.](https://arxiv.org/abs/1612.04426); our exact suffix counts differ
from their neural pointer/cache implementations.

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and
Richard Socher; text is by Wikipedia contributors. The
[upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies
[CC BY-SA3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html).
Retain these notices when redistributing data. The supplied wikitext-2-raw-v 1
revision is `b08601e04326c79dfdd32d625aee71d232d685c3`, joined with newlines and
encoded as UTF-8. The tokenizer is train-fitted. Hashes are in
`code/data/manifest.json`; these data notices do not license classroom code.
