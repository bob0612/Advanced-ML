# MP3 submission: feedback-driven SQL repair agent with a LoRA proposer

**Test accuracy (official scorer, all 3 instances): 77.92% (187/240)**, graded configuration: CPU only, 4 threads, `--workers 1`, `Qwen/Qwen3-0.6B` + our LoRA adapter. Validation: 84.17% (101/120). Starter draft: 36.67% (88/240) on test. Without the model the same export runs as the CPU-only controller (182/240).
Report: [`submission/report.pdf`](submission/report.pdf). Official outputs: [`outputs/test.json`](outputs/test.json), [`outputs/test-score.json`](outputs/test-score.json), model logs `outputs/llm-log-*.jsonl`.

## Method in one paragraph
A deterministic controller. It checks the original draft, reads the shared repair-hint cache, and builds a candidate
pool from the draft, the hints and sqlglot rewrites (projection alignment and completion, dropping unused joins, DISTINCT toggles,
COUNT(*) normalisation, argmax-to-LIMIT 1, "both"-to-INTERSECT, literal capitalisation). It checks candidates in order of a learned
prior and submits the one with the highest learned ranker score (logistic regressions fitted on the train split only). When no checked
candidate reproduces the public example, it asks `Qwen/Qwen3-0.6B` with a LoRA adapter (trained on the train split only, 1,359 prompt
variants) for up to 6 new SQL proposals for its last check, and submits a proposal that matches. When to call the model was chosen by 5-fold cross-fitting on train.

## Files
| Path | Purpose |
|---|---|
| `submission/` | the export: `submission.py`, `mp3agent/`, `config.json`, `tools/` (ranker and LoRA training), report and requirements |
| `outputs/` | official `test.json`, `test-score.json`, the validation run, model logs and measured resources |
| `ablations/` | official validation score JSONs for each controller ablation; `cpu-only/` holds the CPU-only controller outputs (test 182/240), `v1/` the first version (179/240) |
| `experiments/` | development and GPU/CPU run scripts (train and validation only) |
| `mp3-source.zip` | the source ZIP for the course form: `submission/` and `outputs/`. Extract it inside `MP3_SQL_student_package/` |

The LoRA adapter (`mp3-lora-adapter.zip`, 37 MB, adapter weights only) is submitted separately and is not stored in this repository.

## Reproduce
See [`submission/README.md`](submission/README.md) for the full commands, including the graded live-LLM run.

## Resources
- Live LLM (graded configuration): Linux, Python 3.12.3, PyTorch 2.7.1, transformers 4.53.2, peft 0.17.1; 4 cores (`taskset -c 0-3`), 4 threads, no GPU, offline. Test prediction 674 s wall (11 min), peak RSS 5,067 MiB; the model was called on 29 of 240 tasks.
- CPU-only controller: about 30 s wall for test, peak RSS below 80 MiB, no model needed.
- LoRA training and the validation sweeps used rented GPUs (RunPod).

## Data use
Train references were used for fitting the ranker and the LoRA adapter. Validation and grouped 5-fold cross-validation on train were used for design selection. Test references were read only by the official scorer; test results were never used to choose between variants.

## AI disclosure
AI coding assistance (Claude, Anthropic) was used substantially to write the code, run the experiments and draft the report.
