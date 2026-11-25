# MP3 submission: feedback-driven SQL repair agent

**Test accuracy (official scorer, all 3 instances): 75.83% (182/240).** Validation: 80.00% (96/120). Starter draft: 36.67% (88/240) on test.
Report: [`submission/report.pdf`](submission/report.pdf). Official outputs: [`outputs/test.json`](outputs/test.json), [`outputs/test-score.json`](outputs/test-score.json).

## Method in one paragraph
A deterministic CPU controller with no live LLM. It checks the original draft, reads the shared repair-hint cache, and builds a candidate
pool from the draft, the hints and sqlglot rewrites: aligning the SELECT list to the example's column labels, completing missing
label columns, dropping unused joins, DISTINCT toggles, COUNT(*) normalisation, argmax-to-LIMIT 1,
"both"-to-INTERSECT and literal capitalisation. It spends the remaining budget (8 calls) checking
candidates in order of a learned prior, then submits the candidate with the highest learned ranker score. Both models are logistic
regressions fitted on the train split only. "DISTINCT twin" features tell the ranker when a version with and one without DISTINCT both match.

## Files
| Path | Purpose |
|---|---|
| `submission/` | the export: `submission.py`, `mp3agent/`, `config.json`, `tools/train_ranker.py`, report and requirements |
| `outputs/` | official `test.json`, `test-score.json`, plus the validation run |
| `ablations/` | official validation score JSONs for each ablation, and the baseline test score. `v1/` holds the first version (test 179/240) |
| `experiments/` | development scripts (train and validation only) |
| `mp3-source.zip` | the source ZIP for the course form: `submission/` and `outputs/`. Extract it inside `MP3_SQL_student_package/` |

## Reproduce
See [`submission/README.md`](submission/README.md). In short, from `MP3_SQL_student_package/` with the ZIP extracted:
```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt
PYTHONPATH="$PWD/submission" python -m mp3_sql.predict --split test --method submission/submission.py \
  --checkpoint submission/config.json --output outputs/test.json
python -m mp3_sql.score --split test --predictions outputs/test.json --output outputs/test-score.json
```

## Resources
- Linux, Python 3.12.3, sqlglot 27.29.0. Retraining also needs numpy 2.5.5 and scikit-learn 1.9.1.
- Test prediction plus scoring: about 30 s wall, about 80 s user CPU, 4 CPUs (`taskset -c 0-3`), `ulimit -v 2 GiB`, peak RSS below 80 MiB.
- Export size: about 0.3 MB. No model weights, no GPU, no network access.
- Deterministic. Retraining reproduces `config.json` byte for byte, and clean-room runs give 182/240.

## Data use
Train references were used for fitting. Validation and grouped 5-fold cross-validation on train were used for design selection. Test references were read only by the official scorer.

## AI disclosure
AI coding assistance (Claude, Anthropic) was used substantially to write the code, run the experiments and draft the report.
