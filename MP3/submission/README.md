# DASE7506 MP3: feedback-driven SQL repair agent (CPU, no live LLM)

**Official test result: 179/240 = 74.58%** (`outputs/test-score.json`). Validation: 95/120 = 79.17%.
Report: [`report.pdf`](report.pdf).

## Contents
| Path | Purpose |
|---|---|
| `submission.py` | entry point: `create_agent(checkpoint)` returns an agent with `act(observation)` |
| `mp3agent/agent.py` | the perceive-plan-act controller (check draft, read hints, check candidates, submit) |
| `mp3agent/candidates.py`, `mp3agent/sqlops.py` | candidate pool and the sqlglot rewrites (align, complete, dropjoin, DISTINCT toggles, capital) |
| `mp3agent/features.py` | candidate features (static and from check feedback) |
| `config.json` | learned weights for the prior and the ranker (train split only) |
| `tools/train_ranker.py` | regenerates `config.json` from the train split |
| `requirements.txt` | run-time dependency (same pin as the official package) |
| `requirements-train.txt` | extra dependencies for retraining only |
| `report/` | report source (Markdown, figure, CSS); rendered to `report.pdf` with pandoc and Chromium |

## Reproduce (Linux or macOS, Python 3.11/3.12, offline, 4 CPU threads)
Extract this ZIP inside the official `MP3_SQL_student_package/` directory, so that `submission/` and `outputs/` sit next to `mp3_sql/`.
```bash
cd MP3_SQL_student_package
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m mp3_sql.check_data
PYTHONPATH="$PWD/submission" python -m mp3_sql.predict --split test \
  --method submission/submission.py --checkpoint submission/config.json --output outputs/test.json
python -m mp3_sql.score --split test --predictions outputs/test.json --output outputs/test-score.json
```
Expected: `"correct": 179`, `"n": 240`. The run is deterministic. It takes about 62 s wall time for prediction and scoring with 4 CPUs under a 2 GiB address-space limit. Use `--split validation` for the validation result (95/120).

Optional retraining (train split only, about 76 s):
```bash
python -m pip install -r submission/requirements-train.txt
PYTHONPATH="$PWD:$PWD/submission" python submission/tools/train_ranker.py --output submission/config.json
```

## Disclosure
No LLM is loaded or called at run time. The `repair_hint` tool reads the course's fixed hint cache. AI coding assistance (Claude, Anthropic) was used substantially to write the code and the report. Data: only the official Spider-derived package (CC BY-SA 4.0).
