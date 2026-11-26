# Development scripts (not needed for reproduction)

Exploration and ablation helpers used during development. They read only train and validation
references (never test), and they contain absolute paths from the development container
(`/home/user/mp3/...`). Adjust the paths before running them. Ablation score JSONs are in `../ablations/`.

- `oracle.py`: run SQL on the public DB and the 3 grading instances (train/validation only), with a cache.
- `analyze_hints.py`, `coverage.py`, `phase2.py`: candidate coverage studies.
- `collect.py`, `select_eval*.py`: offline ranker comparisons (LR, gradient boosting, random forest).
- `ablate.sh`: retrain (if needed) and score one ablated configuration with the official predictor and scorer.
- `measure.py`: wall/CPU/RSS measurement of test prediction plus scoring under 4 CPUs and `ulimit -v 2 GiB`.
