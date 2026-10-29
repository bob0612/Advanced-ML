# MP2 submission: two-stage verifier

**Test accuracy (official scorer): 82.44% (385/467)**. Validation: 92.86% overall, 83.94% on the hard subset.
Report: [`report/report.pdf`](report/report.pdf) (5 pages). Official score JSONs are in [`code/results/`](code/results/).

## Method in one paragraph
Stage 1 is a logistic regression on group-centred, standardised frozen embeddings. Stage 2 is a gradient-boosted ranker over
27 hand-crafted features (answer form, consensus, surface, arithmetic check) plus 5 comparative stage-1
features. It picks the highest-scoring candidate. See the report for the ablations and analysis.

## Files added
| File | Purpose |
|---|---|
| `code/submission.py` | `predict(bank, arrays, checkpoint)`: inference only |
| `code/features.py`, `code/model.py` | feature extraction and model and bundle I/O |
| `code/train.py` | trains from train inputs and references, writes the bundle |
| `code/checkpoint/model_bundle/` | frozen inference assets (1.14 MiB) |
| `code/baselines/` | majority-vote (initial method) and embedding-only ablation |
| `code/experiments/` | development, ablation and analysis scripts (validation and train CV only) |
| `code/results/` | official prediction and score JSONs, analysis JSON |
| `report/` | LaTeX source, PDF and figures |

## Install (Python 3.12)
```bash
cd code
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

## Train (optional, about 5 s on CPU, deterministic)
```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python train.py --output checkpoint/model_bundle
```

## Evaluate (no retraining needed)
```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint checkpoint/model_bundle --split test --output runs/test-predictions.json
python score_predictions.py --split test --predictions runs/test-predictions.json --output runs/test-score.json
```

## Analysis (optional)
```bash
python experiments/analysis.py   # ablations, calibration, error analysis on validation
python experiments/cv_check.py   # 5-fold CV on train
python experiments/figures.py    # report figures (needs matplotlib)
```

## Resources
- Local hardware: Apple M5 (10 cores), macOS. Python 3.12.15, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1.
- Prediction plus scoring on the full test bank with 4 threads takes **1.34 s wall**, measured externally with `/usr/bin/time`. The very first run in a freshly created venv took about 6.4 s, because Python compiles the sklearn/scipy bytecode on first import. That is still well under 30 s.
- Inference assets total 1,197,761 B (1.14 MiB), under the 16 MiB limit.
- Seeds: KFold seed 0, ranker seeds 0, 1 and 2. Retraining reproduces identical hashes and predictions.

| Asset | Bytes | SHA-256 |
|---|---|---|
| `verifier.npz` | 13,286 | `784a179882102bec90bdd8e9f568f21bf8369b478a018788d17b6f4f4e6fff77` |
| `rankers.pkl` | 1,183,688 | `bb527cf797ed5477245989df451ff1edd237e2dc753e7e79eca0926a1bd278df` |
| `config.json` | 787 | `bf307dfc0cd0d65304add65f62c76f12ad1513c3e71d088f59f7745ba71d91f8` |

## Data use
Only training references were used for fitting. Validation was used for development and selection, including the input-only
"hard" subset with at least 2 distinct parsed answers. Test references were read once, by the official scorer, after the method was frozen.

## AI and reuse disclosure
AI coding assistance (Claude, Anthropic) was used substantively to write the code and draft the report. Beyond
numpy, scikit-learn and matplotlib, no external data, pretrained weights or third-party code was used.
