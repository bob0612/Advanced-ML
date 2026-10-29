"""Variants of the embedding verifier (pointwise vs pairwise, regularisation, training subset)."""
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stack import report, centered, flat, tr, va  # noqa: E402

Etr, Eva = centered(tr), centered(va)
sc = StandardScaler().fit(flat(Etr))
Ztr = sc.transform(flat(Etr)).reshape(Etr.shape)
Zva = sc.transform(flat(Eva)).reshape(Eva.shape)

for C in (0.003, 0.01, 0.03):
    for subset in ('all', 'hard'):
        m = tr['hard'] if subset == 'hard' else np.ones(len(tr['y']), bool)
        lr = LogisticRegression(C=C, max_iter=3000).fit(flat(Ztr[m]), tr['y'][m].ravel())
        report(f'pointwise C={C} train={subset}', (Zva @ lr.coef_[0]))


def pairs(Z, Y):
    X, t = [], []
    for g in range(len(Y)):
        for i in range(4):
            for j in range(4):
                if Y[g, i] > Y[g, j]:
                    X.append(Z[g, i] - Z[g, j]); t.append(1)
                    X.append(Z[g, j] - Z[g, i]); t.append(0)
    return np.array(X), np.array(t)


Ptr, ptr = pairs(Ztr, tr['y'])
print('pairs', Ptr.shape)
for C in (0.001, 0.003, 0.01):
    lr = LogisticRegression(C=C, max_iter=3000, fit_intercept=False).fit(Ptr, ptr)
    report(f'pairwise RankNet-linear C={C}', Zva @ lr.coef_[0])
