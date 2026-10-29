"""Round-2: can a different stage-1 embedding model beat centred-LR? (train CV + validation)."""
from pathlib import Path
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opt import load  # noqa: E402
from model import choose, centred  # noqa: E402


def std_fit(x):
    mu, sd = x.mean(0), x.std(0) + 1e-8
    return lambda z: (z - mu)/sd


def m_lr(C, feats):
    def fit(e, y):
        x = feats(e).reshape(len(e)*4, -1)
        s = std_fit(x)
        lr = LogisticRegression(C=C, max_iter=3000).fit(s(x), y.ravel())
        return lambda E: lr.decision_function(s(feats(E).reshape(len(E)*4, -1))).reshape(-1, 4)
    return fit


def m_lda(shrink):
    def fit(e, y):
        x = centred(e).reshape(-1, e.shape[-1])
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrink).fit(x, y.ravel())
        return lambda E: lda.decision_function(centred(E).reshape(-1, E.shape[-1])).reshape(-1, 4)
    return fit


def m_pca_quad(k, C):
    def fit(e, y):
        x = centred(e).reshape(-1, e.shape[-1])
        pca = PCA(k, random_state=0).fit(x)
        f = lambda z: np.concatenate([pca.transform(z), pca.transform(z)**2], 1)  # noqa: E731
        s = std_fit(f(x))
        lr = LogisticRegression(C=C, max_iter=3000).fit(s(f(x)), y.ravel())
        return lambda E: lr.decision_function(s(f(centred(E).reshape(-1, E.shape[-1])))).reshape(-1, 4)
    return fit


def m_ens(fits):
    def fit(e, y):
        ps = [f(e, y) for f in fits]
        def pred(E):
            zs = [p(E) for p in ps]
            return np.mean([(z - z.mean())/z.std() for z in zs], 0)
        return pred
    return fit


def l2norm_centred(e):
    e = e/np.linalg.norm(e, axis=-1, keepdims=True)
    return e - e.mean(1, keepdims=True)


def score(tr, va, fit, name):
    hits = np.zeros(len(tr['y']))
    for a, b in KFold(5, shuffle=True, random_state=1).split(tr['emb']):
        s = fit(tr['emb'][a], tr['y'][a])(tr['emb'][b])
        hits[b] = tr['y'][b][np.arange(len(b)), choose(s)]
    s = fit(tr['emb'], tr['y'])(va['emb'])
    hv = va['y'][np.arange(len(s)), choose(s)]
    print(f'{name:36s} cv_hard={100*hits[tr["hard"]].mean():.2f} val_hard={100*hv[va["hard"]].mean():.2f}', flush=True)


if __name__ == '__main__':
    tr, va = load('train'), load('validation')
    for C in (0.001, 0.003):
        score(tr, va, m_lr(C, centred), f'centred LR C={C}')
    score(tr, va, m_lr(0.003, l2norm_centred), 'l2norm+centred LR C=0.003')
    score(tr, va, m_lr(0.003, lambda e: np.concatenate([centred(e), e.astype(np.float64)], -1)), 'centred+raw LR')
    for sh in (0.1, 0.3, 0.6):
        score(tr, va, m_lda(sh), f'shrinkage LDA {sh}')
    for k in (64, 128):
        score(tr, va, m_pca_quad(k, 0.01), f'PCA{k} + squares LR')
    score(tr, va, m_ens([m_lr(0.003, centred), m_lda(0.3), m_lr(0.003, lambda e: e.astype(np.float64))]),
          'ensemble centredLR+LDA+rawLR')
