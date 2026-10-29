"""Further variants: listwise embedding model, text n-gram model, answer-level selection."""
import pickle
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURE_NAMES  # noqa: E402
from run_stack import acc, report, centered, flat, group_stats, fit_hgb, tr, va, F  # noqa: E402

CACHE = Path(__file__).resolve().parents[1]/'runs'/'cache'
emb = pickle.load(open(CACHE/'emb_scores.pkl', 'rb'))


# ---- (a) listwise softmax on centered embeddings (numpy, full-batch Adam)
def listwise_fit(X, Y, l2=1e-2, steps=400, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1e-3, X.shape[-1])
    m = v = np.zeros_like(w)
    T = Y / Y.sum(1, keepdims=True)  # uniform target over correct candidates
    mask = (Y.sum(1) > 0) & (Y.sum(1) < 4)
    X, T = X[mask], T[mask]
    for t in range(1, steps+1):
        z = X @ w
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = np.einsum('gk,gkd->d', p - T, X)/len(X) + l2*w
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*g*g
        w -= lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8)
    return w


Etr, Eva = centered(tr), centered(va)
sc = StandardScaler().fit(flat(Etr))
Ztr = sc.transform(flat(Etr)).reshape(Etr.shape); Zva = sc.transform(flat(Eva)).reshape(Eva.shape)
for l2 in (1e-3, 1e-2, 1e-1):
    w = listwise_fit(Ztr, tr['y'], l2=l2)
    report(f'listwise emb l2={l2}', Zva @ w)

# ---- (b) text n-gram model (hashing => fixed small asset)
texts_tr = [c['text'] for r in tr['bank'] for c in r['candidates']]
texts_va = [c['text'] for r in va['bank'] for c in r['candidates']]
hv = HashingVectorizer(n_features=2**16, ngram_range=(1, 2), alternate_sign=False, norm=None)
Htr, Hva = hv.transform(texts_tr), hv.transform(texts_va)
tf = TfidfTransformer(sublinear_tf=True).fit(Htr)
Htr, Hva = tf.transform(Htr), tf.transform(Hva)
for C in (0.3, 1, 3):
    lr = LogisticRegression(C=C, max_iter=3000).fit(Htr, tr['y'].ravel())
    s_txt = lr.predict_proba(Hva)[:, 1].reshape(-1, 4)
    report(f'text tfidf LR C={C}', s_txt)

# OOF text scores for stacking
C = 1
oof_txt = np.zeros(tr['y'].shape)
idx = np.arange(len(tr['y']))
for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
    rows_a = (a[:, None]*4 + np.arange(4)).ravel(); rows_b = (b[:, None]*4 + np.arange(4)).ravel()
    m = LogisticRegression(C=C, max_iter=3000).fit(Htr[rows_a], tr['y'][a].ravel())
    oof_txt[b] = m.predict_proba(Htr[rows_b])[:, 1].reshape(-1, 4)
s_txt = LogisticRegression(C=C, max_iter=3000).fit(Htr, tr['y'].ravel()).predict_proba(Hva)[:, 1].reshape(-1, 4)
pickle.dump((oof_txt, s_txt), open(CACHE/'txt_scores.pkl', 'wb'))

oof_lr, sva_lr = emb['lr']
cols = [i for n, i in F.items() if not n.startswith('emb')]
base_tr = np.concatenate([tr['X'][:, :, cols], group_stats(oof_lr, tr['answers'])], -1)
base_va = np.concatenate([va['X'][:, :, cols], group_stats(sva_lr, va['answers'])], -1)
Xtr = np.concatenate([base_tr, group_stats(oof_txt, tr['answers'])], -1)
Xva = np.concatenate([base_va, group_stats(s_txt, va['answers'])], -1)
for seed in (0, 1, 2):
    report(f'stack hand+emb seed{seed}', fit_hgb(flat(base_tr), tr['y'].ravel(), seed).predict_proba(flat(base_va))[:, 1].reshape(-1, 4))
    report(f'stack hand+emb+text seed{seed}', fit_hgb(flat(Xtr), tr['y'].ravel(), seed).predict_proba(flat(Xva))[:, 1].reshape(-1, 4))


# ---- (c) answer-level selection: pool candidate features per distinct answer
def answer_level(X, answers):
    """For each candidate, mean/max/min of features over candidates with the same answer."""
    out = np.zeros(X.shape[:2] + (3*X.shape[2],))
    for g, row in enumerate(answers):
        for i, a in enumerate(row):
            same = [j for j in range(4) if a is not None and row[j] == a] or [i]
            sub = X[g, same]
            out[g, i] = np.concatenate([sub.mean(0), sub.max(0), sub.min(0)])
    return out


Atr, Ava = answer_level(base_tr, tr['answers']), answer_level(base_va, va['answers'])
for seed in (0, 1, 2):
    report(f'answer-level stack hand+emb seed{seed}', fit_hgb(flat(Atr), tr['y'].ravel(), seed).predict_proba(flat(Ava))[:, 1].reshape(-1, 4))
