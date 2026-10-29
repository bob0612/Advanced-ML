"""Development comparisons on validation (all groups and the test-like hard subset)."""
import pickle
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURE_NAMES  # noqa: E402

CACHE = Path(__file__).resolve().parents[1]/'runs'/'cache'
tr = pickle.load(open(CACHE/'train.pkl', 'rb'))
va = pickle.load(open(CACHE/'validation.pkl', 'rb'))
F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def acc(scores, d, mask=None):
    pick = np.argmax(scores + 1e-9*np.arange(4)[::-1], axis=1)  # ties -> lowest index
    hit = d['y'][np.arange(len(pick)), pick]
    return 100*hit.mean(), 100*hit[d['hard']].mean()


def aggregate(scores, answers, how='sum'):
    """Pool candidate scores over candidates sharing a parsed answer."""
    out = scores.copy()
    for g, row in enumerate(answers):
        for i, a in enumerate(row):
            if a is None:
                continue
            same = [j for j in range(4) if row[j] == a]
            vals = scores[g, same]
            out[g, i] = vals.sum() if how == 'sum' else vals.max()
    return out


def report(name, s_va):
    a, h = acc(s_va, va)
    print(f'{name:45s} val_all={a:6.2f}  val_hard={h:6.2f}')


def emb_inputs(d, mode):
    e = d['emb'].astype(np.float64)
    if mode == 'raw':
        return e
    if mode == 'centered':
        return e - e.mean(1, keepdims=True)
    if mode == 'both':
        return np.concatenate([e, e - e.mean(1, keepdims=True)], -1)


def flat(x):
    return x.reshape(-1, x.shape[-1])


report('first candidate', np.tile([1., 0, 0, 0], (len(va['y']), 1)))
report('majority vote (tie -> first)', va['X'][:, :, F['vote_share']])
report('oracle', va['y'])

# embedding-only logistic regression
emb_scores_va = {}
for mode in ('raw', 'centered', 'both'):
    sc = StandardScaler().fit(flat(emb_inputs(tr, mode)))
    for C in (0.01, 0.1):
        lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(flat(emb_inputs(tr, mode))), tr['y'].ravel())
        s = lr.predict_proba(sc.transform(flat(emb_inputs(va, mode))))[:, 1].reshape(-1, 4)
        emb_scores_va[(mode, C)] = s
        report(f'emb LR {mode} C={C}', s)
        report(f'emb LR {mode} C={C} + answer-sum', aggregate(s, va['answers']))


def fit_hgb(X, y, seed=0, **kw):
    params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                  random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params).fit(X, y)


# hand features only
for cols_name, cols in [('all hand', list(range(len(FEATURE_NAMES)))),
                        ('no vote', [i for n, i in F.items() if not n.startswith(('vote', 'n_distinct'))]),
                        ('no emb-sim', [i for n, i in F.items() if not n.startswith('emb')]),
                        ('vote only', [F[n] for n in ('vote_share', 'vote_is_max', 'vote_unique_max', 'n_distinct', 'vote_rank', 'n_none_group')]),
                        ('surface only', [F[n] for n in ('log_chars', 'log_lines', 'n_boxed', 'has_hash', 'n_steps', 'hedge_count', 'n_numbers', 'rel_len')])]:
    m = fit_hgb(flat(tr['X'][:, :, cols]), tr['y'].ravel())
    s = m.predict_proba(flat(va['X'][:, :, cols]))[:, 1].reshape(-1, 4)
    report(f'HGB {cols_name}', s)
    hm = tr['hard']
    m2 = fit_hgb(flat(tr['X'][hm][:, :, cols]), tr['y'][hm].ravel())
    s2 = m2.predict_proba(flat(va['X'][:, :, cols]))[:, 1].reshape(-1, 4)
    report(f'HGB {cols_name} (train hard only)', s2)
