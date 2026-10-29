"""Stacking experiments: out-of-fold embedding verifier score + hand features."""
import pickle
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FEATURE_NAMES  # noqa: E402

CACHE = Path(__file__).resolve().parents[1]/'runs'/'cache'
tr = pickle.load(open(CACHE/'train.pkl', 'rb'))
va = pickle.load(open(CACHE/'validation.pkl', 'rb'))
F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def acc(scores, d):
    pick = np.argmax(scores + 1e-9*np.arange(4)[::-1], axis=1)
    hit = d['y'][np.arange(len(pick)), pick]
    return 100*hit.mean(), 100*hit[d['hard']].mean()


def report(name, s):
    a, h = acc(s, va)
    print(f'{name:50s} val_all={a:6.2f}  val_hard={h:6.2f}', flush=True)


def centered(d):
    e = d['emb'].astype(np.float64)
    return e - e.mean(1, keepdims=True)


def flat(x):
    return x.reshape(-1, x.shape[-1])


def make_emb_model(kind):
    if kind == 'lr':
        return LogisticRegression(C=0.01, max_iter=3000)
    return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=200, early_stopping=True,
                         random_state=0)


def emb_scores(kind, train_idx_groups=None):
    """OOF scores on train and full-train-fit scores on validation."""
    Etr, Eva = centered(tr), centered(va)
    sc = StandardScaler().fit(flat(Etr))
    Xtr, Xva = sc.transform(flat(Etr)).reshape(Etr.shape), sc.transform(flat(Eva)).reshape(Eva.shape)
    oof = np.zeros(tr['y'].shape)
    for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xtr)):
        m = make_emb_model(kind).fit(flat(Xtr[a]), tr['y'][a].ravel())
        oof[b] = m.predict_proba(flat(Xtr[b]))[:, 1].reshape(-1, 4)
    m = make_emb_model(kind).fit(flat(Xtr), tr['y'].ravel())
    return oof, m.predict_proba(flat(Xva))[:, 1].reshape(-1, 4)


def logit(p):
    p = np.clip(p, 1e-5, 1-1e-5)
    return np.log(p/(1-p))


def group_stats(s, answers):
    """Comparative features of a candidate score inside its group."""
    z = logit(s)
    rel_max = z - z.max(1, keepdims=True)
    rel_mean = z - z.mean(1, keepdims=True)
    ans_sum = np.zeros_like(s)
    ans_max_other = np.zeros_like(s)
    for g, row in enumerate(answers):
        for i, a in enumerate(row):
            same = [j for j in range(4) if a is not None and row[j] == a]
            other = [j for j in range(4) if not (a is not None and row[j] == a)]
            ans_sum[g, i] = s[g, same].sum() if same else s[g, i]
            ans_max_other[g, i] = z[g, other].max() if other else -12
    return np.stack([z, rel_max, rel_mean, ans_sum, z - ans_max_other], -1)


def fit_hgb(X, y, seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed).fit(X, y)


def main():
    results = {}
    for kind in ('lr', 'mlp'):
        oof, sva = emb_scores(kind)
        results[kind] = (oof, sva)
        report(f'emb {kind} (full-train fit)', sva)
        Gtr, Gva = group_stats(oof, tr['answers']), group_stats(sva, va['answers'])
        for name, cols in [('hand+emb', list(range(len(FEATURE_NAMES)))),
                           ('hand(no emb-sim)+emb', [i for n, i in F.items() if not n.startswith('emb')])]:
            Xtr = np.concatenate([tr['X'][:, :, cols], Gtr], -1)
            Xva = np.concatenate([va['X'][:, :, cols], Gva], -1)
            for seed in (0, 1):
                m = fit_hgb(flat(Xtr), tr['y'].ravel(), seed)
                report(f'stack HGB {kind} {name} seed{seed}', m.predict_proba(flat(Xva))[:, 1].reshape(-1, 4))
            hm = tr['hard']
            m = fit_hgb(flat(Xtr[hm]), tr['y'][hm].ravel())
            report(f'stack HGB {kind} {name} train-hard', m.predict_proba(flat(Xva))[:, 1].reshape(-1, 4))

    pickle.dump(results, open(CACHE/'emb_scores.pkl', 'wb'))



if __name__ == "__main__":
    main()
