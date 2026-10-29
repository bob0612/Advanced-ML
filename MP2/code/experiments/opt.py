"""Round-2 optimisation: extra feature families, judged by train 5-fold CV and validation (never test)."""
from fractions import Fraction
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_inputs  # noqa: E402
from features import NUM_RE, bank_features  # noqa: E402
from model import EmbeddingVerifier, choose, make_ranker, stack_features  # noqa: E402
from train import labels, oof_verifier_scores  # noqa: E402


def load(split):
    bank, arrays = load_inputs(split)
    X, answers = bank_features(bank, arrays['embeddings'])
    y = labels(answers, split)
    hard = np.array([len({a for a in r if a is not None}) >= 2 for r in answers])
    return dict(bank=bank, emb=arrays['embeddings'], X=X, answers=answers, y=y, hard=hard)


def text_numbers(text):
    out = []
    for t in NUM_RE.findall(text):
        try:
            out.append(str(Fraction(t.replace(',', ''))))
        except (ValueError, ZeroDivisionError):
            pass
    return out


def cross_features(d):
    """Cross-candidate textual evidence."""
    F = np.zeros(d['y'].shape + (4,))
    for g, row in enumerate(d['bank']):
        nums = [text_numbers(c['text']) for c in row['candidates']]
        ans = d['answers'][g]
        for i, a in enumerate(ans):
            if a is None:
                continue
            others = [j for j in range(4) if j != i and ans[j] != a]
            F[g, i, 0] = sum(a in nums[j] for j in others)              # my answer appears in rival reasoning
            rivals = {ans[j] for j in others if ans[j] is not None}
            F[g, i, 1] = sum(r in nums[i] for r in rivals)               # rival answers appear in my reasoning
            F[g, i, 2] = np.log1p(nums[i].count(a))                      # how often I restate my answer
            F[g, i, 3] = float(bool(rivals)) and sum(a in nums[j] for j in range(4) if j != i) / 3
    return F


def cluster_features(p, answers):
    """Answer-cluster statistics of stage-1 logits."""
    z = np.log(np.clip(p, 1e-5, 1-1e-5)/np.clip(1-p, 1e-5, 1))
    F = np.zeros(p.shape + (4,))
    for g, row in enumerate(answers):
        clusters = {}
        for j, a in enumerate(row):
            clusters.setdefault(a if a is not None else f'none{j}', []).append(j)
        cm = {k: z[g, v].mean() for k, v in clusters.items()}
        for i, a in enumerate(row):
            k = a if a is not None else f'none{i}'
            other = [v for kk, v in cm.items() if kk != k]
            F[g, i] = [cm[k], z[g, clusters[k]].max(), cm[k] - (max(other) if other else -12),
                       len(clusters[k])*cm[k]]
    return F


def raw_verifier_scores(emb_a, y_a, emb_b, oof=True):
    """Second stage-1 model on raw (uncentred) embeddings for diversity."""
    def fit(e, yy):
        x = e.reshape(-1, e.shape[-1]).astype(np.float64)
        mu, sd = x.mean(0), x.std(0) + 1e-8
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=0.003, max_iter=3000).fit((x-mu)/sd, yy.ravel())
        return lambda E: lr.predict_proba(((E.reshape(-1, E.shape[-1])-mu)/sd))[:, 1].reshape(-1, 4)
    o = np.zeros(y_a.shape)
    for a, b in KFold(5, shuffle=True, random_state=0).split(emb_a):
        o[b] = fit(emb_a[a], y_a[a])(emb_a[b])
    return o, fit(emb_a, y_a)(emb_b)


def build(d, p, extras):
    S = stack_features(d['X'], p, d['answers'])
    return np.concatenate([S] + extras, -1) if extras else S


def run_variant(tr, te, variant, ranker_kw=None, seeds=(0, 1, 2)):
    oof = oof_verifier_scores(tr['emb'], tr['y'])
    p = EmbeddingVerifier().fit(tr['emb'], tr['y']).predict_proba(te['emb'])
    ex_tr, ex_te = [], []
    if 'cross' in variant:
        ex_tr.append(tr['cross']); ex_te.append(te['cross'])
    if 'cluster' in variant:
        ex_tr.append(cluster_features(oof, tr['answers'])); ex_te.append(cluster_features(p, te['answers']))
    if 'raw' in variant:
        o2, p2 = raw_verifier_scores(tr['emb'], tr['y'], te['emb'])
        from model import group_stats
        ex_tr.append(group_stats(o2, tr['answers'])); ex_te.append(group_stats(p2, te['answers']))
    Str, Ste = build(tr, oof, ex_tr), build(te, p, ex_te)
    ms = []
    for s in seeds:
        m = make_ranker(s)
        if ranker_kw:
            m.set_params(**ranker_kw)
        ms.append(m.fit(Str.reshape(-1, Str.shape[-1]), tr['y'].ravel()))
    sc = np.mean([m.predict_proba(Ste.reshape(-1, Ste.shape[-1]))[:, 1] for m in ms], 0).reshape(-1, 4)
    return te['y'][np.arange(len(sc)), choose(sc)]


def subset(d, idx):
    return {k: ([v[i] for i in idx] if isinstance(v, list) else v[idx]) for k, v in d.items()}


def evaluate(tr, va, variant, ranker_kw=None, label=None):
    hits = np.zeros(len(tr['y']))
    for a, b in KFold(5, shuffle=True, random_state=1).split(tr['emb']):
        hits[b] = run_variant(subset(tr, a), subset(tr, b), variant, ranker_kw)
    hv = run_variant(tr, va, variant, ranker_kw)
    print(f'{label or "+".join(variant) or "baseline":40s} cv_all={100*hits.mean():.2f} '
          f'cv_hard={100*hits[tr["hard"]].mean():.2f} val_all={100*hv.mean():.2f} '
          f'val_hard={100*hv[va["hard"]].mean():.2f}', flush=True)


if __name__ == '__main__':
    tr, va = load('train'), load('validation')
    tr['cross'], va['cross'] = cross_features(tr), cross_features(va)
    for v in [(), ('cross',), ('cluster',), ('raw',), ('cross', 'cluster'), ('cross', 'cluster', 'raw')]:
        evaluate(tr, va, v)
    for kw in [dict(max_iter=600, learning_rate=0.03), dict(max_leaf_nodes=7), dict(max_leaf_nodes=31, min_samples_leaf=50),
               dict(l2_regularization=5.0, min_samples_leaf=40)]:
        evaluate(tr, va, (), kw, label=f'baseline {kw}')
