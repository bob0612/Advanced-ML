"""Round-3: condition coverage, answer-magnitude plausibility and kNN retrieval (train CV + validation only)."""
from fractions import Fraction
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import NUM_RE  # noqa: E402
from model import EmbeddingVerifier, centred, choose, make_ranker, stack_features  # noqa: E402
from opt import load, subset  # noqa: E402
from train import oof_verifier_scores  # noqa: E402

WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
         'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'twenty': 20, 'thirty': 30, 'forty': 40,
         'fifty': 50, 'hundred': 100, 'dozen': 12, 'twice': 2, 'double': 2, 'triple': 3, 'thrice': 3,
         'half': Fraction(1, 2), 'third': Fraction(1, 3), 'quarter': Fraction(1, 4)}


def frac(t):
    try:
        return Fraction(t.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


def question_conditions(q):
    """Distinct quantities stated in the question, each with acceptable written forms."""
    conds = []
    for m in re.finditer(r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|percent)?', q):
        f = frac(m.group(1))
        if f is None:
            continue
        forms = {f}
        if m.group(2):
            forms.add(f/100)
        conds.append(forms)
    low = q.lower()
    for w, v in WORDS.items():
        if re.search(rf'\b{w}\b', low):
            forms = {Fraction(v)}
            if v == Fraction(1, 2):
                forms |= {Fraction(2)}
            conds.append(forms)
    uniq = []
    for c in conds:
        if c not in uniq:
            uniq.append(c)
    return uniq


def coverage_features(d):
    F = np.zeros(d['y'].shape + (5,))
    for g, row in enumerate(d['bank']):
        conds = question_conditions(row['question'])
        digit_conds = [c for c in conds if any(x.denominator == 1 or True for x in c)]
        qmax = max([max(abs(float(x)) for x in c) for c in conds], default=1.0)
        cov = []
        for i, c in enumerate(row['candidates']):
            nums = {frac(t) for t in NUM_RE.findall(c['text'])} - {None}
            hit = [bool(cs & nums) for cs in digit_conds]
            cov.append(np.mean(hit) if hit else 1.0)
            a = frac(d['answers'][g][i]) if d['answers'][g][i] is not None else None
            F[g, i, 1] = float(len(hit) - sum(hit))                      # number of unused conditions
            F[g, i, 2] = math.log10((abs(float(a)) + 1)/(qmax + 1)) if a is not None else 0.0
        cov = np.array(cov)
        F[g, :, 0] = cov
        F[g, :, 3] = cov - cov.max()
        mags = F[g, :, 2]
        F[g, :, 4] = np.abs(mags - np.median(mags))
    return F


def knn_scores(tr_emb, tr_y, te_emb, k=50, dims=64, exclude_self=False):
    """Mean correctness of the k nearest train candidates in a PCA space of centred embeddings."""
    x_tr = centred(tr_emb).reshape(-1, tr_emb.shape[-1])
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    pca = PCA(dims, random_state=0).fit((x_tr - mu)/sd)
    proj = lambda e: pca.transform((centred(e).reshape(-1, e.shape[-1]) - mu)/sd)  # noqa: E731
    A = proj(tr_emb); A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
    B = proj(te_emb); B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
    lab = tr_y.ravel()
    grp = np.repeat(np.arange(len(tr_emb)), 4)
    out = np.zeros(len(B))
    for s in range(0, len(B), 2048):
        sim = B[s:s+2048] @ A.T
        if exclude_self:
            rows = np.arange(s, min(s+2048, len(B)))
            sim[grp[rows][:, None] == grp[None, :]] = -9
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        out[s:s+2048] = lab[idx].mean(1)
    return out.reshape(-1, 4)


def knn_feats(s):
    return np.stack([s, s - s.max(1, keepdims=True), s - s.mean(1, keepdims=True)], -1)


def run(tr, te, variant, seeds=(0, 1, 2)):
    oof = oof_verifier_scores(tr['emb'], tr['y'])
    p = EmbeddingVerifier().fit(tr['emb'], tr['y']).predict_proba(te['emb'])
    ex_tr, ex_te = [], []
    if 'cov' in variant:
        ex_tr.append(tr['cov']); ex_te.append(te['cov'])
    if 'knn' in variant:
        ex_tr.append(knn_feats(knn_scores(tr['emb'], tr['y'], tr['emb'], exclude_self=True)))
        ex_te.append(knn_feats(knn_scores(tr['emb'], tr['y'], te['emb'])))
    Str = np.concatenate([stack_features(tr['X'], oof, tr['answers'])] + ex_tr, -1)
    Ste = np.concatenate([stack_features(te['X'], p, te['answers'])] + ex_te, -1)
    ms = [make_ranker(s).fit(Str.reshape(-1, Str.shape[-1]), tr['y'].ravel()) for s in seeds]
    sc = np.mean([m.predict_proba(Ste.reshape(-1, Ste.shape[-1]))[:, 1] for m in ms], 0).reshape(-1, 4)
    return te['y'][np.arange(len(sc)), choose(sc)]


def evaluate(tr, va, variant, cv_seeds=(1, 2)):
    cvh = []
    for cs in cv_seeds:
        hits = np.zeros(len(tr['y']))
        for a, b in KFold(5, shuffle=True, random_state=cs).split(tr['emb']):
            hits[b] = run(subset(tr, a), subset(tr, b), variant)
        cvh.append(100*hits[tr['hard']].mean())
    hv = run(tr, va, variant)
    print(f'{"+".join(variant) or "baseline":20s} cv_hard={np.mean(cvh):.2f} ({" / ".join(f"{c:.2f}" for c in cvh)}) '
          f'val_all={100*hv.mean():.2f} val_hard={100*hv[va["hard"]].mean():.2f}', flush=True)


if __name__ == '__main__':
    tr, va = load('train'), load('validation')
    tr['cov'], va['cov'] = coverage_features(tr), coverage_features(va)
    # stage-1-only diagnostic for the kNN signal
    s = knn_scores(tr['emb'], tr['y'], va['emb'])
    hv = va['y'][np.arange(len(s)), choose(s)]
    print(f'kNN alone val_hard={100*hv[va["hard"]].mean():.2f}')
    for c in range(5):
        x, yy = va['cov'][va['hard']][:, :, c].ravel(), va['y'][va['hard']].ravel()
        print(f'cov feat {c}: mean_correct={x[yy == 1].mean():.3f} mean_wrong={x[yy == 0].mean():.3f}')
    for v in [(), ('cov',), ('knn',), ('cov', 'knn')]:
        evaluate(tr, va, v)
