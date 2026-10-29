"""Two-stage verifier: embedding logistic regression -> gradient-boosted group ranker.

Stage 1 scores each candidate from its group-centred embedding.
Stage 2 combines that score (and how it compares with the other candidates)
with hand-crafted answer, consensus, surface and arithmetic features.
"""
import json
import pickle

import numpy as np

from features import FEATURE_NAMES

HAND_COLS = [n for n in FEATURE_NAMES if not n.startswith('emb')]
STACK_NAMES = ['emb_logit', 'emb_rel_max', 'emb_rel_mean', 'emb_answer_sum', 'emb_margin_other']


def centred(emb):
    e = np.asarray(emb, dtype=np.float64)
    return e - e.mean(1, keepdims=True)


class EmbeddingVerifier:
    """Standardised logistic regression on group-centred embeddings."""

    def __init__(self, mean=None, scale=None, coef=None, intercept=0.0):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    def fit(self, emb, y, C=0.003, max_iter=3000):
        from sklearn.linear_model import LogisticRegression
        x = centred(emb).reshape(-1, emb.shape[-1])
        self.mean, self.scale = x.mean(0), x.std(0) + 1e-8
        lr = LogisticRegression(C=C, max_iter=max_iter).fit((x - self.mean)/self.scale, y.ravel())
        self.coef, self.intercept = lr.coef_[0], float(lr.intercept_[0])
        return self

    def predict_proba(self, emb):
        x = centred(emb)
        z = ((x - self.mean)/self.scale) @ self.coef + self.intercept
        return 1/(1 + np.exp(-z))

    def state(self):
        return {'mean': self.mean, 'scale': self.scale, 'coef': self.coef,
                'intercept': np.array(self.intercept)}


def group_stats(p, answers):
    """How a candidate's stage-1 score compares with the rest of its group."""
    z = np.log(np.clip(p, 1e-5, 1-1e-5)/np.clip(1-p, 1e-5, 1))
    ans_sum = np.zeros_like(p)
    margin = np.zeros_like(p)
    for g, row in enumerate(answers):
        for i, a in enumerate(row):
            same = [j for j in range(4) if a is not None and row[j] == a]
            other = [j for j in range(4) if not (a is not None and row[j] == a)]
            ans_sum[g, i] = p[g, same].sum() if same else p[g, i]
            margin[g, i] = z[g, i] - (z[g, other].max() if other else -12.0)
    return np.stack([z, z - z.max(1, keepdims=True), z - z.mean(1, keepdims=True), ans_sum, margin], -1)


def stack_features(X, p, answers, hand_cols=None):
    cols = [FEATURE_NAMES.index(n) for n in (HAND_COLS if hand_cols is None else hand_cols)]
    parts = [X[:, :, cols]]
    if p is not None:
        parts.append(group_stats(p, answers))
    return np.concatenate(parts, -1)


def make_ranker(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def choose(scores):
    """Highest score wins; exact ties go to the lowest candidate index."""
    return [int(np.argmax(s + 1e-9*np.arange(4)[::-1])) for s in scores]


def save_bundle(path, verifier, rankers, config):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path/'verifier.npz', **{k: v.astype(np.float32) for k, v in verifier.state().items()})
    with open(path/'rankers.pkl', 'wb') as f:
        pickle.dump(rankers, f, protocol=5)
    (path/'config.json').write_text(json.dumps(config, indent=2)+'\n')


def load_bundle(path):
    with np.load(path/'verifier.npz') as f:
        verifier = EmbeddingVerifier(f['mean'].astype(np.float64), f['scale'].astype(np.float64),
                                     f['coef'].astype(np.float64), float(f['intercept']))
    with open(path/'rankers.pkl', 'rb') as f:
        rankers = pickle.load(f)
    config = json.loads((path/'config.json').read_text())
    return verifier, rankers, config
