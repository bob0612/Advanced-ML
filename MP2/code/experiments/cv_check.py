"""5-fold cross-validation on train for close design decisions (arithmetic features, ensemble size)."""
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_inputs  # noqa: E402
from features import bank_features  # noqa: E402
from model import EmbeddingVerifier, HAND_COLS, choose, make_ranker, stack_features  # noqa: E402
from train import labels, oof_verifier_scores  # noqa: E402

bank, arrays = load_inputs('train')
emb = arrays['embeddings']
X, answers = bank_features(bank, emb)
y = labels(answers, 'train')
hard = np.array([len({a for a in r if a is not None}) >= 2 for r in answers])
ARITH = ['arith_checked', 'arith_wrong', 'arith_any_wrong', 'arith_wrong_frac']
variants = {'full_3seeds': (HAND_COLS, (0, 1, 2)), 'full_1seed': (HAND_COLS, (0,)),
            'no_arith_1seed': ([c for c in HAND_COLS if c not in ARITH], (0,))}
res = {k: [] for k in variants}
res['emb_only'] = []
for a, b in KFold(5, shuffle=True, random_state=1).split(emb):
    oof = oof_verifier_scores(emb[a], y[a])
    p_b = EmbeddingVerifier().fit(emb[a], y[a]).predict_proba(emb[b])
    ans_a = [answers[i] for i in a]; ans_b = [answers[i] for i in b]
    hit = lambda s: y[b][np.arange(len(b)), choose(s)]  # noqa: E731
    res['emb_only'].append(hit(p_b))
    for k, (cols, seeds) in variants.items():
        Sa = stack_features(X[a], oof, ans_a, cols); Sb = stack_features(X[b], p_b, ans_b, cols)
        ms = [make_ranker(s).fit(Sa.reshape(-1, Sa.shape[-1]), y[a].ravel()) for s in seeds]
        s = np.mean([m.predict_proba(Sb.reshape(-1, Sb.shape[-1]))[:, 1] for m in ms], 0).reshape(-1, 4)
        res[k].append(hit(s))
order = np.concatenate([b for _, b in KFold(5, shuffle=True, random_state=1).split(emb)])
for k, v in res.items():
    h = np.zeros(len(y)); h[order] = np.concatenate(v)
    print(f'{k:16s} cv_all={100*h.mean():.2f} cv_hard={100*h[hard].mean():.2f} (n_hard={hard.sum()})')
